use crate::filter::validate_filter;
use crate::filter_parser::parse_filter;
use crate::state::CollectionRuntime;
use garuda_index_scalar::ScalarIndex;
use garuda_index_scalar::prefilter_doc_ids;
use garuda_meta::DeleteStore;
use garuda_planner::{
    FilterPlan, PrefilterPlan, ProjectionPlan, QueryPlan, ResidualPlan, build_query_plan,
};
use garuda_segment::{
    PersistedSegment, SegmentExecutionRequest, SegmentFilter, SegmentFilterContext,
    SegmentSearchHit, WritingSegment, search_persisted, search_writing,
};
use garuda_types::{
    CollectionSchema, DenseVector, Doc, FieldName, FilterExpr, InternalDocId, QueryVectorSource,
    RecallPlan, Status, StatusCode, VectorProjection, VectorQuery,
};
use std::collections::BTreeMap;
use std::collections::HashSet;

#[derive(Clone, Copy)]
enum QuerySegment<'a> {
    Persisted(&'a PersistedSegment, &'a DeleteStore),
    Writing(&'a WritingSegment),
}

impl<'a> QuerySegment<'a> {
    fn scalar_indexes(self) -> &'a BTreeMap<FieldName, ScalarIndex> {
        match self {
            Self::Persisted(segment, _) => &segment.scalar_indexes,
            Self::Writing(segment) => &segment.scalar_indexes,
        }
    }

    fn search(self, request: SegmentExecutionRequest<'a>) -> Result<Vec<SegmentSearchHit>, Status> {
        match self {
            Self::Persisted(segment, _) => search_persisted(segment, request),
            Self::Writing(segment) => search_writing(segment, request),
        }
    }

    fn delete_store(self) -> Option<&'a DeleteStore> {
        match self {
            Self::Persisted(_, delete_store) => Some(delete_store),
            Self::Writing(_) => None,
        }
    }
}

pub(crate) fn parse_query_filter(
    raw_filter: Option<&str>,
    schema: &CollectionSchema,
) -> Result<Option<garuda_types::FilterExpr>, Status> {
    let Some(raw_filter) = raw_filter else {
        return Ok(None);
    };

    Ok(Some(parse_required_filter(raw_filter, schema)?))
}

pub(crate) fn parse_required_filter(
    raw_filter: &str,
    schema: &CollectionSchema,
) -> Result<FilterExpr, Status> {
    let expr = parse_filter(raw_filter)?;
    validate_filter(&expr, schema)?;
    Ok(expr)
}

pub(crate) fn execute_query(
    state: &CollectionRuntime,
    query: VectorQuery,
) -> Result<Vec<Doc>, Status> {
    let filter = parse_query_filter(query.filter.as_deref(), &state.schema)?;
    let plan = build_query_plan(query, filter, &state.schema)?;

    if plan.field_name != state.schema.vector.name {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "unknown vector field",
        ));
    }

    let query_vector = resolve_query_vector(&plan, state)?;
    let docs = collect_matching_docs(state, &plan, &query_vector)?;
    Ok(finalize_docs(docs, &plan))
}

fn apply_query_projection(doc: &mut Doc, projection: &ProjectionPlan) {
    if matches!(projection.vector, VectorProjection::Exclude) {
        doc.vector = DenseVector::default();
    }

    let Some(output_fields) = &projection.output_fields else {
        return;
    };

    doc.fields
        .retain(|field_name, _| output_fields.iter().any(|field| field == field_name));
}

fn resolve_query_vector(
    plan: &QueryPlan,
    state: &CollectionRuntime,
) -> Result<DenseVector, Status> {
    match &plan.source {
        QueryVectorSource::Vector(vector) => {
            if vector.len() != state.schema.vector.dimension.get() {
                return Err(Status::err(
                    StatusCode::InvalidArgument,
                    "query vector dimension does not match schema",
                ));
            }

            Ok(vector.clone())
        }
        QueryVectorSource::DocumentId(id) => {
            let Some(record) = state.record(id) else {
                return Err(Status::err(
                    StatusCode::NotFound,
                    "query document id not found",
                ));
            };

            Ok(record.doc.vector.clone())
        }
    }
}

fn collect_matching_docs(
    state: &CollectionRuntime,
    plan: &QueryPlan,
    query_vector: &DenseVector,
) -> Result<Vec<Doc>, Status> {
    let mut docs = Vec::new();

    for segment in state.segments.persisted_segments() {
        if segment.meta.doc_count == 0 {
            continue;
        }

        collect_docs_from_segment(
            &mut docs,
            state,
            plan,
            query_vector,
            QuerySegment::Persisted(segment, state.meta.delete_store()),
        )?;
    }

    if state.segments.writing_segment().meta.doc_count == 0 {
        return Ok(docs);
    }

    collect_docs_from_segment(
        &mut docs,
        state,
        plan,
        query_vector,
        QuerySegment::Writing(state.segments.writing_segment()),
    )?;

    Ok(docs)
}

fn collect_docs_from_segment(
    docs: &mut Vec<Doc>,
    state: &CollectionRuntime,
    plan: &QueryPlan,
    query_vector: &DenseVector,
    segment: QuerySegment<'_>,
) -> Result<(), Status> {
    let allowed_doc_ids =
        prefilter_doc_ids(prefilter_predicates(&plan.filter), segment.scalar_indexes());
    if matches!(allowed_doc_ids, Some(ref doc_ids) if doc_ids.is_empty()) {
        return Ok(());
    }

    let request = segment_request(state, plan, query_vector, segment, allowed_doc_ids.as_ref());
    let hits = segment.search(request)?;

    for hit in hits {
        let mut doc = hit.record.doc;
        doc.score = Some(hit.score);
        docs.push(doc);
    }

    Ok(())
}

fn segment_request<'a>(
    state: &CollectionRuntime,
    plan: &'a QueryPlan,
    query_vector: &'a DenseVector,
    segment: QuerySegment<'a>,
    allowed_doc_ids: Option<&'a HashSet<InternalDocId>>,
) -> SegmentExecutionRequest<'a> {
    SegmentExecutionRequest {
        query_vector,
        metric: state.schema.vector.metric,
        recall: plan.recall,
        filter: SegmentFilterContext {
            allowed_doc_ids,
            delete_store: segment.delete_store(),
            residual: residual_filter(&plan.filter),
        },
    }
}

fn finalize_docs(mut docs: Vec<Doc>, plan: &QueryPlan) -> Vec<Doc> {
    docs.sort_by(|lhs, rhs| {
        rhs.score
            .unwrap_or_default()
            .total_cmp(&lhs.score.unwrap_or_default())
            .then_with(|| lhs.id.cmp(&rhs.id))
    });
    docs.truncate(plan_top_k(plan).get());

    for doc in &mut docs {
        apply_query_projection(doc, &plan.projection);
    }

    docs
}

fn plan_top_k(plan: &QueryPlan) -> garuda_types::TopK {
    match plan.recall {
        RecallPlan::Flat(recall) => recall.top_k,
        RecallPlan::Hnsw(recall) => recall.top_k,
        RecallPlan::Ivf(recall) => recall.top_k,
    }
}

fn prefilter_predicates(filter: &FilterPlan) -> Option<&[garuda_types::ScalarPredicate]> {
    match &filter.prefilter {
        PrefilterPlan::All => None,
        PrefilterPlan::ScalarAnd(predicates) => Some(predicates),
    }
}

fn residual_filter(filter: &FilterPlan) -> SegmentFilter<'_> {
    match &filter.residual {
        ResidualPlan::All => SegmentFilter::All,
        ResidualPlan::Filter(filter) => SegmentFilter::Matching(filter),
    }
}
