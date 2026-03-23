use crate::filter::validate_filter;
use crate::filter_parser::parse_filter;
use crate::state::CollectionRuntime;
use garuda_index_scalar::ScalarIndex;
use garuda_index_scalar::prefilter_doc_ids;
use garuda_meta::DeleteStore;
use garuda_planner::{QueryPlan, SegmentFilterPlan, SegmentSearchPlan, build_query_plan};
use garuda_segment::{
    FlatSearchRequest, HnswSegmentSearchRequest, PersistedSegment, SegmentFilter, SegmentSearchHit,
    SegmentSearchRequest, WritingSegment, search_persisted, search_writing,
};
use garuda_types::{
    CollectionSchema, DenseVector, Doc, FieldName, FilterExpr, InternalDocId, QueryVectorSource,
    Status, StatusCode, VectorProjection, VectorQuery,
};
use std::collections::BTreeMap;
use std::collections::HashSet;

#[derive(Clone, Copy)]
enum QuerySegment<'a> {
    Persisted(&'a PersistedSegment, &'a DeleteStore),
    Writing(&'a WritingSegment),
}

impl<'a> QuerySegment<'a> {
    fn doc_count(self) -> usize {
        match self {
            Self::Persisted(segment, _) => segment.meta.doc_count,
            Self::Writing(segment) => segment.meta.doc_count,
        }
    }

    fn scalar_indexes(self) -> &'a BTreeMap<FieldName, ScalarIndex> {
        match self {
            Self::Persisted(segment, _) => &segment.scalar_indexes,
            Self::Writing(segment) => &segment.scalar_indexes,
        }
    }

    fn search(
        self,
        request: SegmentSearchRequest<'a>,
        allowed_doc_ids: Option<&HashSet<InternalDocId>>,
    ) -> Result<Vec<SegmentSearchHit>, Status> {
        match self {
            Self::Persisted(segment, delete_store) => {
                search_persisted(segment, request, allowed_doc_ids, delete_store)
            }
            Self::Writing(segment) => search_writing(segment, request, allowed_doc_ids),
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
    let filter = parse_query_filter(query.filter.as_deref(), &state.catalog.schema)?;
    let plan = build_query_plan(query, filter, &state.catalog.schema);

    if plan.field_name != state.catalog.schema.vector.name {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "unknown vector field",
        ));
    }

    let query_vector = resolve_query_vector(&plan, state)?;
    let docs = collect_matching_docs(state, &plan, &query_vector)?;
    Ok(finalize_docs(docs, &plan))
}

fn apply_query_projection(doc: &mut Doc, plan: &QueryPlan) {
    if matches!(plan.vector_projection, VectorProjection::Exclude) {
        doc.vector = DenseVector::default();
    }

    let Some(output_fields) = &plan.output_fields else {
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
            if vector.len() != state.catalog.schema.vector.dimension.get() {
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
        prefilter_doc_ids(&plan.filter.scalar_prefilter, segment.scalar_indexes());
    let request = segment_request(state, plan, query_vector, segment.doc_count());
    let hits = if matches!(allowed_doc_ids, Some(ref doc_ids) if doc_ids.is_empty()) {
        Vec::new()
    } else {
        segment.search(request, allowed_doc_ids.as_ref())?
    };

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
    segment_doc_count: usize,
) -> SegmentSearchRequest<'a> {
    let filter = match &plan.filter.residual {
        SegmentFilterPlan::All => SegmentFilter::All,
        SegmentFilterPlan::Matching(filter) => SegmentFilter::Matching(filter),
    };

    match plan.search {
        SegmentSearchPlan::Flat => SegmentSearchRequest::Flat(FlatSearchRequest {
            metric: state.catalog.schema.vector.metric,
            query_vector,
            top_k: segment_top_k(segment_doc_count),
            filter,
        }),
        SegmentSearchPlan::Hnsw { ef_search } => {
            SegmentSearchRequest::Hnsw(HnswSegmentSearchRequest {
                query_vector,
                top_k: plan.top_k,
                ef_search,
                filter,
            })
        }
    }
}

fn segment_top_k(doc_count: usize) -> garuda_types::TopK {
    garuda_types::TopK::new(doc_count)
        .expect("queryable segment must have at least one live document")
}

fn finalize_docs(mut docs: Vec<Doc>, plan: &QueryPlan) -> Vec<Doc> {
    docs.sort_by(|lhs, rhs| {
        rhs.score
            .unwrap_or_default()
            .total_cmp(&lhs.score.unwrap_or_default())
            .then_with(|| lhs.id.cmp(&rhs.id))
    });
    docs.truncate(plan.top_k.get());

    for doc in &mut docs {
        apply_query_projection(doc, plan);
    }

    docs
}
