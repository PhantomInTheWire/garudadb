use crate::filter::validate_filter;
use crate::filter_parser::parse_filter;
use crate::state::CollectionRuntime;
use garuda_planner::{QueryPlan, SegmentFilterPlan, SegmentSearchPlan, build_query_plan};
use garuda_segment::{
    FlatSearchRequest, HnswSegmentSearchRequest, PersistedSegment, SearchVisibility, SegmentFilter,
    WritingSegment, search_persisted_flat, search_persisted_hnsw, search_writing_flat,
    search_writing_hnsw,
};
use garuda_types::{
    CollectionSchema, DenseVector, Doc, QueryVectorSource, Status, StatusCode, TopK,
    VectorProjection, VectorQuery,
};

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
) -> Result<garuda_types::FilterExpr, Status> {
    let expr = parse_filter(raw_filter)?;
    validate_filter(&expr, schema)?;
    Ok(expr)
}

pub(crate) fn execute_query(
    state: &CollectionRuntime,
    query: VectorQuery,
) -> Result<Vec<Doc>, Status> {
    let filter = parse_query_filter(query.filter.as_deref(), &state.catalog.schema)?;
    let plan = build_query_plan(query, filter, &state.catalog.schema.vector.indexes);

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

        collect_docs_from_persisted_segment(&mut docs, state, plan, query_vector, segment)?;
    }

    if state.segments.writing_segment().meta.doc_count == 0 {
        return Ok(docs);
    }

    collect_docs_from_writing_segment(
        &mut docs,
        state,
        plan,
        query_vector,
        state.segments.writing_segment(),
    )?;

    Ok(docs)
}

fn collect_docs_from_persisted_segment(
    docs: &mut Vec<Doc>,
    state: &CollectionRuntime,
    plan: &QueryPlan,
    query_vector: &DenseVector,
    segment: &PersistedSegment,
) -> Result<(), Status> {
    let filter = segment_filter(&plan.filter);
    let hits = match plan.search {
        SegmentSearchPlan::Flat => search_persisted_flat(
            segment,
            FlatSearchRequest {
                metric: state.catalog.schema.vector.metric,
                query_vector,
                top_k: segment_top_k(segment.meta.doc_count),
                filter,
            },
            SearchVisibility::HideDeleted(state.meta.delete_store()),
        )?,
        SegmentSearchPlan::Hnsw { ef_search } => search_persisted_hnsw(
            segment,
            HnswSegmentSearchRequest {
                query_vector,
                top_k: plan.top_k,
                ef_search,
                filter,
            },
            SearchVisibility::HideDeleted(state.meta.delete_store()),
        )?,
    };

    collect_docs_from_hits(docs, hits);
    Ok(())
}

fn collect_docs_from_writing_segment(
    docs: &mut Vec<Doc>,
    state: &CollectionRuntime,
    plan: &QueryPlan,
    query_vector: &DenseVector,
    segment: &WritingSegment,
) -> Result<(), Status> {
    let filter = segment_filter(&plan.filter);
    let hits = match plan.search {
        SegmentSearchPlan::Flat => search_writing_flat(
            segment,
            FlatSearchRequest {
                metric: state.catalog.schema.vector.metric,
                query_vector,
                top_k: segment_top_k(segment.meta.doc_count),
                filter,
            },
        )?,
        SegmentSearchPlan::Hnsw { ef_search } => search_writing_hnsw(
            segment,
            HnswSegmentSearchRequest {
                query_vector,
                top_k: plan.top_k,
                ef_search,
                filter,
            },
        )?,
    };

    collect_docs_from_hits(docs, hits);
    Ok(())
}

fn collect_docs_from_hits(docs: &mut Vec<Doc>, hits: Vec<garuda_segment::SegmentSearchHit>) {
    for hit in hits {
        let mut doc = hit.record.doc;
        doc.score = Some(hit.score);
        docs.push(doc);
    }
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

fn segment_filter(filter: &SegmentFilterPlan) -> SegmentFilter<'_> {
    match filter {
        SegmentFilterPlan::All => SegmentFilter::All,
        SegmentFilterPlan::Matching(filter) => SegmentFilter::Matching(filter),
    }
}

fn segment_top_k(doc_count: usize) -> TopK {
    TopK::new(doc_count).expect("queryable segment must have at least one live document")
}
