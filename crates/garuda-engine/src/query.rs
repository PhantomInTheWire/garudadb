use crate::filter::validate_filter;
use crate::filter_parser::parse_filter;
use crate::state::CollectionRuntime;
use garuda_planner::{QueryPlan, SegmentScanMode, build_query_plan};
use garuda_segment::exact_search;
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
    let plan = build_query_plan(query, filter);

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

fn resolve_query_vector(plan: &QueryPlan, state: &CollectionRuntime) -> Result<DenseVector, Status> {
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
    let filter = required_filter(plan)?;
    let metric = state.catalog.schema.vector.metric;
    let mut docs = Vec::new();

    for segment in state.segments.persisted_segments() {
        if segment.meta.doc_count == 0 {
            continue;
        }

        collect_docs_from_segment(
            &mut docs,
            segment,
            metric,
            query_vector,
            TopK::new(segment.meta.doc_count).expect("segment doc count must be positive"),
            filter,
        )?;
    }

    if state.segments.writing_segment().meta.doc_count == 0 {
        return Ok(docs);
    }

    collect_docs_from_segment(
        &mut docs,
        state.segments.writing_segment(),
        metric,
        query_vector,
        TopK::new(state.segments.writing_segment().meta.doc_count)
            .expect("segment doc count must be positive"),
        filter,
    )?;

    Ok(docs)
}

fn collect_docs_from_segment(
    docs: &mut Vec<Doc>,
    segment: &garuda_segment::SegmentFile,
    metric: garuda_types::DistanceMetric,
    query_vector: &DenseVector,
    top_k: TopK,
    filter: Option<&garuda_types::FilterExpr>,
) -> Result<(), Status> {
    let hits = exact_search(segment, metric, query_vector, top_k, filter)?;

    for hit in hits {
        let mut doc = hit.record.doc;
        doc.score = Some(hit.score);
        docs.push(doc);
    }

    Ok(())
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

fn required_filter(plan: &QueryPlan) -> Result<Option<&garuda_types::FilterExpr>, Status> {
    if matches!(plan.scan_mode, SegmentScanMode::FullScan) {
        return Ok(None);
    }

    let Some(filter) = plan.filter.as_ref() else {
        return Err(Status::err(
            StatusCode::Internal,
            "filtered scans must carry a filter",
        ));
    };

    Ok(Some(filter))
}
