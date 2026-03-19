use std::borrow::Cow;
use crate::filter::{parse_filter, validate_filter};
use crate::state::CollectionRuntime;
use garuda_math::score_doc;
use garuda_meta::evaluate_filter;
use garuda_planner::{QueryPlan, SegmentScanMode, build_query_plan};
use garuda_types::{CollectionSchema, Doc, QueryVectorSource, Status, StatusCode, VectorQuery};

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
    ensure_query_uses_known_vector_field(state, &plan)?;

    if plan.top_k == 0 {
        return Ok(Vec::new());
    }

    let query_vector = resolve_query_vector(&plan, state)?;
    let docs = collect_matching_docs(state, &plan)?;

    Ok(score_and_sort_docs(state, docs, &query_vector, &plan))
}

fn apply_query_projection(doc: &mut Doc, plan: &QueryPlan) {
    if !plan.include_vector {
        doc.vector.clear();
    }

    let Some(output_fields) = &plan.output_fields else {
        return;
    };

    doc.fields
        .retain(|field_name, _| output_fields.iter().any(|field| field == field_name));
}

fn resolve_query_vector<'a>(
    plan: &'a QueryPlan,
    state: &'a CollectionRuntime,
) -> Result<Cow<'a, [f32]>, Status> {
    match &plan.source {
        QueryVectorSource::Vector(vector) => {
            if vector.len() != state.catalog.schema.vector.dimension {
                return Err(Status::err(
                    StatusCode::InvalidArgument,
                    "query vector dimension does not match schema",
                ));
            }

            Ok(Cow::Borrowed(vector.as_slice()))
        }
        QueryVectorSource::DocumentId(id) => {
            let Some(record) = state.find_live_record(id) else {
                return Err(Status::err(
                    StatusCode::NotFound,
                    "query document id not found",
                ));
            };

            Ok(Cow::Borrowed(record.doc.vector.as_slice()))
        }
    }
}

fn ensure_query_uses_known_vector_field(
    state: &CollectionRuntime,
    plan: &QueryPlan,
) -> Result<(), Status> {
    if plan.field_name == state.catalog.schema.vector.name {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "unknown vector field",
    ))
}

fn collect_matching_docs(state: &CollectionRuntime, plan: &QueryPlan) -> Result<Vec<Doc>, Status> {
    let mut docs = state.all_live_docs();

    if matches!(plan.scan_mode, SegmentScanMode::FilteredScan) {
        let Some(filter) = plan.filter.as_ref() else {
            return Err(Status::err(
                StatusCode::Internal,
                "filtered scans must carry a filter",
            ));
        };
        docs.retain(|doc| evaluate_filter(filter, &doc.fields));
    }

    Ok(docs)
}

fn score_and_sort_docs(
    state: &CollectionRuntime,
    docs: Vec<Doc>,
    query_vector: &[f32],
    plan: &QueryPlan,
) -> Vec<Doc> {
    let mut scored_docs = Vec::with_capacity(docs.len());

    for mut doc in docs {
        doc.score = Some(score_doc(
            state.catalog.schema.vector.metric,
            query_vector,
            &doc.vector,
        ));
        apply_query_projection(&mut doc, plan);
        scored_docs.push(doc);
    }

    scored_docs.sort_by(|lhs, rhs| {
        rhs.score
            .unwrap_or_default()
            .total_cmp(&lhs.score.unwrap_or_default())
            .then_with(|| lhs.id.cmp(&rhs.id))
    });
    scored_docs.truncate(plan.top_k);

    scored_docs
}
