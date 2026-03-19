use crate::filter::{parse_filter, validate_filter};
use crate::state::CollectionState;
use garuda_types::{CollectionSchema, Doc, Status, StatusCode, VectorQuery};
use std::collections::BTreeMap;

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

pub(crate) fn apply_query_projection(doc: &mut Doc, query: &VectorQuery) {
    if !query.include_vector {
        doc.vector.clear();
    }

    let Some(output_fields) = &query.output_fields else {
        return;
    };

    let mut filtered_fields = BTreeMap::new();
    for field in output_fields {
        let Some(value) = doc.fields.get(field) else {
            continue;
        };

        filtered_fields.insert(field.clone(), value.clone());
    }

    doc.fields = filtered_fields;
}

pub(crate) fn resolve_query_vector(
    query: &VectorQuery,
    state: &CollectionState,
) -> Result<Vec<f32>, Status> {
    if let Some(vector) = &query.vector {
        if vector.len() != state.manifest.schema.vector.dimension {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "query vector dimension does not match schema",
            ));
        }

        return Ok(vector.as_slice().to_vec());
    }

    let Some(id) = &query.id else {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "query must provide either a vector or a document id",
        ));
    };

    let Some(record) = state.find_live_record(id) else {
        return Err(Status::err(
            StatusCode::NotFound,
            "query document id not found",
        ));
    };

    Ok(record.doc.vector.clone())
}
