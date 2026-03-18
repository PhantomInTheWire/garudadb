use garuda_types::{CollectionSchema, FilterExpr, ScalarType, ScalarValue, Status, StatusCode};
use std::cmp::Ordering;
use std::collections::BTreeMap;

pub fn parse_filter(input: &str) -> Result<FilterExpr, Status> {
    crate::filter_parser::parse_filter(input)
}

pub fn validate_filter(expr: &FilterExpr, schema: &CollectionSchema) -> Result<(), Status> {
    match expr {
        FilterExpr::Eq(field, value)
        | FilterExpr::Ne(field, value)
        | FilterExpr::Gt(field, value)
        | FilterExpr::Gte(field, value)
        | FilterExpr::Lt(field, value)
        | FilterExpr::Lte(field, value) => validate_filter_leaf(field, value, schema),
        FilterExpr::And(lhs, rhs) | FilterExpr::Or(lhs, rhs) => {
            validate_filter(lhs, schema)?;
            validate_filter(rhs, schema)?;
            Ok(())
        }
    }
}

pub fn evaluate_filter(expr: &FilterExpr, fields: &BTreeMap<String, ScalarValue>) -> bool {
    match expr {
        FilterExpr::Eq(field, value) => fields.get(field) == Some(value),
        FilterExpr::Ne(field, value) => fields
            .get(field)
            .is_some_and(|candidate| candidate != value),
        FilterExpr::Gt(field, value) => compare_field_value(fields, field, value, Ordering::is_gt),
        FilterExpr::Gte(field, value) => compare_field_value(fields, field, value, Ordering::is_ge),
        FilterExpr::Lt(field, value) => compare_field_value(fields, field, value, Ordering::is_lt),
        FilterExpr::Lte(field, value) => compare_field_value(fields, field, value, Ordering::is_le),
        FilterExpr::And(lhs, rhs) => evaluate_filter(lhs, fields) && evaluate_filter(rhs, fields),
        FilterExpr::Or(lhs, rhs) => evaluate_filter(lhs, fields) || evaluate_filter(rhs, fields),
    }
}

fn validate_filter_leaf(
    field: &str,
    value: &ScalarValue,
    schema: &CollectionSchema,
) -> Result<(), Status> {
    let Some(field_schema) = schema
        .fields
        .iter()
        .find(|candidate| candidate.name.as_str() == field)
    else {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            format!("unknown filter field: {field}"),
        ));
    };

    validate_filter_value(field_schema.field_type, value)
}

fn compare_field_value(
    fields: &BTreeMap<String, ScalarValue>,
    field: &str,
    value: &ScalarValue,
    predicate: impl FnOnce(Ordering) -> bool,
) -> bool {
    let Some(candidate) = fields.get(field) else {
        return false;
    };

    let Some(ordering) = compare_values(candidate, value) else {
        return false;
    };

    predicate(ordering)
}

fn compare_values(lhs: &ScalarValue, rhs: &ScalarValue) -> Option<Ordering> {
    match (lhs, rhs) {
        (ScalarValue::Bool(a), ScalarValue::Bool(b)) => Some(a.cmp(b)),
        (ScalarValue::Int64(a), ScalarValue::Int64(b)) => Some(a.cmp(b)),
        (ScalarValue::Float64(a), ScalarValue::Float64(b)) => a.partial_cmp(b),
        (ScalarValue::String(a), ScalarValue::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

fn validate_filter_value(expected_type: ScalarType, value: &ScalarValue) -> Result<(), Status> {
    let valid = matches!(
        (expected_type, value),
        (ScalarType::Bool, ScalarValue::Bool(_))
            | (ScalarType::Int64, ScalarValue::Int64(_))
            | (ScalarType::Float64, ScalarValue::Float64(_))
            | (ScalarType::String, ScalarValue::String(_))
    );

    if valid {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "filter literal type does not match field type",
    ))
}
