use garuda_types::{CollectionSchema, FilterExpr, ScalarType, ScalarValue, Status, StatusCode};

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
