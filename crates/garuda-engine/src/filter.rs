use garuda_types::{
    CollectionSchema, FilterExpr, ScalarFieldSchema, ScalarType, ScalarValue, Status, StatusCode,
};

pub fn validate_filter(expr: &FilterExpr, schema: &CollectionSchema) -> Result<(), Status> {
    match expr {
        FilterExpr::Eq(field, value)
        | FilterExpr::Ne(field, value)
        | FilterExpr::Gt(field, value)
        | FilterExpr::Gte(field, value)
        | FilterExpr::Lt(field, value)
        | FilterExpr::Lte(field, value) => {
            validate_filter_value(field_schema(schema, field)?.field_type, value)
        }
        FilterExpr::StringMatch(field, _) => validate_string_field(field_schema(schema, field)?),
        FilterExpr::IsNull(field) => {
            field_schema(schema, field)?;
            Ok(())
        }
        FilterExpr::And(lhs, rhs) | FilterExpr::Or(lhs, rhs) => {
            validate_filter(lhs, schema)?;
            validate_filter(rhs, schema)?;
            Ok(())
        }
    }
}

fn field_schema<'a>(
    schema: &'a CollectionSchema,
    field: &str,
) -> Result<&'a ScalarFieldSchema, Status> {
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

    Ok(field_schema)
}

fn validate_string_field(field_schema: &ScalarFieldSchema) -> Result<(), Status> {
    if matches!(field_schema.field_type, ScalarType::String) {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "string filter operator requires a string field",
    ))
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
