use garuda_types::{
    CollectionSchema, Doc, ScalarFieldSchema, ScalarType, ScalarValue, Status, StatusCode,
};

pub fn validate_doc(schema: &CollectionSchema, doc: &Doc) -> Result<(), Status> {
    validate_required_vector(schema, &doc.vector)?;
    validate_doc_fields(schema, doc)
}

pub fn apply_schema_defaults(schema: &CollectionSchema, doc: &mut Doc) {
    for field in &schema.fields {
        if doc.fields.contains_key(field.name.as_str()) {
            continue;
        }

        let Some(default_value) = &field.default_value else {
            continue;
        };

        doc.fields
            .insert(field.name.to_string(), default_value.clone());
    }
}

pub fn validate_field_default(field: &ScalarFieldSchema) -> Result<(), Status> {
    let Some(default_value) = &field.default_value else {
        if field.nullable {
            return Ok(());
        }

        return Err(Status::err(
            StatusCode::InvalidArgument,
            "non-nullable column requires a default value",
        ));
    };

    validate_scalar_value(field.field_type, field.nullable, default_value)
}

fn validate_required_vector(schema: &CollectionSchema, vector: &[f32]) -> Result<(), Status> {
    if vector.len() != schema.vector.dimension {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "vector dimension does not match schema",
        ));
    }

    Ok(())
}

fn validate_doc_fields(schema: &CollectionSchema, doc: &Doc) -> Result<(), Status> {
    for field in &schema.fields {
        let Some(value) = doc.fields.get(field.name.as_str()) else {
            if field.nullable {
                continue;
            }

            return Err(Status::err(
                StatusCode::InvalidArgument,
                format!("missing required field: {}", field.name.as_str()),
            ));
        };

        validate_scalar_value(field.field_type, field.nullable, value)?;
    }

    Ok(())
}

fn validate_scalar_value(
    expected: ScalarType,
    nullable: bool,
    value: &ScalarValue,
) -> Result<(), Status> {
    if matches!(value, ScalarValue::Null) {
        if nullable {
            return Ok(());
        }

        return Err(Status::err(
            StatusCode::InvalidArgument,
            "non-nullable field cannot be null",
        ));
    }

    let valid = matches!(
        (expected, value),
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
        "scalar field type does not match schema",
    ))
}
