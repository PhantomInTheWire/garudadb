use garuda_types::{
    CollectionOptions, CollectionSchema, FieldName, ScalarFieldSchema, ScalarType, ScalarValue,
    Status, StatusCode,
};
use std::collections::HashSet;

pub(crate) fn validate_create_options(options: &CollectionOptions) -> Result<(), Status> {
    if options.read_only {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "cannot create a collection in read-only mode",
        ));
    }

    if options.segment_max_docs == 0 {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "segment_max_docs must be greater than zero",
        ));
    }

    Ok(())
}

pub(crate) fn validate_schema(schema: &CollectionSchema) -> Result<(), Status> {
    validate_vector_dimension(schema)?;
    validate_primary_key(schema)?;

    let mut seen: HashSet<FieldName> = HashSet::new();
    for field in &schema.fields {
        validate_unique_field_name(&mut seen, field)?;
        validate_declared_default(field)?;
    }

    if seen.contains(&schema.vector.name) {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "vector field name conflicts with scalar field name",
        ));
    }

    Ok(())
}

fn validate_vector_dimension(schema: &CollectionSchema) -> Result<(), Status> {
    if schema.vector.dimension > 0 {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "vector dimension must be greater than zero",
    ))
}

fn validate_primary_key(schema: &CollectionSchema) -> Result<(), Status> {
    if schema
        .fields
        .iter()
        .any(|field| field.name == schema.primary_key)
    {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "primary key must reference an existing scalar field",
    ))
}

fn validate_unique_field_name(
    seen: &mut HashSet<FieldName>,
    field: &ScalarFieldSchema,
) -> Result<(), Status> {
    if seen.insert(field.name.clone()) {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "duplicate scalar field name in schema",
    ))
}

fn validate_declared_default(field: &ScalarFieldSchema) -> Result<(), Status> {
    let Some(default_value) = &field.default_value else {
        return Ok(());
    };

    validate_scalar_value(field.field_type, field.nullable, default_value)
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
