use garuda_types::{
    CollectionSchema, FieldName, FlatIndexParams, IndexParams, ScalarFieldSchema, Status,
    StatusCode,
};

pub(crate) fn ensure_vector_index_field(
    schema: &CollectionSchema,
    field_name: &FieldName,
) -> Result<(), Status> {
    if *field_name == schema.vector.name {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "cannot create a vector index on a scalar field",
    ))
}

pub(crate) fn set_vector_index_params(schema: &mut CollectionSchema, params: IndexParams) {
    schema.vector.index = params;
}

pub(crate) fn flat_index_params() -> IndexParams {
    IndexParams::Flat(FlatIndexParams)
}

pub(crate) fn ensure_column_can_be_added(
    schema: &CollectionSchema,
    field: &ScalarFieldSchema,
) -> Result<(), Status> {
    if schema.vector.name == field.name {
        return Err(Status::err(
            StatusCode::AlreadyExists,
            "field already exists",
        ));
    }

    if schema.fields.iter().any(|existing| existing.name == field.name) {
        return Err(Status::err(
            StatusCode::AlreadyExists,
            "field already exists",
        ));
    }

    Ok(())
}

pub(crate) fn rename_column(
    schema: &mut CollectionSchema,
    old_name: &FieldName,
    new_name: &FieldName,
) -> Result<(), Status> {
    if *old_name == schema.primary_key {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "cannot rename the primary key field",
        ));
    }

    if *new_name == schema.vector.name {
        return Err(Status::err(
            StatusCode::AlreadyExists,
            "field name conflicts with the vector field",
        ));
    }

    if new_name != old_name && schema.fields.iter().any(|field| field.name == *new_name) {
        return Err(Status::err(
            StatusCode::AlreadyExists,
            "field already exists",
        ));
    }

    let Some(field) = schema.fields.iter_mut().find(|field| field.name == *old_name) else {
        return Err(Status::err(StatusCode::NotFound, "field not found"));
    };

    field.name = new_name.clone();
    Ok(())
}

pub(crate) fn drop_column(schema: &mut CollectionSchema, name: &FieldName) -> Result<(), Status> {
    if *name == schema.primary_key {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "cannot drop the primary key field",
        ));
    }

    let before = schema.fields.len();
    schema.fields.retain(|field| field.name != *name);

    if before != schema.fields.len() {
        return Ok(());
    }

    Err(Status::err(StatusCode::NotFound, "field not found"))
}
