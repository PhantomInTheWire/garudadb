use garuda_types::{
    CollectionSchema, FieldName, IndexKind, IndexParams, Nullability, ScalarFieldSchema,
    ScalarIndexState, Status, StatusCode,
};

pub(crate) fn create_index(
    schema: &mut CollectionSchema,
    field_name: &FieldName,
    params: IndexParams,
) -> Result<(), Status> {
    if *field_name == schema.vector.name {
        return create_vector_index(schema, params);
    }

    create_scalar_index(schema, field_name, params)
}

pub(crate) fn drop_index(
    schema: &mut CollectionSchema,
    field_name: &FieldName,
    kind: IndexKind,
) -> Result<(), Status> {
    if *field_name == schema.vector.name {
        return drop_vector_index(schema, kind);
    }

    drop_scalar_index(schema, field_name, kind)
}

fn create_vector_index(schema: &mut CollectionSchema, params: IndexParams) -> Result<(), Status> {
    schema.vector.indexes = match params {
        IndexParams::Flat(_) => schema.vector.indexes.clone().enable_flat(),
        IndexParams::Hnsw(params) => {
            params.neighbor_config()?;
            schema.vector.indexes.clone().enable_hnsw(params)
        }
        IndexParams::Scalar(_) => {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "cannot create a scalar index on the vector field",
            ));
        }
    };

    Ok(())
}

fn drop_vector_index(schema: &mut CollectionSchema, kind: IndexKind) -> Result<(), Status> {
    schema.vector.indexes = match kind {
        IndexKind::Flat | IndexKind::Hnsw => schema.vector.indexes.clone().drop(kind),
        IndexKind::Scalar => {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "cannot drop a scalar index from the vector field",
            ));
        }
    };

    Ok(())
}

fn create_scalar_index(
    schema: &mut CollectionSchema,
    field_name: &FieldName,
    params: IndexParams,
) -> Result<(), Status> {
    let Some(field) = schema.scalar_field_mut(field_name) else {
        return Err(Status::err(StatusCode::NotFound, "field not found"));
    };

    match params {
        IndexParams::Scalar(_) => {
            ensure_scalar_field_can_be_indexed(field)?;
            field.index = ScalarIndexState::Indexed;
            Ok(())
        }
        IndexParams::Flat(_) | IndexParams::Hnsw(_) => Err(Status::err(
            StatusCode::InvalidArgument,
            "cannot create a vector index on a scalar field",
        )),
    }
}

fn ensure_scalar_field_can_be_indexed(field: &ScalarFieldSchema) -> Result<(), Status> {
    if matches!(field.nullability, Nullability::Required) {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "indexed scalar fields cannot be nullable",
    ))
}

fn drop_scalar_index(
    schema: &mut CollectionSchema,
    field_name: &FieldName,
    kind: IndexKind,
) -> Result<(), Status> {
    if kind != IndexKind::Scalar {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "cannot drop a vector index from a scalar field",
        ));
    }

    let Some(field) = schema.scalar_field_mut(field_name) else {
        return Err(Status::err(StatusCode::NotFound, "field not found"));
    };
    field.index = ScalarIndexState::None;
    Ok(())
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

    if schema
        .fields
        .iter()
        .any(|existing| existing.name == field.name)
    {
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

    let Some(field) = schema.scalar_field_mut(old_name) else {
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
