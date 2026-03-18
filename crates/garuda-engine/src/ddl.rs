use crate::state::CollectionState;
use garuda_segment::{StoredRecord, sync_segment_meta};
use garuda_types::{
    FieldName, FlatIndexParams, IndexParams, ScalarFieldSchema, ScalarValue, Status, StatusCode,
};

pub(crate) fn ensure_vector_index_field(
    state: &CollectionState,
    field_name: &FieldName,
) -> Result<(), Status> {
    if *field_name == state.manifest.schema.vector.name {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "cannot create a vector index on a scalar field",
    ))
}

pub(crate) fn set_vector_index_params(state: &mut CollectionState, params: IndexParams) {
    state.manifest.schema.vector.index = params;
}

pub(crate) fn flat_index_params() -> IndexParams {
    IndexParams::Flat(FlatIndexParams)
}

pub(crate) fn ensure_column_can_be_added(
    state: &CollectionState,
    field: &ScalarFieldSchema,
) -> Result<(), Status> {
    if state.manifest.schema.vector.name == field.name {
        return Err(Status::err(
            StatusCode::AlreadyExists,
            "field already exists",
        ));
    }

    if state
        .manifest
        .schema
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

pub(crate) fn backfill_new_column(state: &mut CollectionState, field: &ScalarFieldSchema) {
    let value = field.default_value.clone().unwrap_or(ScalarValue::Null);

    for segment in &mut state.persisted_segments {
        insert_field_into_records(&mut segment.records, field.name.as_str(), &value);
        sync_segment_meta(segment);
    }

    insert_field_into_records(
        &mut state.writing_segment.records,
        field.name.as_str(),
        &value,
    );
    sync_segment_meta(&mut state.writing_segment);
}

pub(crate) fn rename_column_in_schema(
    state: &mut CollectionState,
    old_name: &FieldName,
    new_name: &FieldName,
) -> Result<(), Status> {
    if *old_name == state.manifest.schema.primary_key {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "cannot rename the primary key field",
        ));
    }

    if *new_name == state.manifest.schema.vector.name {
        return Err(Status::err(
            StatusCode::AlreadyExists,
            "field name conflicts with the vector field",
        ));
    }

    if new_name != old_name
        && state
            .manifest
            .schema
            .fields
            .iter()
            .any(|field| field.name == *new_name)
    {
        return Err(Status::err(
            StatusCode::AlreadyExists,
            "field already exists",
        ));
    }

    let Some(field) = state
        .manifest
        .schema
        .fields
        .iter_mut()
        .find(|field| field.name == *old_name)
    else {
        return Err(Status::err(StatusCode::NotFound, "field not found"));
    };

    field.name = new_name.clone();
    Ok(())
}

pub(crate) fn rename_column_in_state(
    state: &mut CollectionState,
    old_name: &FieldName,
    new_name: &FieldName,
) {
    for segment in &mut state.persisted_segments {
        rename_field_in_records(&mut segment.records, old_name.as_str(), new_name.as_str());
        sync_segment_meta(segment);
    }

    rename_field_in_records(
        &mut state.writing_segment.records,
        old_name.as_str(),
        new_name.as_str(),
    );
    sync_segment_meta(&mut state.writing_segment);
}

pub(crate) fn drop_column_from_schema(
    state: &mut CollectionState,
    name: &FieldName,
) -> Result<(), Status> {
    if *name == state.manifest.schema.primary_key {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "cannot drop the primary key field",
        ));
    }

    let before = state.manifest.schema.fields.len();
    state
        .manifest
        .schema
        .fields
        .retain(|field| field.name != *name);

    if before != state.manifest.schema.fields.len() {
        return Ok(());
    }

    Err(Status::err(StatusCode::NotFound, "field not found"))
}

pub(crate) fn drop_column_from_state(state: &mut CollectionState, name: &FieldName) {
    for segment in &mut state.persisted_segments {
        remove_field_from_records(&mut segment.records, name.as_str());
        sync_segment_meta(segment);
    }

    remove_field_from_records(&mut state.writing_segment.records, name.as_str());
    sync_segment_meta(&mut state.writing_segment);
}

fn insert_field_into_records(records: &mut [StoredRecord], field_name: &str, value: &ScalarValue) {
    for record in records {
        record
            .doc
            .fields
            .insert(field_name.to_string(), value.clone());
    }
}

fn rename_field_in_records(records: &mut [StoredRecord], old_name: &str, new_name: &str) {
    for record in records {
        let Some(value) = record.doc.fields.remove(old_name) else {
            continue;
        };

        record.doc.fields.insert(new_name.to_string(), value);
    }
}

fn remove_field_from_records(records: &mut [StoredRecord], field_name: &str) {
    for record in records {
        record.doc.fields.remove(field_name);
    }
}
