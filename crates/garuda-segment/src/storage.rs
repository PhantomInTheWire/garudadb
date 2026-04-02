use crate::codec::{
    decode_segment, encode_flat_index, encode_hnsw_graph, encode_ivf_index, encode_scalar_index,
    encode_segment,
};
use crate::index::{
    build_hnsw_index, build_ivf_index, flat_index_entries, indexed_scalar_fields,
    load_persisted_search_resources, persistable_flat_entries_from_writing, should_persist_flat,
    should_persist_hnsw, should_persist_ivf,
};
use crate::types::{PersistedSegment, StoredRecord, WritingSegment, sync_segment_meta_fields};
use crate::{RecordState, reset_wal, segment_meta};
use garuda_index_scalar::ScalarIndex;
use garuda_storage::{
    create_dir_all, read_file, remove_path_if_exists, segment_data_path, segment_dir,
    segment_flat_index_path, segment_hnsw_index_path, segment_ivf_index_path,
    segment_scalar_index_dir, segment_scalar_index_path, segment_wal_path, write_file_atomically,
};
use garuda_types::{CollectionSchema, DocId, FieldName, HnswGraph, SegmentId, SegmentMeta, Status};
use std::collections::BTreeMap;

pub fn ensure_segment_files(root: &std::path::Path, segment_id: SegmentId) -> Result<(), Status> {
    let segment_dir = segment_dir(root, segment_id);
    create_dir_all(&segment_dir, "failed to create segment directory")?;

    if !segment_data_path(root, segment_id).exists() {
        let bytes = encode_segment(&segment_meta(segment_id), &[])?;
        write_file_atomically(&segment_data_path(root, segment_id), &bytes)?;
    }

    if !segment_wal_path(root, segment_id).exists() {
        reset_wal(root, segment_id)?;
    }

    Ok(())
}

pub fn write_persisted_segment(
    root: &std::path::Path,
    segment: &PersistedSegment,
    schema: &CollectionSchema,
) -> Result<(), Status> {
    write_segment_data(root, &segment.meta, &segment.records)?;
    let has_deletes = has_deleted_records(&segment.records);

    write_or_remove_sidecar(
        &segment_flat_index_path(root, segment.meta.id),
        should_persist_flat(schema, segment.meta.doc_count),
        || {
            encode_flat_index(
                flat_index_entries(&segment.records, segment.meta.doc_count),
                &schema.vector,
            )
        },
    )?;

    write_or_remove_sidecar(
        &segment_hnsw_index_path(root, segment.meta.id),
        should_persist_hnsw(schema, segment.meta.doc_count),
        || {
            let existing_graph = segment.hnsw_index.as_ref().map(|index| index.graph());
            hnsw_sidecar_bytes(
                schema,
                &segment.meta,
                &segment.records,
                has_deletes,
                existing_graph,
            )
        },
    )?;

    write_or_remove_sidecar(
        &segment_ivf_index_path(root, segment.meta.id),
        should_persist_ivf(schema, segment.meta.doc_count),
        || {
            if has_deletes {
                let index = build_ivf_index(schema, &segment.meta, &segment.records)
                    .expect("enabled persisted ivf state should build for sidecar");
                return encode_ivf_index(index.stored_lists(), &schema.vector);
            }

            let index = segment
                .ivf_index
                .as_ref()
                .expect("enabled persisted ivf state should exist");
            encode_ivf_index(index.stored_lists(), &schema.vector)
        },
    )?;

    write_scalar_indexes(
        root,
        segment.meta.id,
        &segment.scalar_indexes,
        schema,
        segment.meta.doc_count,
    )?;

    Ok(())
}

pub fn write_writing_segment(
    root: &std::path::Path,
    segment: &WritingSegment,
    schema: &CollectionSchema,
) -> Result<(), Status> {
    write_segment_data(root, &segment.meta, &segment.records)?;
    let has_deletes = has_deleted_records(&segment.records);

    write_or_remove_sidecar(
        &segment_flat_index_path(root, segment.meta.id),
        should_persist_flat(schema, segment.meta.doc_count),
        || {
            encode_flat_index(
                persistable_flat_entries_from_writing(segment),
                &schema.vector,
            )
        },
    )?;

    write_or_remove_sidecar(
        &segment_hnsw_index_path(root, segment.meta.id),
        should_persist_hnsw(schema, segment.meta.doc_count),
        || {
            let existing_graph = segment.hnsw_index.as_ref().map(|index| index.graph());
            hnsw_sidecar_bytes(
                schema,
                &segment.meta,
                &segment.records,
                has_deletes,
                existing_graph,
            )
        },
    )?;

    remove_path_if_exists(&segment_ivf_index_path(root, segment.meta.id))?;

    write_scalar_indexes(
        root,
        segment.meta.id,
        &segment.scalar_indexes,
        schema,
        segment.meta.doc_count,
    )?;

    Ok(())
}

pub fn read_writing_segment(
    root: &std::path::Path,
    meta: &SegmentMeta,
    schema: &CollectionSchema,
) -> Result<WritingSegment, Status> {
    let bytes = read_file(&segment_data_path(root, meta.id))?;
    let decoded = decode_segment(&bytes)?;
    let mut segment_meta = decoded.meta;
    sync_segment_meta_fields(&mut segment_meta, &decoded.records);
    segment_meta.path = meta.path.clone();
    Ok(WritingSegment::new(segment_meta, decoded.records, schema))
}

pub fn read_persisted_segment(
    root: &std::path::Path,
    meta: &SegmentMeta,
    schema: &CollectionSchema,
) -> Result<PersistedSegment, Status> {
    let bytes = read_file(&segment_data_path(root, meta.id))?;
    let decoded = decode_segment(&bytes)?;
    let mut segment_meta = decoded.meta;
    sync_segment_meta_fields(&mut segment_meta, &decoded.records);
    segment_meta.path = meta.path.clone();
    let resources =
        load_persisted_search_resources(root, meta.id, schema, &segment_meta, &decoded.records)?;

    Ok(PersistedSegment {
        meta: segment_meta,
        records: decoded.records,
        flat_index: resources.flat_index,
        hnsw_index: resources.hnsw_index,
        ivf_index: resources.ivf_index,
        scalar_indexes: resources.scalar_indexes,
    })
}

pub fn remove_segment(root: &std::path::Path, segment_id: SegmentId) -> Result<(), Status> {
    remove_path_if_exists(&segment_dir(root, segment_id))
}

pub fn doc_exists(records: &[StoredRecord], id: &DocId) -> bool {
    records
        .iter()
        .any(|record| record.doc.id == *id && matches!(record.state, RecordState::Live))
}

fn write_scalar_indexes(
    root: &std::path::Path,
    segment_id: SegmentId,
    scalar_indexes: &BTreeMap<FieldName, ScalarIndex>,
    schema: &CollectionSchema,
    live_doc_count: usize,
) -> Result<(), Status> {
    let scalar_dir = segment_scalar_index_dir(root, segment_id);
    remove_path_if_exists(&scalar_dir)?;

    if live_doc_count == 0 {
        return Ok(());
    }

    create_dir_all(&scalar_dir, "failed to create scalar index directory")?;

    for field in indexed_scalar_fields(schema) {
        let index = scalar_indexes
            .get(&field.name)
            .expect("enabled scalar index should exist");
        let bytes = encode_scalar_index(index, live_doc_count)?;
        write_file_atomically(
            &segment_scalar_index_path(root, segment_id, &field.name),
            &bytes,
        )?;
    }

    Ok(())
}

fn write_segment_data(
    root: &std::path::Path,
    meta: &SegmentMeta,
    records: &[StoredRecord],
) -> Result<(), Status> {
    let bytes = encode_segment(meta, records)?;
    write_file_atomically(&segment_data_path(root, meta.id), &bytes)
}

fn write_or_remove_sidecar(
    path: &std::path::Path,
    should_persist: bool,
    build: impl FnOnce() -> Result<Vec<u8>, Status>,
) -> Result<(), Status> {
    if !should_persist {
        return remove_path_if_exists(path);
    }

    let bytes = build()?;
    write_file_atomically(path, &bytes)
}

fn hnsw_sidecar_bytes(
    schema: &CollectionSchema,
    meta: &SegmentMeta,
    records: &[StoredRecord],
    has_deletes: bool,
    existing_graph: Option<&HnswGraph>,
) -> Result<Vec<u8>, Status> {
    if has_deletes {
        let index = build_hnsw_index(schema, meta, records)
            .expect("enabled hnsw state should build for sidecar");
        return encode_hnsw_graph(index.graph());
    }

    let existing_graph = existing_graph.expect("enabled hnsw state should exist when no deletes");
    encode_hnsw_graph(existing_graph)
}

fn has_deleted_records(records: &[StoredRecord]) -> bool {
    records
        .iter()
        .any(|record| matches!(record.state, RecordState::Deleted))
}
