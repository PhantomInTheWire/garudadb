use crate::codec::{
    decode_segment, encode_empty_segment, encode_flat_index, encode_hnsw_graph, encode_segment,
};
use crate::index::{
    flat_index_entries, into_persisted_segment, load_persisted_search_resources,
    persistable_flat_entries_from_writing, should_persist_flat, should_persist_hnsw,
};
use crate::types::{PersistedSegment, StoredRecord, WritingSegment, sync_segment_meta_fields};
use crate::{RecordState, reset_wal, segment_meta};
use garuda_storage::{
    create_dir_all, read_file, remove_path_if_exists, segment_data_path, segment_dir,
    segment_flat_index_path, segment_hnsw_index_path, segment_wal_path, write_file_atomically,
};
use garuda_types::{DocId, SegmentId, SegmentMeta, Status, VectorFieldSchema};

pub fn ensure_segment_files(root: &std::path::Path, segment_id: SegmentId) -> Result<(), Status> {
    let segment_dir = segment_dir(root, segment_id);
    create_dir_all(&segment_dir, "failed to create segment directory")?;

    if !segment_data_path(root, segment_id).exists() {
        let bytes = encode_empty_segment(&segment_meta(segment_id))?;
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
    vector_field: &VectorFieldSchema,
) -> Result<(), Status> {
    let bytes = encode_segment(&segment.meta, &segment.records)?;
    write_file_atomically(&segment_data_path(root, segment.meta.id), &bytes)?;

    if should_persist_flat(vector_field, segment.meta.doc_count) {
        let sidecar = encode_flat_index(
            flat_index_entries(&segment.records, segment.meta.doc_count),
            vector_field,
        )?;
        write_file_atomically(&segment_flat_index_path(root, segment.meta.id), &sidecar)?;
    } else {
        remove_path_if_exists(&segment_flat_index_path(root, segment.meta.id))?;
    }

    if should_persist_hnsw(vector_field, segment.meta.doc_count) {
        let index = segment
            .hnsw_index
            .as_ref()
            .expect("enabled persisted hnsw state should exist");
        let sidecar = encode_hnsw_graph(index.graph())?;
        write_file_atomically(&segment_hnsw_index_path(root, segment.meta.id), &sidecar)?;
    } else {
        remove_path_if_exists(&segment_hnsw_index_path(root, segment.meta.id))?;
    }

    Ok(())
}

pub fn write_writing_segment(
    root: &std::path::Path,
    segment: &WritingSegment,
    vector_field: &VectorFieldSchema,
) -> Result<(), Status> {
    let bytes = encode_segment(&segment.meta, &segment.records)?;
    write_file_atomically(&segment_data_path(root, segment.meta.id), &bytes)?;

    if should_persist_flat(vector_field, segment.meta.doc_count) {
        let sidecar =
            encode_flat_index(persistable_flat_entries_from_writing(segment), vector_field)?;
        write_file_atomically(&segment_flat_index_path(root, segment.meta.id), &sidecar)?;
    } else {
        remove_path_if_exists(&segment_flat_index_path(root, segment.meta.id))?;
    }

    if should_persist_hnsw(vector_field, segment.meta.doc_count) {
        let index = segment
            .hnsw_index
            .as_ref()
            .expect("enabled writing hnsw state should exist");
        let sidecar = encode_hnsw_graph(index.graph())?;
        write_file_atomically(&segment_hnsw_index_path(root, segment.meta.id), &sidecar)?;
    } else {
        remove_path_if_exists(&segment_hnsw_index_path(root, segment.meta.id))?;
    }

    Ok(())
}

pub fn read_writing_segment(
    root: &std::path::Path,
    meta: &SegmentMeta,
    vector_field: &VectorFieldSchema,
) -> Result<WritingSegment, Status> {
    let bytes = read_file(&segment_data_path(root, meta.id))?;
    let decoded = decode_segment(&bytes)?;
    let mut segment_meta = decoded.meta;
    sync_segment_meta_fields(&mut segment_meta, &decoded.records);
    segment_meta.path = meta.path.clone();
    Ok(WritingSegment::new(
        segment_meta,
        decoded.records,
        vector_field,
    ))
}

pub fn read_persisted_segment(
    root: &std::path::Path,
    meta: &SegmentMeta,
    vector_field: &VectorFieldSchema,
) -> Result<PersistedSegment, Status> {
    let bytes = read_file(&segment_data_path(root, meta.id))?;
    let decoded = decode_segment(&bytes)?;
    let mut segment_meta = decoded.meta;
    sync_segment_meta_fields(&mut segment_meta, &decoded.records);
    segment_meta.path = meta.path.clone();
    let (flat_index, hnsw_index) = load_persisted_search_resources(
        root,
        meta.id,
        &segment_meta,
        &decoded.records,
        vector_field,
    )?;

    Ok(PersistedSegment {
        meta: segment_meta,
        records: decoded.records,
        flat_index,
        hnsw_index,
    })
}

pub fn seal_writing_segment(
    segment: WritingSegment,
    segment_id: SegmentId,
    vector_field: &VectorFieldSchema,
) -> PersistedSegment {
    into_persisted_segment(segment, segment_id, vector_field)
}

pub fn remove_segment(root: &std::path::Path, segment_id: SegmentId) -> Result<(), Status> {
    remove_path_if_exists(&segment_dir(root, segment_id))
}

pub fn doc_exists(records: &[StoredRecord], id: &DocId) -> bool {
    records
        .iter()
        .any(|record| record.doc.id == *id && matches!(record.state, RecordState::Live))
}
