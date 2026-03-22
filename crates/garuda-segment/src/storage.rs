use crate::codec::{
    decode_segment, encode_empty_segment, encode_flat_index, encode_hnsw_graph, encode_segment,
};
use crate::index::{
    build_vector_search_state, flat_index_entries, load_vector_search_state, should_persist_hnsw,
    should_use_persisted_flat,
};
use crate::types::{SegmentFile, SegmentKind, sync_segment_meta_fields};
use garuda_storage::{
    create_dir_all, read_file, remove_path_if_exists, segment_data_path, segment_dir,
    segment_flat_index_path, segment_hnsw_index_path, segment_wal_path, write_file_atomically,
};
use garuda_types::{DocId, SegmentId, SegmentMeta, Status, VectorFieldSchema};

pub fn rebuild_search_resources(segment: &mut SegmentFile, vector_field: &VectorFieldSchema) {
    sync_segment_meta_fields(&mut segment.meta, &segment.records);
    let (flat_index, hnsw_index) =
        build_vector_search_state(vector_field, &segment.meta, &segment.records);
    segment.flat_index = flat_index;
    segment.hnsw_index = hnsw_index;
}

pub fn ensure_segment_files(root: &std::path::Path, segment_id: SegmentId) -> Result<(), Status> {
    let segment_dir = segment_dir(root, segment_id);
    create_dir_all(&segment_dir, "failed to create segment directory")?;

    if !segment_data_path(root, segment_id).exists() {
        let bytes = encode_empty_segment(&crate::segment_meta(segment_id))?;
        write_file_atomically(&segment_data_path(root, segment_id), &bytes)?;
    }

    if !segment_wal_path(root, segment_id).exists() {
        crate::reset_wal(root, segment_id)?;
    }

    Ok(())
}

pub fn write_segment(
    root: &std::path::Path,
    segment: &SegmentFile,
    vector_field: &VectorFieldSchema,
) -> Result<(), Status> {
    let bytes = encode_segment(segment)?;
    write_file_atomically(&segment_data_path(root, segment.meta.id), &bytes)?;

    match (&segment.flat_index, &segment.hnsw_index) {
        (Some(_), None) if should_use_persisted_flat(segment, vector_field) => {
            let sidecar = encode_flat_index(
                flat_index_entries(&segment.records, segment.meta.doc_count),
                vector_field,
            )?;
            write_file_atomically(&segment_flat_index_path(root, segment.meta.id), &sidecar)?;
            remove_path_if_exists(&segment_hnsw_index_path(root, segment.meta.id))?;
            return Ok(());
        }
        (None, Some(index)) if should_persist_hnsw(segment) => {
            let sidecar = encode_hnsw_graph(index.graph())?;
            write_file_atomically(&segment_hnsw_index_path(root, segment.meta.id), &sidecar)?;
            remove_path_if_exists(&segment_flat_index_path(root, segment.meta.id))?;
            return Ok(());
        }
        _ => {}
    }

    remove_path_if_exists(&segment_flat_index_path(root, segment.meta.id))?;
    remove_path_if_exists(&segment_hnsw_index_path(root, segment.meta.id))?;
    Ok(())
}

pub fn read_segment(
    root: &std::path::Path,
    meta: &SegmentMeta,
    vector_field: &VectorFieldSchema,
) -> Result<SegmentFile, Status> {
    let bytes = read_file(&segment_data_path(root, meta.id))?;
    let decoded = decode_segment(&bytes)?;
    let kind = SegmentKind::from_segment_id(meta.id);
    let mut segment_meta = decoded.meta;
    sync_segment_meta_fields(&mut segment_meta, &decoded.records);
    segment_meta.path = meta.path.clone();

    let (flat_index, hnsw_index) = if matches!(kind, SegmentKind::Writing) {
        build_vector_search_state(vector_field, &segment_meta, &decoded.records)
    } else {
        load_vector_search_state(root, meta.id, &segment_meta, &decoded.records, vector_field)?
    };

    Ok(SegmentFile {
        kind,
        meta: segment_meta,
        records: decoded.records,
        flat_index,
        hnsw_index,
    })
}
pub fn remove_segment(root: &std::path::Path, segment_id: SegmentId) -> Result<(), Status> {
    remove_path_if_exists(&segment_dir(root, segment_id))
}

pub fn doc_exists(records: &[crate::StoredRecord], id: &DocId) -> bool {
    records
        .iter()
        .any(|record| record.doc.id == *id && matches!(record.state, crate::RecordState::Live))
}
