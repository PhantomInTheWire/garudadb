mod codec;
mod doc_codec;
mod index;
mod search;
mod storage;
mod types;
mod wal;

pub use garuda_index_flat::WritingFlatIndex;
pub use garuda_index_hnsw::WritingHnswIndex;
pub use search::{
    SearchVisibility, search_persisted_flat, search_persisted_hnsw, search_writing_flat,
    search_writing_hnsw,
};
pub use storage::{
    doc_exists, ensure_segment_files, read_persisted_segment, read_writing_segment, remove_segment,
    seal_writing_segment, write_persisted_segment, write_writing_segment,
};
pub use types::{
    FlatSearchRequest, HnswSegmentSearchRequest, PersistedSegment, RecordState, SegmentFilter,
    SegmentSearchHit, StoredRecord, WritingSegment, segment_file_name, segment_meta,
};
pub use wal::{WalOp, append_wal_ops, read_wal_ops, reset_wal};
