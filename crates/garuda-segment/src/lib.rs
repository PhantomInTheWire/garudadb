//! Segment persistence, WAL handling, and search entry points.

mod codec;
mod doc_codec;
mod index;
mod search;
mod storage;
mod types;
mod wal;

pub use garuda_index_flat::WritingFlatIndex;
pub use garuda_index_hnsw::WritingHnswIndex;
pub use search::{search_persisted, search_writing};
pub use storage::{
    doc_exists, ensure_segment_files, read_persisted_segment, read_writing_segment, remove_segment,
    write_persisted_segment, write_writing_segment,
};
pub use types::{
    FlatSearchRequest, HnswSegmentSearchRequest, IvfSegmentSearchRequest, PersistedSegment,
    RecordState, SegmentFilter, SegmentSearchHit, SegmentSearchRequest, StoredRecord,
    WritingSegment, segment_file_name, segment_meta,
};
pub use wal::{WalOp, append_wal_ops, read_wal_ops, reset_wal};
