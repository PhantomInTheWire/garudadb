use crate::segment_manager::SegmentManager;

pub fn optimize_segments(
    segments: &mut SegmentManager,
    next_segment_id: &mut u64,
    segment_max_docs: usize,
) {
    segments.optimize(next_segment_id, segment_max_docs);
}
