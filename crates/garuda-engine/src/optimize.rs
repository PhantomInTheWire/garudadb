use crate::state::CollectionState;
use garuda_segment::{SegmentFile, segment_file_name, sync_segment_meta};

pub fn optimize_segments(state: &mut CollectionState) {
    let all_live_records = state.all_live_records();
    let mut rebuilt_segments = Vec::new();
    let mut current_segment = CollectionState::empty_writing_segment();

    for record in all_live_records {
        if current_segment.records.len() >= state.manifest.options.segment_max_docs {
            seal_segment(state, &mut rebuilt_segments, current_segment);
            current_segment = CollectionState::empty_writing_segment();
        }

        current_segment.records.push(record);
    }

    if current_segment.records.is_empty() {
        state.persisted_segments = rebuilt_segments;
        state.writing_segment = CollectionState::empty_writing_segment();
        return;
    }

    seal_segment(state, &mut rebuilt_segments, current_segment);
    state.persisted_segments = rebuilt_segments;
    state.writing_segment = CollectionState::empty_writing_segment();
}

fn seal_segment(
    state: &mut CollectionState,
    rebuilt_segments: &mut Vec<SegmentFile>,
    mut segment: SegmentFile,
) {
    sync_segment_meta(&mut segment);
    segment.meta.id = state.manifest.next_segment_id;
    segment.meta.path = segment_file_name(state.manifest.next_segment_id);
    state.manifest.next_segment_id += 1;
    rebuilt_segments.push(segment);
}
