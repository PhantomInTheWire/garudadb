use std::cmp::Ordering;

// sort by score descending, then by doc_id ascending
pub(crate) fn compare_score_then_doc_id<T: Ord>(
    left_score: f32,
    left_doc_id: T,
    right_score: f32,
    right_doc_id: T,
) -> Ordering {
    right_score
        .total_cmp(&left_score)
        .then_with(|| left_doc_id.cmp(&right_doc_id))
}
