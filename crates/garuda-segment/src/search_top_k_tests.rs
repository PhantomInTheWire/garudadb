use crate::search::flat_candidate_top_k;
use crate::types::{SegmentFilter, SegmentFilterContext};
use garuda_meta::DeleteStore;
use garuda_types::{FilterExpr, ScalarValue, TopK};

fn top_k(value: usize) -> TopK {
    TopK::new(value).expect("top_k")
}

#[test]
fn flat_candidate_top_k_should_keep_requested_top_k_without_filtering() {
    assert_eq!(
        flat_candidate_top_k(
            top_k(10),
            SegmentFilterContext {
                allowed_doc_ids: None,
                delete_store: Some(&DeleteStore::new()),
                residual: SegmentFilter::All,
            },
            100_000,
            100_000,
        ),
        top_k(10)
    );
}

#[test]
fn flat_candidate_top_k_should_widen_for_residual_filtering() {
    let filter = FilterExpr::Eq(
        "category".to_string(),
        ScalarValue::String("alpha".to_string()),
    );

    assert_eq!(
        flat_candidate_top_k(
            top_k(10),
            SegmentFilterContext {
                allowed_doc_ids: None,
                delete_store: Some(&DeleteStore::new()),
                residual: SegmentFilter::Matching(&filter),
            },
            100,
            80,
        ),
        top_k(80)
    );
}

#[test]
fn flat_candidate_top_k_should_widen_for_visible_deletes() {
    let mut delete_store = DeleteStore::new();
    delete_store.insert(garuda_types::InternalDocId::new(1).expect("doc id"));

    assert_eq!(
        flat_candidate_top_k(
            top_k(10),
            SegmentFilterContext {
                allowed_doc_ids: None,
                delete_store: Some(&delete_store),
                residual: SegmentFilter::All,
            },
            100,
            80,
        ),
        top_k(80)
    );
}
