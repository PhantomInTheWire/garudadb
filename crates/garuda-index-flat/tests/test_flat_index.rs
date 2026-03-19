use garuda_index_flat::{FlatIndex, FlatIndexEntry};
use garuda_types::DistanceMetric;

#[test]
fn flat_index_returns_exact_top_k_in_score_order() {
    let index = FlatIndex::build(
        4,
        vec![
            FlatIndexEntry::new(1, vec![1.0, 0.0, 0.0, 0.0]),
            FlatIndexEntry::new(2, vec![0.9, 0.1, 0.0, 0.0]),
            FlatIndexEntry::new(3, vec![0.0, 1.0, 0.0, 0.0]),
        ],
    )
    .expect("build flat index");

    let hits = index
        .search(DistanceMetric::Cosine, &[1.0, 0.0, 0.0, 0.0], 2)
        .expect("search flat index");

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, 1);
    assert_eq!(hits[1].doc_id, 2);
    assert!(hits[0].score >= hits[1].score);
}

#[test]
fn flat_index_breaks_score_ties_by_doc_id() {
    let index = FlatIndex::build(
        2,
        vec![
            FlatIndexEntry::new(4, vec![1.0, 0.0]),
            FlatIndexEntry::new(2, vec![1.0, 0.0]),
            FlatIndexEntry::new(3, vec![0.0, 1.0]),
        ],
    )
    .expect("build flat index");

    let hits = index
        .search(DistanceMetric::InnerProduct, &[1.0, 0.0], 2)
        .expect("search flat index");

    assert_eq!(
        hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
        vec![2, 4]
    );
}

#[test]
fn flat_index_rejects_entries_or_queries_with_the_wrong_dimension() {
    let bad_build = FlatIndex::build(4, vec![FlatIndexEntry::new(1, vec![1.0, 0.0])]);
    assert!(bad_build.is_err());

    let index = FlatIndex::build(4, vec![FlatIndexEntry::new(1, vec![1.0, 0.0, 0.0, 0.0])])
        .expect("build flat index");
    let bad_query = index.search(DistanceMetric::Cosine, &[1.0, 0.0], 1);
    assert!(bad_query.is_err());
}

#[test]
fn flat_index_returns_no_hits_for_zero_top_k() {
    let index = FlatIndex::build(2, vec![FlatIndexEntry::new(1, vec![1.0, 0.0])])
        .expect("build flat index");

    let hits = index
        .search(DistanceMetric::InnerProduct, &[1.0, 0.0], 0)
        .expect("search with zero top_k");

    assert!(hits.is_empty());
}
