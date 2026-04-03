use garuda_index_flat::{FlatIndex, FlatIndexEntry};
use garuda_types::{DenseVector, DistanceMetric, InternalDocId, TopK, VectorDimension};

#[test]
fn flat_index_returns_exact_top_k_in_score_order() {
    let index = FlatIndex::build(
        VectorDimension::new(4).expect("valid dimension"),
        vec![
            FlatIndexEntry::new(
                InternalDocId::new(1).expect("valid doc id"),
                DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            ),
            FlatIndexEntry::new(
                InternalDocId::new(2).expect("valid doc id"),
                DenseVector::parse(vec![0.9, 0.1, 0.0, 0.0]).expect("valid vector"),
            ),
            FlatIndexEntry::new(
                InternalDocId::new(3).expect("valid doc id"),
                DenseVector::parse(vec![0.0, 1.0, 0.0, 0.0]).expect("valid vector"),
            ),
        ],
    )
    .expect("build flat index");

    let hits = index
        .search(
            DistanceMetric::Cosine,
            &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            TopK::new(2).expect("valid top_k"),
        )
        .expect("search flat index");

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, InternalDocId::new(1).expect("valid doc id"));
    assert_eq!(hits[1].doc_id, InternalDocId::new(2).expect("valid doc id"));
    assert!(hits[0].score >= hits[1].score);
}

#[test]
fn flat_index_keeps_all_hits_tied_at_the_cutoff_score() {
    let index = FlatIndex::build(
        VectorDimension::new(2).expect("valid dimension"),
        vec![
            FlatIndexEntry::new(
                InternalDocId::new(4).expect("valid doc id"),
                DenseVector::parse(vec![1.0, 0.0]).expect("valid vector"),
            ),
            FlatIndexEntry::new(
                InternalDocId::new(2).expect("valid doc id"),
                DenseVector::parse(vec![1.0, 0.0]).expect("valid vector"),
            ),
            FlatIndexEntry::new(
                InternalDocId::new(3).expect("valid doc id"),
                DenseVector::parse(vec![0.0, 1.0]).expect("valid vector"),
            ),
        ],
    )
    .expect("build flat index");

    let hits = index
        .search(
            DistanceMetric::InnerProduct,
            &DenseVector::parse(vec![1.0, 0.0]).expect("valid vector"),
            TopK::new(1).expect("valid top_k"),
        )
        .expect("search flat index");

    assert_eq!(hits.len(), 2);
    assert!(hits.contains(&garuda_index_flat::FlatSearchHit {
        doc_id: InternalDocId::new(2).expect("valid doc id"),
        score: 1.0,
    }));
    assert!(hits.contains(&garuda_index_flat::FlatSearchHit {
        doc_id: InternalDocId::new(4).expect("valid doc id"),
        score: 1.0,
    }));
}

#[test]
fn flat_index_rejects_entries_or_queries_with_the_wrong_dimension() {
    let bad_build = FlatIndex::build(
        VectorDimension::new(4).expect("valid dimension"),
        vec![FlatIndexEntry::new(
            InternalDocId::new(1).expect("valid doc id"),
            DenseVector::parse(vec![1.0, 0.0]).expect("valid vector"),
        )],
    );
    assert!(bad_build.is_err());

    let index = FlatIndex::build(
        VectorDimension::new(4).expect("valid dimension"),
        vec![FlatIndexEntry::new(
            InternalDocId::new(1).expect("valid doc id"),
            DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
        )],
    )
    .expect("build flat index");
    let bad_query = index.search(
        DistanceMetric::Cosine,
        &DenseVector::parse(vec![1.0, 0.0]).expect("valid vector"),
        TopK::new(1).expect("valid top_k"),
    );
    assert!(bad_query.is_err());
}

#[test]
fn top_k_rejects_zero() {
    assert!(TopK::new(0).is_err());
}
