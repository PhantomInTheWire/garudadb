use garuda_index_ivf::{
    IvfBuildEntry, IvfIndex, IvfIndexConfig, IvfSearchRequest, IvfStoredLists, WritingIvfIndex,
};
use garuda_types::{
    DenseVector, DistanceMetric, InternalDocId, IvfIndexParams, IvfListCount, IvfProbeCount,
    IvfTrainingIterations, StatusCode, TopK, VectorDimension,
};

fn config() -> IvfIndexConfig {
    IvfIndexConfig::new(
        VectorDimension::new(2).expect("dimension"),
        DistanceMetric::L2,
        IvfIndexParams {
            n_list: IvfListCount::new(2).expect("list count"),
            n_probe: IvfProbeCount::new(1).expect("probe count"),
            training_iterations: IvfTrainingIterations::new(4).expect("iterations"),
        },
    )
}

fn config_with_iterations(iterations: u32) -> IvfIndexConfig {
    IvfIndexConfig::new(
        VectorDimension::new(2).expect("dimension"),
        DistanceMetric::L2,
        IvfIndexParams {
            n_list: IvfListCount::new(2).expect("list count"),
            n_probe: IvfProbeCount::new(1).expect("probe count"),
            training_iterations: IvfTrainingIterations::new(iterations).expect("iterations"),
        },
    )
}

fn entry(doc_id: u64, vector: [f32; 2]) -> IvfBuildEntry {
    IvfBuildEntry::new(
        VectorDimension::new(2).expect("dimension"),
        InternalDocId::new(doc_id).expect("doc id"),
        DenseVector::parse(vector.to_vec()).expect("vector"),
    )
    .expect("entry")
}

fn vector(values: [f32; 2]) -> DenseVector {
    DenseVector::parse(values.to_vec()).expect("vector")
}

#[test]
fn search_rejects_dimension_mismatch() {
    let index = IvfIndex::build(config(), vec![entry(1, [0.0, 0.0]), entry(2, [10.0, 10.0])]);

    let error = index
        .search(IvfSearchRequest::new(
            &DenseVector::parse(vec![1.0]).expect("query"),
            TopK::new(1).expect("top k"),
            IvfProbeCount::new(1).expect("probe count"),
        ))
        .expect_err("dimension mismatch");

    assert_eq!(error.code, StatusCode::InvalidArgument);
}

#[test]
fn wider_nprobe_should_not_reduce_recall() {
    let index = IvfIndex::build(
        config(),
        vec![
            entry(1, [0.0, 0.0]),
            entry(2, [0.1, 0.0]),
            entry(3, [10.0, 10.0]),
            entry(4, [10.1, 10.0]),
        ],
    );
    let query = vector([0.05, 0.0]);
    let narrow = index.search(IvfSearchRequest::new(
        &query,
        TopK::new(2).expect("top k"),
        IvfProbeCount::new(1).expect("probe count"),
    ));
    let wide = index.search(IvfSearchRequest::new(
        &query,
        TopK::new(2).expect("top k"),
        IvfProbeCount::new(2).expect("probe count"),
    ));

    let narrow = narrow.expect("narrow search");
    let wide = wide.expect("wide search");

    assert!(wide.len() >= narrow.len());
    assert!(
        wide.iter()
            .any(|hit| hit.doc_id == InternalDocId::new(1).expect("doc id"))
    );
    assert!(
        wide.iter()
            .any(|hit| hit.doc_id == InternalDocId::new(2).expect("doc id"))
    );
}

#[test]
fn build_should_use_deterministic_farthest_initializer() {
    let index = IvfIndex::build(
        config_with_iterations(1),
        vec![
            entry(1, [0.0, 0.0]),
            entry(2, [1.0, 0.0]),
            entry(3, [10.0, 0.0]),
            entry(4, [11.0, 0.0]),
        ],
    );
    let lists = index.stored_lists();

    assert_eq!(lists.centroids.len(), 2);
    assert_eq!(lists.doc_ids_by_list.len(), 2);
    assert_eq!(lists.doc_ids_by_list[0].len(), 2);
    assert_eq!(lists.doc_ids_by_list[1].len(), 2);
}

#[test]
fn build_should_allow_empty_inputs() {
    let index = IvfIndex::build(config(), Vec::new());

    assert!(index.is_empty());
    assert_eq!(index.list_count(), 0);
    assert!(index.stored_lists().centroids.is_empty());
    assert!(index.stored_lists().doc_ids_by_list.is_empty());
}

#[test]
fn writing_index_insert_should_keep_existing_hits_stable() {
    let mut index = WritingIvfIndex::from_entries_incremental(
        config(),
        vec![
            entry(1, [0.0, 0.0]),
            entry(2, [0.0, 1.0]),
            entry(3, [10.0, 10.0]),
            entry(4, [10.0, 11.0]),
        ],
    );
    let query = vector([0.0, 0.1]);
    let before = index
        .search(IvfSearchRequest::new(
            &query,
            TopK::new(2).expect("top k"),
            IvfProbeCount::new(2).expect("probe count"),
        ))
        .expect("before search");

    index.insert(entry(5, [0.1, 0.2]));

    let after = index
        .search(IvfSearchRequest::new(
            &query,
            TopK::new(3).expect("top k"),
            IvfProbeCount::new(2).expect("probe count"),
        ))
        .expect("after search");

    assert_eq!(before[0].doc_id, InternalDocId::new(1).expect("doc id"));
    assert_eq!(before[1].doc_id, InternalDocId::new(2).expect("doc id"));
    assert_eq!(after[0].doc_id, InternalDocId::new(1).expect("doc id"));
    assert!(
        after
            .iter()
            .any(|hit| hit.doc_id == InternalDocId::new(5).expect("doc id"))
    );
}

#[test]
fn from_parts_rejects_duplicate_doc_ids() {
    let config = config();
    let entries = vec![entry(1, [0.0, 0.0]), entry(2, [10.0, 10.0])];
    let duplicate_doc_id = InternalDocId::new(1).expect("doc id");
    let stored = IvfStoredLists {
        centroids: vec![vector([0.0, 0.0]), vector([10.0, 10.0])],
        doc_ids_by_list: vec![vec![duplicate_doc_id], vec![duplicate_doc_id]],
    };

    let error = IvfIndex::from_parts(config, entries, stored).expect_err("duplicate doc id");

    assert_eq!(error.code, StatusCode::Internal);
}

#[test]
fn from_parts_rejects_missing_doc_ids() {
    let config = config();
    let entries = vec![entry(1, [0.0, 0.0]), entry(2, [10.0, 10.0])];
    let stored = IvfStoredLists {
        centroids: vec![vector([0.0, 0.0]), vector([10.0, 10.0])],
        doc_ids_by_list: vec![vec![InternalDocId::new(1).expect("doc id")], vec![]],
    };

    let error = IvfIndex::from_parts(config, entries, stored).expect_err("missing doc id");

    assert_eq!(error.code, StatusCode::Internal);
}
