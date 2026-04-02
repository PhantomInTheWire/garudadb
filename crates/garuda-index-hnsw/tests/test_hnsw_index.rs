use garuda_index_hnsw::{
    HnswBuildConfig, HnswBuildEntry, HnswIndex, HnswIndexConfig, HnswSearchRequest,
};
use garuda_types::{
    DenseVector, DistanceMetric, HnswEfConstruction, HnswEfSearch, HnswGraph, HnswLevel, HnswM,
    HnswMinNeighborCount, HnswNeighborConfig, HnswNeighborLimits, HnswPruneWidth,
    HnswScalingFactor, InternalDocId, NodeIndex, RemoveResult, TopK, VectorDimension,
};

#[test]
fn graph_from_parts_rejects_malformed_level_shape() {
    let result = HnswGraph::from_parts(
        vec![HnswLevel::new(0), HnswLevel::new(0)],
        vec![vec![Vec::<NodeIndex>::new()]],
        2,
        HnswNeighborLimits::new(HnswM::new(16).expect("valid hnsw max_neighbors")),
    );

    assert!(result.is_err());
}

#[test]
fn graph_from_parts_rejects_graph_without_top_level_node() {
    let result = HnswGraph::from_parts(
        vec![HnswLevel::new(0)],
        vec![vec![Vec::<NodeIndex>::new()], vec![Vec::<NodeIndex>::new()]],
        1,
        HnswNeighborLimits::new(HnswM::new(16).expect("valid hnsw max_neighbors")),
    );

    assert!(result.is_err());
}

#[test]
fn graph_from_parts_rejects_level_count_that_does_not_match_node_levels() {
    let result = HnswGraph::from_parts(
        vec![HnswLevel::new(1), HnswLevel::new(0)],
        vec![vec![Vec::<NodeIndex>::new(), Vec::<NodeIndex>::new()]],
        2,
        HnswNeighborLimits::new(HnswM::new(16).expect("valid hnsw max_neighbors")),
    );

    assert!(result.is_err());
}

#[test]
fn graph_from_parts_allows_level_zero_to_use_twice_m_neighbors() {
    let result = HnswGraph::from_parts(
        vec![HnswLevel::new(0), HnswLevel::new(0), HnswLevel::new(0)],
        vec![vec![
            vec![NodeIndex::new(1), NodeIndex::new(2)],
            Vec::new(),
            Vec::new(),
        ]],
        3,
        HnswNeighborLimits::new(HnswM::new(1).expect("valid hnsw max_neighbors")),
    );

    assert!(result.is_ok());
}

#[test]
fn graph_from_parts_rejects_level_zero_neighbor_count_above_twice_m() {
    let result = HnswGraph::from_parts(
        vec![
            HnswLevel::new(0),
            HnswLevel::new(0),
            HnswLevel::new(0),
            HnswLevel::new(0),
        ],
        vec![vec![
            vec![NodeIndex::new(1), NodeIndex::new(2), NodeIndex::new(3)],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ]],
        4,
        HnswNeighborLimits::new(HnswM::new(1).expect("valid hnsw max_neighbors")),
    );

    assert!(result.is_err());
}

#[test]
fn graph_from_parts_rejects_out_of_bounds_neighbor() {
    let result = HnswGraph::from_parts(
        vec![HnswLevel::new(0), HnswLevel::new(0)],
        vec![vec![vec![NodeIndex::new(2)], Vec::new()]],
        2,
        HnswNeighborLimits::new(HnswM::new(16).expect("valid hnsw max_neighbors")),
    );

    assert!(result.is_err());
}

#[test]
fn graph_from_parts_rejects_edges_above_node_level() {
    let result = HnswGraph::from_parts(
        vec![HnswLevel::new(0), HnswLevel::new(1)],
        vec![
            vec![Vec::new(), Vec::new()],
            vec![vec![NodeIndex::new(1)], Vec::new()],
        ],
        2,
        HnswNeighborLimits::new(HnswM::new(16).expect("valid hnsw max_neighbors")),
    );

    assert!(result.is_err());
}

#[test]
fn graph_from_parts_rejects_neighbor_that_does_not_exist_on_level() {
    let result = HnswGraph::from_parts(
        vec![HnswLevel::new(1), HnswLevel::new(0)],
        vec![
            vec![Vec::new(), Vec::new()],
            vec![vec![NodeIndex::new(1)], Vec::new()],
        ],
        2,
        HnswNeighborLimits::new(HnswM::new(16).expect("valid hnsw max_neighbors")),
    );

    assert!(result.is_err());
}

#[test]
fn search_returns_scored_hits_in_deterministic_order() {
    let config = HnswIndexConfig::new(
        VectorDimension::new(2).unwrap(),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(16).unwrap(),
                HnswMinNeighborCount::new(8).unwrap(),
            )
            .unwrap(),
            HnswScalingFactor::new(2).unwrap(),
            HnswEfConstruction::new(200).unwrap(),
            HnswPruneWidth::new(64).unwrap(),
        ),
    );
    let entries = vec![
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(2).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(1).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
    ];
    let index = HnswIndex::build(config, entries);

    let hits = index
        .search(HnswSearchRequest::new(
            &DenseVector::parse(vec![1.0, 0.0]).unwrap(),
            TopK::new(2).unwrap(),
            HnswEfSearch::new(32).unwrap(),
        ))
        .unwrap();

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, InternalDocId::new(1).unwrap());
    assert_eq!(hits[1].doc_id, InternalDocId::new(2).unwrap());
}

#[test]
fn search_keeps_lower_doc_id_when_top_k_and_ef_search_are_one() {
    let config = HnswIndexConfig::new(
        VectorDimension::new(2).unwrap(),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(2).unwrap(),
                HnswMinNeighborCount::new(1).unwrap(),
            )
            .unwrap(),
            HnswScalingFactor::new(2).unwrap(),
            HnswEfConstruction::new(8).unwrap(),
            HnswPruneWidth::new(8).unwrap(),
        ),
    );
    let entries = vec![
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(1).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(2).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
    ];
    let index = HnswIndex::build(config, entries);

    let hits = index
        .search(HnswSearchRequest::new(
            &DenseVector::parse(vec![1.0, 0.0]).unwrap(),
            TopK::new(1).unwrap(),
            HnswEfSearch::new(1).unwrap(),
        ))
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, InternalDocId::new(1).unwrap());
}

#[test]
fn build_connects_new_node_to_best_previous_neighbor() {
    let config = HnswIndexConfig::new(
        VectorDimension::new(2).unwrap(),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(2).unwrap(),
                HnswMinNeighborCount::new(1).unwrap(),
            )
            .unwrap(),
            HnswScalingFactor::new(2).unwrap(),
            HnswEfConstruction::new(200).unwrap(),
            HnswPruneWidth::new(64).unwrap(),
        ),
    );
    let entries = vec![
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(1).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(2).unwrap(),
            DenseVector::parse(vec![0.0, 1.0]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(3).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
    ];
    let index = HnswIndex::build(config, entries);

    assert_eq!(
        index
            .graph()
            .neighbors(HnswLevel::new(0), NodeIndex::new(2))[0],
        NodeIndex::new(0)
    );
}

#[test]
fn build_replaces_weaker_reverse_neighbors_when_node_is_full() {
    let config = HnswIndexConfig::new(
        VectorDimension::new(2).unwrap(),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(1).unwrap(),
                HnswMinNeighborCount::new(1).unwrap(),
            )
            .unwrap(),
            HnswScalingFactor::new(2).unwrap(),
            HnswEfConstruction::new(200).unwrap(),
            HnswPruneWidth::new(64).unwrap(),
        ),
    );
    let entries = vec![
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(2).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(1).unwrap(),
            DenseVector::parse(vec![0.0, 1.0]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(4).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
    ];
    let index = HnswIndex::build(config, entries);

    assert_eq!(
        index
            .graph()
            .neighbors(HnswLevel::new(0), NodeIndex::new(0)),
        &[NodeIndex::new(2)]
    );
}

#[test]
fn build_prunes_redundant_neighbors_when_min_neighbor_count_allows_it() {
    let config = HnswIndexConfig::new(
        VectorDimension::new(2).unwrap(),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(2).unwrap(),
                HnswMinNeighborCount::new(1).unwrap(),
            )
            .unwrap(),
            HnswScalingFactor::new(50).unwrap(),
            HnswEfConstruction::new(200).unwrap(),
            HnswPruneWidth::new(4).unwrap(),
        ),
    );
    let entries = vec![
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(1).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(2).unwrap(),
            DenseVector::parse(vec![0.9, 0.1]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(3).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
    ];
    let index = HnswIndex::build(config, entries);

    assert_eq!(
        index
            .graph()
            .neighbors(HnswLevel::new(0), NodeIndex::new(2)),
        &[NodeIndex::new(0)]
    );
}

#[test]
fn build_respects_prune_width_when_selecting_neighbors() {
    let config = HnswIndexConfig::new(
        VectorDimension::new(2).unwrap(),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(2).unwrap(),
                HnswMinNeighborCount::new(1).unwrap(),
            )
            .unwrap(),
            HnswScalingFactor::new(50).unwrap(),
            HnswEfConstruction::new(200).unwrap(),
            HnswPruneWidth::new(1).unwrap(),
        ),
    );
    let entries = vec![
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(1).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(2).unwrap(),
            DenseVector::parse(vec![0.8, 0.2]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(3).unwrap(),
            DenseVector::parse(vec![0.7, 0.3]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(4).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
    ];
    let index = HnswIndex::build(config, entries);

    assert_eq!(
        index
            .graph()
            .neighbors(HnswLevel::new(0), NodeIndex::new(3)),
        &[NodeIndex::new(0)]
    );
}

#[test]
fn remove_should_hide_deleted_doc_from_search_results() {
    let config = HnswIndexConfig::new(
        VectorDimension::new(2).expect("dimension"),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(2).expect("max neighbors"),
                HnswMinNeighborCount::new(1).expect("min neighbors"),
            )
            .expect("neighbor config"),
            HnswScalingFactor::new(2).expect("scaling factor"),
            HnswEfConstruction::new(8).expect("ef construction"),
            HnswPruneWidth::new(8).expect("prune width"),
        ),
    );

    let mut index = HnswIndex::build(
        config.clone(),
        vec![
            HnswBuildEntry::new(
                &config,
                InternalDocId::new(1).expect("doc id"),
                DenseVector::parse(vec![1.0, 0.0]).expect("vector"),
            )
            .expect("entry"),
            HnswBuildEntry::new(
                &config,
                InternalDocId::new(2).expect("doc id"),
                DenseVector::parse(vec![0.0, 1.0]).expect("vector"),
            )
            .expect("entry"),
        ],
    );

    assert_eq!(
        index.remove(InternalDocId::new(1).expect("doc id")),
        RemoveResult::Removed
    );
    assert_eq!(
        index.remove(InternalDocId::new(1).expect("doc id")),
        RemoveResult::Missing
    );

    let hits = index
        .search(HnswSearchRequest::new(
            &DenseVector::parse(vec![1.0, 0.0]).expect("query"),
            TopK::new(2).expect("top k"),
            HnswEfSearch::new(8).expect("ef search"),
        ))
        .expect("search");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, InternalDocId::new(2).expect("doc id"));
}

#[test]
fn remove_should_return_empty_when_all_docs_are_deleted() {
    let config = HnswIndexConfig::new(
        VectorDimension::new(2).expect("dimension"),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(2).expect("max neighbors"),
                HnswMinNeighborCount::new(1).expect("min neighbors"),
            )
            .expect("neighbor config"),
            HnswScalingFactor::new(2).expect("scaling factor"),
            HnswEfConstruction::new(8).expect("ef construction"),
            HnswPruneWidth::new(8).expect("prune width"),
        ),
    );

    let mut index = HnswIndex::build(
        config.clone(),
        vec![HnswBuildEntry::new(
            &config,
            InternalDocId::new(1).expect("doc id"),
            DenseVector::parse(vec![1.0, 0.0]).expect("vector"),
        )
        .expect("entry")],
    );

    assert_eq!(
        index.remove(InternalDocId::new(1).expect("doc id")),
        RemoveResult::Removed
    );

    let hits = index
        .search(HnswSearchRequest::new(
            &DenseVector::parse(vec![1.0, 0.0]).expect("query"),
            TopK::new(1).expect("top k"),
            HnswEfSearch::new(8).expect("ef search"),
        ))
        .expect("search");

    assert!(hits.is_empty());
}

#[test]
fn build_uses_graph_search_to_choose_insertion_neighbors() {
    let config = HnswIndexConfig::new(
        VectorDimension::new(2).unwrap(),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(2).unwrap(),
                HnswMinNeighborCount::new(1).unwrap(),
            )
            .unwrap(),
            HnswScalingFactor::new(2).unwrap(),
            HnswEfConstruction::new(8).unwrap(),
            HnswPruneWidth::new(8).unwrap(),
        ),
    );
    let entries = vec![
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(2).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(1).unwrap(),
            DenseVector::parse(vec![0.0, 1.0]).unwrap(),
        )
        .unwrap(),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(4).unwrap(),
            DenseVector::parse(vec![1.0, 0.0]).unwrap(),
        )
        .unwrap(),
    ];
    let index = HnswIndex::build(config, entries);

    assert_eq!(
        index
            .graph()
            .neighbors(HnswLevel::new(0), NodeIndex::new(2)),
        &[NodeIndex::new(0)]
    );
}
