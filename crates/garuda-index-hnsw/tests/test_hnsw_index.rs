use garuda_index_hnsw::{
    HnswBuildConfig, HnswBuildEntry, HnswIndex, HnswIndexConfig, HnswSearchRequest,
};
use garuda_types::{
    DenseVector, DistanceMetric, HnswEfConstruction, HnswEfSearch, HnswGraph, HnswLevel, HnswM,
    HnswMinNeighborCount, HnswNeighborConfig, HnswNeighborLimits, HnswPruneWidth,
    HnswScalingFactor, InternalDocId, NodeIndex, TopK, VectorDimension,
};

#[test]
fn graph_from_parts_rejects_malformed_level_shape() {
    let result = HnswGraph::from_parts(
        vec![HnswLevel::new(0), HnswLevel::new(0)],
        vec![vec![Vec::<NodeIndex>::new()]],
        2,
        HnswNeighborLimits::new(HnswM::new(16).expect("m")),
    );

    assert!(result.is_err());
}

#[test]
fn graph_from_parts_rejects_graph_without_top_level_node() {
    let result = HnswGraph::from_parts(
        vec![HnswLevel::new(0)],
        vec![vec![Vec::<NodeIndex>::new()], vec![Vec::<NodeIndex>::new()]],
        1,
        HnswNeighborLimits::new(HnswM::new(16).expect("m")),
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
        HnswNeighborLimits::new(HnswM::new(1).expect("m")),
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
        HnswNeighborLimits::new(HnswM::new(1).expect("m")),
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
            HnswScalingFactor::new(50).unwrap(),
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
