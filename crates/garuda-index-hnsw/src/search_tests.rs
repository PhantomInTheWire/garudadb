use crate::{HnswBuildConfig, HnswBuildEntry, HnswIndex, HnswIndexConfig};
use garuda_types::{
    DenseVector, DistanceMetric, HnswEfConstruction, HnswGraph, HnswLevel, HnswM,
    HnswMinNeighborCount, HnswNeighborConfig, HnswPruneWidth, HnswScalingFactor,
    InternalDocId, NodeIndex, TopK, VectorDimension,
};

#[test]
fn search_layer_returns_best_candidate_first() {
    let config = HnswIndexConfig::new(
        VectorDimension::new(2).expect("valid dimension"),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(2).expect("valid max_neighbors"),
                HnswMinNeighborCount::new(1).expect("valid min_neighbor_count"),
            )
            .expect("valid neighbor config"),
            HnswScalingFactor::new(2).expect("valid scaling_factor"),
            HnswEfConstruction::new(8).expect("valid ef_construction"),
            HnswPruneWidth::new(8).expect("valid prune_width"),
        ),
    );
    let entries = vec![
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(1).expect("valid doc id"),
            DenseVector::parse(vec![1.0, 0.0]).expect("valid vector"),
        )
        .expect("valid entry"),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(2).expect("valid doc id"),
            DenseVector::parse(vec![0.5, 0.5]).expect("valid vector"),
        )
        .expect("valid entry"),
        HnswBuildEntry::new(
            &config,
            InternalDocId::new(3).expect("valid doc id"),
            DenseVector::parse(vec![0.0, 1.0]).expect("valid vector"),
        )
        .expect("valid entry"),
    ];
    let graph = HnswGraph::from_parts(
        vec![HnswLevel::new(0), HnswLevel::new(0), HnswLevel::new(0)],
        vec![vec![
            vec![NodeIndex::new(1), NodeIndex::new(2)],
            Vec::new(),
            Vec::new(),
        ]],
        3,
        config.neighbor_limits(),
    )
    .expect("valid graph");
    let index = HnswIndex::from_parts(config, entries, graph);

    let candidates = index.search_layer(
        HnswLevel::new(0),
        NodeIndex::new(0),
        &DenseVector::parse(vec![1.0, 0.0]).expect("valid vector"),
        TopK::new(3).expect("valid top_k").get(),
    );

    assert_eq!(candidates[0].index, NodeIndex::new(0));
    assert_eq!(candidates[1].index, NodeIndex::new(1));
    assert_eq!(candidates[2].index, NodeIndex::new(2));
}
