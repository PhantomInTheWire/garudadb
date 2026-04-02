use garuda_index_hnsw::{
    HnswBuildConfig, HnswBuildEntry, HnswIndex, HnswIndexConfig, HnswSearchRequest,
};
use garuda_types::{
    DenseVector, DistanceMetric, HnswEfConstruction, HnswEfSearch, HnswGraph, HnswLevel, HnswM,
    HnswMinNeighborCount, HnswNeighborConfig, HnswNeighborLimits, HnswPruneWidth,
    HnswScalingFactor, InternalDocId, NodeIndex, RemoveResult, TopK, VectorDimension,
};

fn test_config(max_neighbors: u32, min_neighbors: u32) -> HnswIndexConfig {
    HnswIndexConfig::new(
        VectorDimension::new(2).expect("dimension"),
        DistanceMetric::InnerProduct,
        HnswBuildConfig::new(
            HnswNeighborConfig::new(
                HnswM::new(max_neighbors).expect("max neighbors"),
                HnswMinNeighborCount::new(min_neighbors).expect("min neighbors"),
            )
            .expect("neighbor config"),
            HnswScalingFactor::new(2).expect("scaling factor"),
            HnswEfConstruction::new(8).expect("ef construction"),
            HnswPruneWidth::new(8).expect("prune width"),
        ),
    )
}

fn entry(config: &HnswIndexConfig, doc_id: u64, vector: Vec<f32>) -> HnswBuildEntry {
    HnswBuildEntry::new(
        config,
        InternalDocId::new(doc_id).expect("doc id"),
        DenseVector::parse(vector).expect("vector"),
    )
    .expect("entry")
}

#[test]
fn remove_from_parts_with_incoming_only_edges_keeps_remaining_node_searchable() {
    let config = test_config(2, 1);
    let entries = vec![
        entry(&config, 1, vec![1.0, 0.0]),
        entry(&config, 2, vec![0.0, 1.0]),
    ];
    let graph = HnswGraph::from_parts(
        vec![HnswLevel::new(0), HnswLevel::new(0)],
        vec![vec![vec![NodeIndex::new(1)], vec![]]],
        2,
        HnswNeighborLimits::new(HnswM::new(2).expect("max neighbors")),
    )
    .expect("graph");
    let mut index = HnswIndex::from_parts(config, entries, graph);

    assert_eq!(
        index.remove(InternalDocId::new(2).expect("doc id")),
        RemoveResult::Removed
    );

    let hits = index
        .search(HnswSearchRequest::new(
            &DenseVector::parse(vec![1.0, 0.0]).expect("query"),
            TopK::new(1).expect("top k"),
            HnswEfSearch::new(8).expect("ef search"),
        ))
        .expect("search");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, InternalDocId::new(1).expect("doc id"));
}
