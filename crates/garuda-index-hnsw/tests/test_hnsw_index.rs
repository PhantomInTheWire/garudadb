use garuda_types::{HnswGraph, HnswLevel, HnswM, HnswNeighborLimits, NodeIndex};

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
