use garuda_types::{HnswGraph, HnswLevel, NodeIndex};

#[test]
fn push_node_adds_missing_slots_when_it_creates_a_new_level() {
    let mut graph = HnswGraph::new(vec![HnswLevel::new(0)]);

    let node = graph.push_node(HnswLevel::new(1));

    assert_eq!(node, NodeIndex::new(1));
    assert!(
        graph
            .neighbors(HnswLevel::new(1), NodeIndex::new(0))
            .is_empty()
    );
    assert!(
        graph
            .neighbors(HnswLevel::new(1), NodeIndex::new(1))
            .is_empty()
    );
}
