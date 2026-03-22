use garuda_types::NodeIndex;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScoredNode {
    pub(crate) index: NodeIndex,
    pub(crate) score: f32,
}

impl Eq for ScoredNode {}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WorstScoredNode {
    pub(crate) index: NodeIndex,
    pub(crate) score: f32,
}

impl From<ScoredNode> for WorstScoredNode {
    fn from(value: ScoredNode) -> Self {
        Self {
            index: value.index,
            score: value.score,
        }
    }
}

impl From<WorstScoredNode> for ScoredNode {
    fn from(value: WorstScoredNode) -> Self {
        Self {
            index: value.index,
            score: value.score,
        }
    }
}

impl Eq for WorstScoredNode {}

impl Ord for WorstScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for WorstScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
