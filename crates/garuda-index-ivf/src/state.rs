use crate::centroids::{centroid_for_list, nearest_centroid_index};
use crate::{
    IvfBuildEntry, IvfCentroids, IvfIndexConfig, IvfSearchHit, IvfSearchRequest, IvfStoredLists,
    search_entries, train_lists,
};
use garuda_math::l2_norm;
use garuda_types::{DenseVector, DistanceMetric, InternalDocId, RemoveResult, Status};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct IvfState {
    entries: Vec<IvfBuildEntry>,
    inverted_lists: Vec<IvfInvertedList>,
    entry_index_by_doc_id: HashMap<InternalDocId, IvfEntryIndex>,
    entry_slots: Vec<IvfEntrySlot>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct IvfInvertedList {
    pub(super) centroid: DenseVector,
    pub(super) residual_bound: IvfResidualBound,
    pub(super) entry_indexes: Vec<IvfEntryIndex>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum IvfResidualBound {
    Cosine,
    InnerProduct { max_residual_norm: f32 },
    L2 { max_distance_to_centroid: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IvfEntrySlot {
    Live { list_index: usize },
    Deleted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct IvfEntryIndex(usize);

impl IvfEntryIndex {
    pub(super) fn new(value: usize) -> Self {
        Self(value)
    }

    pub(super) fn get(self) -> usize {
        self.0
    }
}

impl IvfState {
    pub(super) fn new(
        config: &IvfIndexConfig,
        entries: Vec<IvfBuildEntry>,
        centroids: IvfCentroids,
        list_entry_indexes: Vec<Vec<IvfEntryIndex>>,
    ) -> Self {
        assert_eq!(centroids.len(), list_entry_indexes.len(), "ivf list layout");

        let mut entry_index_by_doc_id = HashMap::with_capacity(entries.len());
        for (entry_index, entry) in entries.iter().enumerate() {
            entry_index_by_doc_id.insert(entry.doc_id, IvfEntryIndex::new(entry_index));
        }

        let mut entry_slots = vec![IvfEntrySlot::Deleted; entries.len()];
        for (list_index, entry_indexes) in list_entry_indexes.iter().enumerate() {
            for &entry_index in entry_indexes {
                entry_slots[entry_index.get()] = IvfEntrySlot::Live { list_index };
            }
        }

        let inverted_lists = centroids
            .into_vec()
            .into_iter()
            .zip(list_entry_indexes)
            .map(|(centroid, entry_indexes)| {
                let residual_bound =
                    residual_bound_for_list(config.metric, &centroid, &entries, &entry_indexes);

                IvfInvertedList {
                    centroid,
                    residual_bound,
                    entry_indexes,
                }
            })
            .collect();

        Self {
            entries,
            inverted_lists,
            entry_index_by_doc_id,
            entry_slots,
        }
    }

    pub(super) fn empty() -> Self {
        Self {
            entries: Vec::new(),
            inverted_lists: Vec::new(),
            entry_index_by_doc_id: HashMap::new(),
            entry_slots: Vec::new(),
        }
    }

    pub(super) fn search(
        &self,
        config: &IvfIndexConfig,
        request: IvfSearchRequest<'_>,
    ) -> Result<Vec<IvfSearchHit>, Status> {
        search_entries(config, &self.entries, &self.inverted_lists, request)
    }

    pub(super) fn stored_lists(&self) -> IvfStoredLists {
        let mut doc_ids_by_list = Vec::with_capacity(self.inverted_lists.len());

        for list in &self.inverted_lists {
            let mut doc_ids = Vec::with_capacity(list.entry_indexes.len());

            for &entry_index in &list.entry_indexes {
                doc_ids.push(self.entries[entry_index.get()].doc_id);
            }

            doc_ids_by_list.push(doc_ids);
        }

        IvfStoredLists {
            centroids: IvfCentroids::new(
                self.inverted_lists
                    .iter()
                    .map(|list| list.centroid.clone())
                    .collect(),
            ),
            doc_ids_by_list,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.entry_index_by_doc_id.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entry_index_by_doc_id.is_empty()
    }

    pub(super) fn list_count(&self) -> usize {
        self.inverted_lists.len()
    }

    pub(super) fn non_empty_list_count(&self) -> usize {
        self.inverted_lists
            .iter()
            .filter(|list| !list.entry_indexes.is_empty())
            .count()
    }

    fn push_new_list(&mut self, config: &IvfIndexConfig, entry_index: IvfEntryIndex) -> usize {
        let list_index = self.inverted_lists.len();
        let centroid = self.entries[entry_index.get()].vector.clone();
        let entry_indexes = vec![entry_index];
        let residual_bound =
            residual_bound_for_list(config.metric, &centroid, &self.entries, &entry_indexes);

        self.inverted_lists.push(IvfInvertedList {
            centroid,
            residual_bound,
            entry_indexes,
        });

        list_index
    }

    pub(super) fn insert_incremental(&mut self, config: &IvfIndexConfig, entry: IvfBuildEntry) {
        assert_eq!(
            entry.vector.len(),
            config.dimension.get(),
            "ivf index entry dimension"
        );

        let entry_index = self.entries.len();
        self.entries.push(entry);
        let entry_index = IvfEntryIndex::new(entry_index);
        self.entry_index_by_doc_id
            .insert(self.entries[entry_index.get()].doc_id, entry_index);
        self.entry_slots.push(IvfEntrySlot::Deleted);

        if self.inverted_lists.is_empty() {
            let list_index = self.push_new_list(config, entry_index);
            self.entry_slots[entry_index.get()] = IvfEntrySlot::Live { list_index };
            return;
        }

        let list_count = config.list_count(self.len());
        if self.inverted_lists.len() < list_count {
            let list_index = self.push_new_list(config, entry_index);
            self.entry_slots[entry_index.get()] = IvfEntrySlot::Live { list_index };
            return;
        }

        let list_index = nearest_centroid_index(
            config.metric,
            &self.entries[entry_index.get()].vector,
            self.inverted_lists.iter().map(|list| &list.centroid),
        );
        self.inverted_lists[list_index]
            .entry_indexes
            .push(entry_index);
        self.recompute_list(config, list_index);
        self.entry_slots[entry_index.get()] = IvfEntrySlot::Live { list_index };
    }

    pub(super) fn remove_incremental(
        &mut self,
        config: &IvfIndexConfig,
        doc_id: InternalDocId,
    ) -> RemoveResult {
        let Some(entry_index) = self.entry_index_by_doc_id.remove(&doc_id) else {
            return RemoveResult::Missing;
        };

        let raw_entry_index = entry_index.get();
        let list_index = match self.entry_slots[raw_entry_index] {
            IvfEntrySlot::Live { list_index } => list_index,
            IvfEntrySlot::Deleted => unreachable!("ivf entry slot should be live for known doc id"),
        };
        self.entry_slots[raw_entry_index] = IvfEntrySlot::Deleted;

        let list = &mut self.inverted_lists[list_index];
        let original_len = list.entry_indexes.len();
        list.entry_indexes.retain(|&index| index != entry_index);
        assert_ne!(
            original_len,
            list.entry_indexes.len(),
            "ivf list membership must contain removed entry"
        );

        if !list.entry_indexes.is_empty() {
            self.recompute_list(config, list_index);
            return RemoveResult::Removed;
        }

        list.residual_bound =
            residual_bound_for_list(config.metric, &list.centroid, &self.entries, &[]);
        RemoveResult::Removed
    }

    pub(super) fn empty_list_count(&self) -> usize {
        self.inverted_lists
            .iter()
            .filter(|list| list.entry_indexes.is_empty())
            .count()
    }

    pub(super) fn retrained(&self, config: &IvfIndexConfig) -> Self {
        let entries = self.live_entries();
        let trained = train_lists(config, &entries);
        Self::new(
            config,
            entries,
            trained.centroids,
            trained.list_entry_indexes,
        )
    }

    pub(super) fn live_entries(&self) -> Vec<IvfBuildEntry> {
        let mut entries = Vec::with_capacity(self.len());

        for (entry_index, entry) in self.entries.iter().enumerate() {
            if !matches!(self.entry_slots[entry_index], IvfEntrySlot::Live { .. }) {
                continue;
            }

            entries.push(entry.clone());
        }

        entries
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.inverted_lists.clear();
        self.entry_index_by_doc_id.clear();
        self.entry_slots.clear();
    }

    fn recompute_list(&mut self, config: &IvfIndexConfig, list_index: usize) {
        let list = &mut self.inverted_lists[list_index];
        assert!(
            !list.entry_indexes.is_empty(),
            "recompute requires live list"
        );
        list.centroid = centroid_for_list(config.dimension, &self.entries, &list.entry_indexes);
        list.residual_bound = residual_bound_for_list(
            config.metric,
            &list.centroid,
            &self.entries,
            &list.entry_indexes,
        );
    }
}

fn residual_bound_for_list(
    metric: DistanceMetric,
    centroid: &DenseVector,
    entries: &[IvfBuildEntry],
    entry_indexes: &[IvfEntryIndex],
) -> IvfResidualBound {
    match metric {
        DistanceMetric::Cosine => IvfResidualBound::Cosine,
        DistanceMetric::InnerProduct => IvfResidualBound::InnerProduct {
            max_residual_norm: max_residual_norm(centroid, entries, entry_indexes),
        },
        DistanceMetric::L2 => IvfResidualBound::L2 {
            max_distance_to_centroid: max_residual_norm(centroid, entries, entry_indexes),
        },
    }
}

impl IvfResidualBound {
    pub(super) fn upper_bound(self, centroid_score: f32, query_vector: &[f32]) -> f32 {
        match self {
            Self::Cosine => f32::INFINITY,
            Self::InnerProduct { max_residual_norm } => {
                centroid_score + l2_norm(query_vector) * max_residual_norm
            }
            Self::L2 {
                max_distance_to_centroid,
            } => {
                let centroid_distance = -centroid_score;
                -(centroid_distance - max_distance_to_centroid).max(0.0)
            }
        }
    }
}

fn max_residual_norm(
    centroid: &DenseVector,
    entries: &[IvfBuildEntry],
    entry_indexes: &[IvfEntryIndex],
) -> f32 {
    let mut max_norm = 0.0;

    for &entry_index in entry_indexes {
        let norm = l2_norm_of_difference(
            centroid.as_slice(),
            entries[entry_index.get()].vector.as_slice(),
        );
        if norm > max_norm {
            max_norm = norm;
        }
    }

    max_norm
}

fn l2_norm_of_difference(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut sum = 0.0;

    for (left, right) in lhs.iter().zip(rhs) {
        let delta = left - right;
        sum += delta * delta;
    }

    sum.sqrt()
}
