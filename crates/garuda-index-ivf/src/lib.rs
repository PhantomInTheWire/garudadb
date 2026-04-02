//! IVF index construction, centroid assignment, and list search.

use garuda_math::score_doc;
use garuda_types::{
    DenseVector, DistanceMetric, InternalDocId, IvfIndexParams, IvfProbeCount, RemoveResult,
    Status, StatusCode, TopK, VectorDimension,
};
use std::collections::HashMap;

const CHURN_EVENT_DIVISOR: usize = 6;
const CHURN_EVENT_MIN: usize = 32;
const EMPTY_LIST_RETRAIN_NUMERATOR: usize = 1;
const EMPTY_LIST_RETRAIN_DENOMINATOR: usize = 5;

#[derive(Clone, Debug, PartialEq)]
pub struct IvfBuildEntry {
    doc_id: InternalDocId,
    vector: DenseVector,
}

impl IvfBuildEntry {
    pub fn new(doc_id: InternalDocId, vector: DenseVector) -> Self {
        Self { doc_id, vector }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvfSearchHit {
    pub doc_id: InternalDocId,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvfStoredLists {
    pub centroids: IvfCentroids,
    pub doc_ids_by_list: Vec<Vec<InternalDocId>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IvfCentroids(Vec<DenseVector>);

impl IvfCentroids {
    pub fn new(centroids: Vec<DenseVector>) -> Self {
        Self(centroids)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, DenseVector> {
        self.0.iter()
    }

    fn initialize(metric: DistanceMetric, entries: &[IvfBuildEntry], list_count: usize) -> Self {
        let mut centroid_indexes = Vec::with_capacity(list_count);
        centroid_indexes.push(Self::initial_centroid_index(metric, entries));

        while centroid_indexes.len() < list_count {
            let next = Self::next_centroid_index(metric, entries, &centroid_indexes);
            centroid_indexes.push(next);
        }

        let mut centroids = Vec::with_capacity(list_count);

        for index in centroid_indexes {
            centroids.push(entries[index].vector.clone());
        }

        Self::new(centroids)
    }

    fn initial_centroid_index(metric: DistanceMetric, entries: &[IvfBuildEntry]) -> usize {
        let mut sums = vec![0.0; entries[0].vector.len()];

        for entry in entries {
            add_to_sums(&mut sums, entry.vector.as_slice());
        }

        let mean = centroid_from_sums(&mut sums, entries.len());
        let mut best_index = 0usize;
        let mut best_score = score_doc(metric, entries[0].vector.as_slice(), mean.as_slice());

        for (entry_index, entry) in entries.iter().enumerate().skip(1) {
            let score = score_doc(metric, entry.vector.as_slice(), mean.as_slice());
            if score < best_score {
                best_index = entry_index;
                best_score = score;
            }
        }

        best_index
    }

    fn assign_entries(&self, metric: DistanceMetric, entries: &[IvfBuildEntry]) -> Vec<usize> {
        let mut assignments = Vec::with_capacity(entries.len());

        for entry in entries {
            assignments.push(nearest_centroid_index(metric, &entry.vector, self.iter()));
        }

        assignments
    }

    fn recompute(
        &self,
        dimension: VectorDimension,
        entries: &[IvfBuildEntry],
        assignments: &[usize],
    ) -> Self {
        let list_count = self.len();
        let dimension = dimension.get();
        let mut sums = vec![vec![0.0; dimension]; list_count];
        let mut counts = vec![0usize; list_count];

        for (entry, &list_index) in entries.iter().zip(assignments) {
            counts[list_index] += 1;
            add_to_sums(&mut sums[list_index], entry.vector.as_slice());
        }

        let mut centroids = Vec::with_capacity(list_count);

        for list_index in 0..list_count {
            if counts[list_index] == 0 {
                centroids.push(entries[list_index].vector.clone());
                continue;
            }

            centroids.push(centroid_from_sums(
                &mut sums[list_index],
                counts[list_index],
            ));
        }

        Self::new(centroids)
    }

    fn next_centroid_index(
        metric: DistanceMetric,
        entries: &[IvfBuildEntry],
        centroid_indexes: &[usize],
    ) -> usize {
        let mut next_index = 0usize;
        let mut next_score = f32::INFINITY;

        for (entry_index, entry) in entries.iter().enumerate() {
            if centroid_indexes.contains(&entry_index) {
                continue;
            }

            let nearest_score = centroid_indexes
                .iter()
                .map(|&centroid_index| {
                    score_doc(
                        metric,
                        entry.vector.as_slice(),
                        entries[centroid_index].vector.as_slice(),
                    )
                })
                .fold(f32::NEG_INFINITY, f32::max);

            if nearest_score < next_score {
                next_index = entry_index;
                next_score = nearest_score;
            }
        }

        next_index
    }
    fn into_vec(self) -> Vec<DenseVector> {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvfIndexConfig {
    pub dimension: VectorDimension,
    pub metric: DistanceMetric,
    pub params: IvfIndexParams,
}

impl IvfIndexConfig {
    pub fn new(dimension: VectorDimension, metric: DistanceMetric, params: IvfIndexParams) -> Self {
        Self {
            dimension,
            metric,
            params,
        }
    }

    pub fn list_count(&self, entry_count: usize) -> usize {
        entry_count.min(self.params.n_list.get() as usize)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvfSearchRequest<'a> {
    pub query_vector: &'a DenseVector,
    pub top_k: TopK,
    pub nprobe: IvfProbeCount,
}

impl<'a> IvfSearchRequest<'a> {
    pub fn new(query_vector: &'a DenseVector, top_k: TopK, nprobe: IvfProbeCount) -> Self {
        Self {
            query_vector,
            top_k,
            nprobe,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvfIndex {
    config: IvfIndexConfig,
    state: IvfState,
    churn_events: usize,
    retrain_state: IvfRetrainState,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WritingIvfIndex {
    config: IvfIndexConfig,
    state: IvfState,
    churn_events: usize,
    retrain_state: IvfRetrainState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IvfRetrainState {
    Ready,
    Pending,
}

#[derive(Clone, Debug, PartialEq)]
struct IvfState {
    entries: Vec<IvfBuildEntry>,
    inverted_lists: Vec<IvfInvertedList>,
    entry_index_by_doc_id: HashMap<InternalDocId, IvfEntryIndex>,
    entry_slots: Vec<IvfEntrySlot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IvfEntrySlot {
    Live { list_index: usize },
    Deleted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct IvfEntryIndex(usize);

impl IvfEntryIndex {
    fn new(value: usize) -> Self {
        Self(value)
    }

    fn get(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
struct IvfInvertedList {
    centroid: DenseVector,
    entry_indexes: Vec<IvfEntryIndex>,
}

impl IvfIndex {
    pub fn build(config: IvfIndexConfig, entries: Vec<IvfBuildEntry>) -> Self {
        assert_entries_match_dimension(config.dimension, &entries);

        let trained = train_lists(&config, &entries);

        Self {
            config,
            state: IvfState::new(entries, trained.centroids, trained.list_entry_indexes),
            churn_events: 0,
            retrain_state: IvfRetrainState::Ready,
        }
    }

    pub fn from_parts(
        config: IvfIndexConfig,
        entries: Vec<IvfBuildEntry>,
        stored: IvfStoredLists,
    ) -> Result<Self, Status> {
        validate_stored_lists(&config, &entries, &stored)?;

        let entry_indexes = build_list_entry_indexes(&entries, &stored.doc_ids_by_list)?;
        Ok(Self {
            config,
            state: IvfState::new(entries, stored.centroids, entry_indexes),
            churn_events: 0,
            retrain_state: IvfRetrainState::Ready,
        })
    }

    pub fn search(&self, request: IvfSearchRequest<'_>) -> Result<Vec<IvfSearchHit>, Status> {
        self.state.search(&self.config, request)
    }

    pub fn stored_lists(&self) -> IvfStoredLists {
        self.state.stored_lists()
    }

    pub fn len(&self) -> usize {
        self.state.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    pub fn list_count(&self) -> usize {
        self.state.list_count()
    }

    pub fn remove(&mut self, doc_id: InternalDocId) -> RemoveResult {
        let removed = self.state.remove_incremental(&self.config, doc_id);
        if removed.is_removed() {
            self.churn_events += 1;
            mark_retrain_pending_after_churn(
                &self.state,
                &mut self.churn_events,
                &mut self.retrain_state,
            );
        }

        removed
    }
}

impl WritingIvfIndex {
    fn new(config: IvfIndexConfig) -> Self {
        Self {
            config,
            state: IvfState::empty(),
            churn_events: 0,
            retrain_state: IvfRetrainState::Ready,
        }
    }

    pub fn from_entries_incremental(config: IvfIndexConfig, entries: Vec<IvfBuildEntry>) -> Self {
        let mut index = Self::new(config);

        for entry in entries {
            index.state.insert_incremental(&index.config, entry);
        }

        index
    }

    pub fn train(self) -> IvfIndex {
        IvfIndex::build(self.config, self.state.entries)
    }

    pub fn search(&self, request: IvfSearchRequest<'_>) -> Result<Vec<IvfSearchHit>, Status> {
        self.state.search(&self.config, request)
    }

    pub fn list_count(&self) -> usize {
        self.state.list_count()
    }

    pub fn insert(&mut self, entry: IvfBuildEntry) {
        if matches!(self.retrain_state, IvfRetrainState::Pending) {
            self.state = self.state.retrained(&self.config);
            self.retrain_state = IvfRetrainState::Ready;
            self.churn_events = 0;
        }

        self.state.insert_incremental(&self.config, entry);
    }

    pub fn remove(&mut self, doc_id: InternalDocId) -> RemoveResult {
        let removed = self.state.remove_incremental(&self.config, doc_id);
        if removed.is_removed() {
            self.churn_events += 1;
            mark_retrain_pending_after_churn(
                &self.state,
                &mut self.churn_events,
                &mut self.retrain_state,
            );
        }

        removed
    }
}

impl IvfState {
    fn new(
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
            .map(|(centroid, entry_indexes)| IvfInvertedList {
                centroid,
                entry_indexes,
            })
            .collect();

        Self {
            entries,
            inverted_lists,
            entry_index_by_doc_id,
            entry_slots,
        }
    }

    fn empty() -> Self {
        Self::new(Vec::new(), IvfCentroids::default(), Vec::new())
    }

    fn search(
        &self,
        config: &IvfIndexConfig,
        request: IvfSearchRequest<'_>,
    ) -> Result<Vec<IvfSearchHit>, Status> {
        search_entries(config, &self.entries, &self.inverted_lists, request)
    }

    fn stored_lists(&self) -> IvfStoredLists {
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

    fn len(&self) -> usize {
        self.entry_index_by_doc_id.len()
    }

    fn is_empty(&self) -> bool {
        self.entry_index_by_doc_id.is_empty()
    }

    fn list_count(&self) -> usize {
        self.inverted_lists.len()
    }

    fn push_new_list(&mut self, entry_index: IvfEntryIndex) -> usize {
        let list_index = self.inverted_lists.len();
        self.inverted_lists.push(IvfInvertedList {
            centroid: self.entries[entry_index.get()].vector.clone(),
            entry_indexes: vec![entry_index],
        });

        list_index
    }

    fn insert_incremental(&mut self, config: &IvfIndexConfig, entry: IvfBuildEntry) {
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
            let list_index = self.push_new_list(entry_index);
            self.entry_slots[entry_index.get()] = IvfEntrySlot::Live { list_index };
            return;
        }

        let list_count = config.list_count(self.entries.len());
        if self.inverted_lists.len() < list_count {
            let list_index = self.push_new_list(entry_index);
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
        self.inverted_lists[list_index].centroid = centroid_for_list(
            config.dimension,
            &self.entries,
            &self.inverted_lists[list_index].entry_indexes,
        );
        self.entry_slots[entry_index.get()] = IvfEntrySlot::Live { list_index };
    }

    fn remove_incremental(
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

        if list.entry_indexes.is_empty() {
            return RemoveResult::Removed;
        }

        list.centroid = centroid_for_list(config.dimension, &self.entries, &list.entry_indexes);
        RemoveResult::Removed
    }

    fn empty_list_count(&self) -> usize {
        self.inverted_lists
            .iter()
            .filter(|list| list.entry_indexes.is_empty())
            .count()
    }

    fn retrained(&self, config: &IvfIndexConfig) -> Self {
        let entries = self.live_entries();
        let trained = train_lists(config, &entries);
        Self::new(entries, trained.centroids, trained.list_entry_indexes)
    }

    fn live_entries(&self) -> Vec<IvfBuildEntry> {
        let mut entries = Vec::with_capacity(self.len());

        for (entry_index, entry) in self.entries.iter().enumerate() {
            if !matches!(self.entry_slots[entry_index], IvfEntrySlot::Live { .. }) {
                continue;
            }

            entries.push(entry.clone());
        }

        entries
    }
}

fn mark_retrain_pending_after_churn(
    state: &IvfState,
    churn_events: &mut usize,
    retrain_state: &mut IvfRetrainState,
) {
    let live_len = state.len();
    if live_len == 0 {
        *churn_events = 0;
        *retrain_state = IvfRetrainState::Ready;
        return;
    }

    let churn_threshold = (live_len / CHURN_EVENT_DIVISOR).max(CHURN_EVENT_MIN);
    let list_count = state.list_count();
    let empty_list_trigger = list_count != 0
        && state.empty_list_count() * EMPTY_LIST_RETRAIN_DENOMINATOR
            >= list_count * EMPTY_LIST_RETRAIN_NUMERATOR;

    if *churn_events < churn_threshold && !empty_list_trigger {
        return;
    }

    *retrain_state = IvfRetrainState::Pending;
    *churn_events = 0;
}

#[derive(Clone)]
struct TrainedLists {
    centroids: IvfCentroids,
    list_entry_indexes: Vec<Vec<IvfEntryIndex>>,
}

fn train_lists(config: &IvfIndexConfig, entries: &[IvfBuildEntry]) -> TrainedLists {
    if entries.is_empty() {
        return TrainedLists {
            centroids: IvfCentroids::default(),
            list_entry_indexes: Vec::new(),
        };
    }

    let centroids = initialize_centroids(config, entries);
    let (centroids, assignments) = run_lloyd_iterations(config, entries, centroids);
    let list_entry_indexes = materialize_list_entry_indexes(config, entries, &assignments);
    TrainedLists {
        centroids,
        list_entry_indexes,
    }
}

fn initialize_centroids(config: &IvfIndexConfig, entries: &[IvfBuildEntry]) -> IvfCentroids {
    IvfCentroids::initialize(config.metric, entries, config.list_count(entries.len()))
}

fn assign_entries(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    centroids: &IvfCentroids,
) -> Vec<usize> {
    centroids.assign_entries(config.metric, entries)
}

fn recompute_centroids(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    assignments: &[usize],
    centroids: &IvfCentroids,
) -> IvfCentroids {
    centroids.recompute(config.dimension, entries, assignments)
}

fn run_lloyd_iterations(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    mut centroids: IvfCentroids,
) -> (IvfCentroids, Vec<usize>) {
    let mut assignments = vec![0usize; entries.len()];

    for _ in 0..config.params.training_iterations.get() {
        assignments = assign_entries(config, entries, &centroids);
        centroids = recompute_centroids(config, entries, &assignments, &centroids);
    }

    (centroids, assignments)
}

fn materialize_list_entry_indexes(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    assignments: &[usize],
) -> Vec<Vec<IvfEntryIndex>> {
    let mut list_entry_indexes = vec![Vec::new(); config.list_count(entries.len())];

    for (entry_index, &list_index) in assignments.iter().enumerate() {
        list_entry_indexes[list_index].push(IvfEntryIndex::new(entry_index));
    }

    list_entry_indexes
}
fn nearest_centroid_index<'a>(
    metric: DistanceMetric,
    vector: &DenseVector,
    centroids: impl IntoIterator<Item = &'a DenseVector>,
) -> usize {
    let mut centroids = centroids.into_iter().enumerate();
    let (mut best_index, best_centroid) = centroids.next().expect("ivf centroids");
    let mut best_score = score_doc(metric, vector.as_slice(), best_centroid.as_slice());

    for (index, centroid) in centroids {
        let score = score_doc(metric, vector.as_slice(), centroid.as_slice());
        if score > best_score {
            best_index = index;
            best_score = score;
        }
    }

    best_index
}

fn centroid_for_list(
    dimension: VectorDimension,
    entries: &[IvfBuildEntry],
    entry_indexes: &[IvfEntryIndex],
) -> DenseVector {
    let mut sums = vec![0.0; dimension.get()];

    for &entry_index in entry_indexes {
        add_to_sums(&mut sums, entries[entry_index.get()].vector.as_slice());
    }

    centroid_from_sums(&mut sums, entry_indexes.len())
}

fn add_to_sums(sums: &mut [f32], values: &[f32]) {
    sums.iter_mut()
        .zip(values)
        .for_each(|(sum, value)| *sum += *value);
}

fn centroid_from_sums(sums: &mut [f32], count: usize) -> DenseVector {
    let scale = count as f32;

    for value in sums.iter_mut() {
        *value /= scale;
    }

    DenseVector::parse(sums.to_vec()).expect("ivf centroid vector")
}

fn assert_entries_match_dimension(dimension: VectorDimension, entries: &[IvfBuildEntry]) {
    for entry in entries {
        assert_eq!(
            entry.vector.len(),
            dimension.get(),
            "ivf index entry dimension"
        );
    }
}

fn build_list_entry_indexes(
    entries: &[IvfBuildEntry],
    doc_ids_by_list: &[Vec<InternalDocId>],
) -> Result<Vec<Vec<IvfEntryIndex>>, Status> {
    let mut entry_index_by_doc_id = HashMap::with_capacity(entries.len());

    for (entry_index, entry) in entries.iter().enumerate() {
        entry_index_by_doc_id.insert(entry.doc_id, entry_index);
    }

    let mut list_entry_indexes = Vec::with_capacity(doc_ids_by_list.len());

    for doc_ids in doc_ids_by_list {
        let mut indexes = Vec::with_capacity(doc_ids.len());

        for &doc_id in doc_ids {
            let Some(entry_index) = entry_index_by_doc_id.remove(&doc_id) else {
                return Err(Status::err(
                    StatusCode::Internal,
                    "persisted ivf lists do not match live entries",
                ));
            };

            indexes.push(IvfEntryIndex::new(entry_index));
        }

        list_entry_indexes.push(indexes);
    }

    if entry_index_by_doc_id.is_empty() {
        return Ok(list_entry_indexes);
    }

    Err(Status::err(
        StatusCode::Internal,
        "persisted ivf lists do not match live entries",
    ))
}

fn validate_stored_lists(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    stored: &IvfStoredLists,
) -> Result<(), Status> {
    let expected_list_count = config.list_count(entries.len());
    if stored.centroids.len() != expected_list_count {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted ivf centroid count does not match live entries",
        ));
    }

    if stored.doc_ids_by_list.len() != expected_list_count {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted ivf list count does not match live entries",
        ));
    }

    for centroid in stored.centroids.iter() {
        if centroid.len() == config.dimension.get() {
            continue;
        }

        return Err(Status::err(
            StatusCode::Internal,
            "persisted ivf centroid dimension does not match index dimension",
        ));
    }

    Ok(())
}

fn search_entries(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    inverted_lists: &[IvfInvertedList],
    request: IvfSearchRequest<'_>,
) -> Result<Vec<IvfSearchHit>, Status> {
    if request.query_vector.len() != config.dimension.get() {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "query vector dimension does not match ivf index dimension",
        ));
    }

    if entries.is_empty() {
        return Ok(Vec::new());
    }

    let mut list_scores = Vec::with_capacity(inverted_lists.len());

    for (list_index, list) in inverted_lists.iter().enumerate() {
        if list.entry_indexes.is_empty() {
            continue;
        }

        list_scores.push((
            list_index,
            score_doc(
                config.metric,
                request.query_vector.as_slice(),
                list.centroid.as_slice(),
            ),
        ));
    }

    list_scores.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    list_scores.truncate((request.nprobe.get() as usize).min(list_scores.len()));

    let mut hits = Vec::new();

    for (list_index, _) in list_scores {
        for &entry_index in &inverted_lists[list_index].entry_indexes {
            let entry = &entries[entry_index.get()];
            hits.push(IvfSearchHit {
                doc_id: entry.doc_id,
                score: score_doc(
                    config.metric,
                    request.query_vector.as_slice(),
                    entry.vector.as_slice(),
                ),
            });
        }
    }

    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
    hits.truncate(request.top_k.get());
    Ok(hits)
}
