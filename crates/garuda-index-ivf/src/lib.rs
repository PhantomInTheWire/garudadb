//! IVF index construction, centroid assignment, and list search.

mod centroids;
mod persistence;
mod state;

use garuda_math::score_doc;
use persistence::{build_list_entry_indexes, validate_stored_lists};
use state::IvfEntryIndex;
use state::IvfInvertedList;
use state::IvfState;

use garuda_types::{
    DenseVector, DistanceMetric, InternalDocId, IvfIndexParams, IvfProbeCount, RemoveResult,
    Status, StatusCode, TopK, VectorDimension,
};

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
}

impl IvfIndex {
    pub fn build(config: IvfIndexConfig, entries: Vec<IvfBuildEntry>) -> Self {
        let trained = train_lists(&config, &entries);

        Self {
            config,
            state: IvfState::new(entries, trained.centroids, trained.list_entry_indexes),
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
        self.state.remove_incremental(&self.config, doc_id)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WritingIvfIndex {
    config: IvfIndexConfig,
    state: IvfState,
    churn_events: usize,
    retrain_state: IvfRetrainState,
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
        IvfIndex::build(self.config, self.state.live_entries())
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
                &mut self.state,
                &mut self.churn_events,
                &mut self.retrain_state,
            );
        }

        removed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IvfRetrainState {
    Ready,
    Pending,
}

#[derive(Clone)]
struct TrainedLists {
    centroids: IvfCentroids,
    list_entry_indexes: Vec<Vec<IvfEntryIndex>>,
}

fn mark_retrain_pending_after_churn(
    state: &mut IvfState,
    churn_events: &mut usize,
    retrain_state: &mut IvfRetrainState,
) {
    let live_len = state.len();
    if live_len == 0 {
        state.clear();
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

fn train_lists(config: &IvfIndexConfig, entries: &[IvfBuildEntry]) -> TrainedLists {
    centroids::assert_entries_match_dimension(config.dimension, entries);

    if entries.is_empty() {
        return TrainedLists {
            centroids: IvfCentroids::default(),
            list_entry_indexes: Vec::new(),
        };
    }

    let centroids =
        IvfCentroids::initialize(config.metric, entries, config.list_count(entries.len()));
    let (centroids, assignments) = run_lloyd_iterations(config, entries, centroids);
    let list_entry_indexes = materialize_list_entry_indexes(config, entries, &assignments);
    TrainedLists {
        centroids,
        list_entry_indexes,
    }
}

fn run_lloyd_iterations(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    mut centroids: IvfCentroids,
) -> (IvfCentroids, Vec<usize>) {
    let mut assignments = vec![0usize; entries.len()];

    for _ in 0..config.params.training_iterations.get() {
        assignments = centroids.assign_entries(config.metric, entries);
        centroids = centroids.recompute(config.dimension, entries, &assignments);
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
