//! IVF index construction, centroid assignment, and list search.
//!
//! This crate implements a small, deterministic IVF variant around three
//! pieces of state:
//!
//! - `IvfIndexConfig`: vector dimension, metric, and IVF parameters.
//! - `Vec<IvfBuildEntry>`: the stored `(doc_id, vector)` rows.
//! - `IvfState`: centroids, residual bounds, per-list membership, and
//!   live/deleted bookkeeping.
//!
//! `IvfIndex` is the persisted/searchable form. `WritingIvfIndex` is the
//! incremental mutable form that can accept inserts/removes and occasionally
//! retrain.
//!
//! Implementation outline:
//!
//! 1. Build-time data model
//! - Entries are stored once in `entries`.
//! - Each inverted list stores one centroid and a list of `IvfEntryIndex`
//!   values that point into `entries`.
//! - Live membership is tracked separately from historical storage so deletions
//!   can remove a doc from search without compacting `entries`.
//! - Scores are always "higher is better". `score_doc` provides that ordering
//!   for all supported metrics.
//!
//! 2. List count
//! - The runtime list count is `min(entry_count, n_list)`.
//! - Empty builds therefore produce zero lists.
//! - Small builds never create more lists than live entries.
//!
//! 3. Deterministic centroid initialization
//! - The first centroid is the entry farthest from the global mean.
//! - Each later centroid is the entry whose nearest existing centroid is worst.
//! - Centroids are therefore chosen without RNG state.
//! - Rebuilding the same live entries yields the same initial seeds.
//!
//! 4. Lloyd training
//! - Training alternates between:
//!   - assigning every entry to its nearest centroid, and
//!   - recomputing each centroid as the arithmetic mean of its assigned entries.
//! - The number of passes is exactly `training_iterations`.
//! - If a centroid receives no assignments during recomputation, that slot falls
//!   back to `entries[list_index].vector`.
//! - After the final assignment pass, the crate materializes per-list
//!   `IvfEntryIndex` vectors.
//!
//! 5. Query-time search
//! - Search validates query dimension and returns early on an empty index.
//! - Every non-empty list is scored against the query using its centroid.
//! - Lists are sorted by centroid score descending, then `list_index` ascending.
//! - Search always scans at least the best `min(nprobe, non_empty_lists)`
//!   lists.
//! - L2 and inner-product queries may keep scanning ranked lists beyond that
//!   minimum until every remaining list is upper-bounded below the current
//!   top-k threshold.
//! - Equal-score ties are not pruned early because final hit ordering also
//!   breaks ties by `doc_id`.
//! - Cosine queries do not use residual-bound early stopping.
//! - Cosine search scans exactly the best `min(nprobe, non_empty_lists)` lists
//!   and does not scan beyond that minimum.
//! - Every scanned entry is scored against the query.
//! - Final hits are sorted by score descending, then `doc_id` ascending, and
//!   truncated to `top_k`.
//!
//! 6. Incremental insert
//! - `WritingIvfIndex` starts empty and inserts entries one by one.
//! - While the current list count is still below `min(live_len, n_list)`, each
//!   new live entry becomes its own singleton list whose centroid is that
//!   vector.
//! - Once the target list count is reached, a new entry is assigned to the
//!   nearest centroid, appended to that list, and that list centroid is updated
//!   to the exact mean of its current members.
//! - Residual bounds are recomputed whenever a list centroid changes.
//! - Incremental insert does not run full Lloyd retraining.
//!
//! 7. Incremental remove
//! - Removal looks up the live entry by `doc_id`, marks its slot deleted, and
//!   removes its `IvfEntryIndex` from the owning list.
//! - If the list still has members, its centroid is recomputed from those live
//!   members and its residual bound is recomputed from that same membership.
//! - If the list becomes empty, the empty list remains in place until retrain or
//!   full reset.
//! - Incremental remove therefore does only list-local exact maintenance; it
//!   does not retrain centroids globally on the delete path.
//! - Removed docs disappear from search and from persisted `stored_lists()`.
//!
//! 8. Churn-triggered retraining
//! - `WritingIvfIndex` counts successful removes as churn events.
//! - Retrain becomes pending when either:
//!   - churn reaches `max(live_len / 6, 32)`, or
//!   - at least `1/5` of lists are empty.
//! - The next insert applies retraining first by rebuilding from current live
//!   entries, then inserts the new entry into that rebuilt state.
//! - If all live entries are removed, runtime state is cleared immediately.
//!
//! 9. Persisted layout
//! - `stored_lists()` exports:
//!   - centroid vectors in list order, and
//!   - `doc_id`s for each list in that same order.
//! - `from_parts` validates centroid count, list count, and centroid dimension
//!   before rebuilding in-memory `IvfEntryIndex` membership.
//! - Residual bounds are rebuilt from the persisted centroids and live entries;
//!   they are derived state, not persisted bytes.
//! - Persisted lists must match the live entry set exactly: no duplicates, no
//!   missing docs, no unknown docs.
//!
//! 10. Determinism and non-goals
//! - Build-time training is deterministic because initialization, assignment,
//!   sorting, and tie-breaking are deterministic.
//! - The implementation is single-threaded and in-memory.
//! - Search still scans whole selected lists; there is no product quantization
//!   or residual encoding in this crate.
//! - Deletion is lazy at the storage layer: historical `entries` slots are not
//!   compacted in place.
//! - Incremental maintenance favors simple exact centroid updates plus periodic
//!   rebuilds over continuous online k-means optimization.

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
        let state = IvfState::new(
            &config,
            entries,
            trained.centroids,
            trained.list_entry_indexes,
        );

        Self { config, state }
    }

    pub fn from_parts(
        config: IvfIndexConfig,
        entries: Vec<IvfBuildEntry>,
        stored: IvfStoredLists,
    ) -> Result<Self, Status> {
        validate_stored_lists(&config, &entries, &stored)?;

        let entry_indexes = build_list_entry_indexes(&entries, &stored.doc_ids_by_list)?;
        let state = IvfState::new(&config, entries, stored.centroids, entry_indexes);
        Ok(Self { config, state })
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

    pub fn non_empty_list_count(&self) -> usize {
        self.state.non_empty_list_count()
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

    pub fn non_empty_list_count(&self) -> usize {
        self.state.non_empty_list_count()
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

    if config.metric == DistanceMetric::Cosine {
        return Ok(scan_ranked_lists(
            config,
            entries,
            inverted_lists,
            &request,
            &list_scores[..(request.nprobe.get() as usize).min(list_scores.len())],
        ));
    }

    Ok(scan_ranked_lists_with_pruning(
        config,
        entries,
        inverted_lists,
        &request,
        &list_scores,
    ))
}

fn scan_ranked_lists(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    inverted_lists: &[IvfInvertedList],
    request: &IvfSearchRequest<'_>,
    list_scores: &[(usize, f32)],
) -> Vec<IvfSearchHit> {
    let mut hits = Vec::new();

    for &(list_index, _) in list_scores {
        scan_list(
            config,
            entries,
            &inverted_lists[list_index],
            request,
            &mut hits,
        );
    }

    sort_hits(&mut hits);
    hits.truncate(request.top_k.get());
    hits
}

fn scan_ranked_lists_with_pruning(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    inverted_lists: &[IvfInvertedList],
    request: &IvfSearchRequest<'_>,
    list_scores: &[(usize, f32)],
) -> Vec<IvfSearchHit> {
    let minimum_scanned_list_count = (request.nprobe.get() as usize).min(list_scores.len());
    let mut hits = Vec::new();

    for (rank, (list_index, _)) in list_scores.iter().copied().enumerate() {
        scan_list(
            config,
            entries,
            &inverted_lists[list_index],
            request,
            &mut hits,
        );
        sort_hits(&mut hits);
        if hits.len() > request.top_k.get() {
            hits.truncate(request.top_k.get());
        }

        if rank + 1 < minimum_scanned_list_count {
            continue;
        }

        if hits.len() < request.top_k.get() {
            continue;
        }

        let Some(best_remaining_upper_bound) = best_remaining_upper_bound(
            inverted_lists,
            request.query_vector.as_slice(),
            &list_scores[(rank + 1)..],
        ) else {
            break;
        };
        if best_remaining_upper_bound < hits.last().expect("top_k threshold").score {
            break;
        }
    }

    sort_hits(&mut hits);
    hits.truncate(request.top_k.get());
    hits
}

fn scan_list(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    list: &IvfInvertedList,
    request: &IvfSearchRequest<'_>,
    hits: &mut Vec<IvfSearchHit>,
) {
    for &entry_index in &list.entry_indexes {
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

fn sort_hits(hits: &mut [IvfSearchHit]) {
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
}

fn best_remaining_upper_bound(
    inverted_lists: &[IvfInvertedList],
    query_vector: &[f32],
    remaining_list_scores: &[(usize, f32)],
) -> Option<f32> {
    remaining_list_scores
        .iter()
        .map(|&(list_index, centroid_score)| {
            inverted_lists[list_index]
                .residual_bound
                .upper_bound(centroid_score, query_vector)
        })
        .max_by(f32::total_cmp)
}
