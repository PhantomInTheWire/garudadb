use garuda_math::score_doc;
use garuda_types::{
    DenseVector, DistanceMetric, InternalDocId, IvfIndexParams, IvfProbeCount, Status, StatusCode,
    TopK, VectorDimension,
};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct IvfBuildEntry {
    doc_id: InternalDocId,
    vector: DenseVector,
}

impl IvfBuildEntry {
    pub fn new(
        dimension: VectorDimension,
        doc_id: InternalDocId,
        vector: DenseVector,
    ) -> Result<Self, Status> {
        if vector.len() != dimension.get() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "ivf index entry dimension does not match index dimension",
            ));
        }

        Ok(Self { doc_id, vector })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvfSearchHit {
    pub doc_id: InternalDocId,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvfStoredLists {
    pub centroids: Vec<DenseVector>,
    pub doc_ids_by_list: Vec<Vec<InternalDocId>>,
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

#[derive(Clone, Debug, PartialEq)]
pub struct WritingIvfIndex {
    config: IvfIndexConfig,
    state: IvfState,
}

#[derive(Clone, Debug, PartialEq)]
struct IvfState {
    entries: Vec<IvfBuildEntry>,
    centroids: Vec<DenseVector>,
    list_entry_indexes: Vec<Vec<usize>>,
}

impl IvfIndex {
    pub fn build(config: IvfIndexConfig, entries: Vec<IvfBuildEntry>) -> Result<Self, Status> {
        let trained = train_lists(&config, &entries)?;
        Ok(Self {
            config,
            state: IvfState::new(entries, trained.centroids, trained.list_entry_indexes),
        })
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
}

impl WritingIvfIndex {
    pub fn new(config: IvfIndexConfig) -> Self {
        Self {
            config,
            state: IvfState::empty(),
        }
    }

    pub fn from_entries_incremental(config: IvfIndexConfig, entries: Vec<IvfBuildEntry>) -> Self {
        let mut index = Self::new(config);

        for entry in entries {
            index.insert(entry);
        }

        index
    }

    pub fn train(self) -> Result<IvfIndex, Status> {
        IvfIndex::build(self.config, self.state.entries)
    }

    pub fn search(&self, request: IvfSearchRequest<'_>) -> Result<Vec<IvfSearchHit>, Status> {
        self.state.search(&self.config, request)
    }

    pub fn active_list_count(&self) -> usize {
        self.state.list_count()
    }

    pub fn insert(&mut self, entry: IvfBuildEntry) {
        self.state.insert_incremental(&self.config, entry);
    }
}

impl IvfState {
    fn new(
        entries: Vec<IvfBuildEntry>,
        centroids: Vec<DenseVector>,
        list_entry_indexes: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            entries,
            centroids,
            list_entry_indexes,
        }
    }

    fn empty() -> Self {
        Self::new(Vec::new(), Vec::new(), Vec::new())
    }

    fn search(
        &self,
        config: &IvfIndexConfig,
        request: IvfSearchRequest<'_>,
    ) -> Result<Vec<IvfSearchHit>, Status> {
        search_entries(
            config,
            &self.entries,
            &self.centroids,
            &self.list_entry_indexes,
            request,
        )
    }

    fn stored_lists(&self) -> IvfStoredLists {
        let mut doc_ids_by_list = Vec::with_capacity(self.list_entry_indexes.len());

        for list in &self.list_entry_indexes {
            let mut doc_ids = Vec::with_capacity(list.len());

            for &entry_index in list {
                doc_ids.push(self.entries[entry_index].doc_id);
            }

            doc_ids_by_list.push(doc_ids);
        }

        IvfStoredLists {
            centroids: self.centroids.clone(),
            doc_ids_by_list,
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn list_count(&self) -> usize {
        self.centroids.len()
    }

    fn insert_incremental(&mut self, config: &IvfIndexConfig, entry: IvfBuildEntry) {
        let entry_index = self.entries.len();
        self.entries.push(entry);

        if self.centroids.is_empty() {
            self.centroids
                .push(self.entries[entry_index].vector.clone());
            self.list_entry_indexes.push(vec![entry_index]);
            return;
        }

        let list_count = config.list_count(self.entries.len());
        if self.centroids.len() < list_count {
            self.centroids
                .push(self.entries[entry_index].vector.clone());
            self.list_entry_indexes.push(vec![entry_index]);
            return;
        }

        let list_index = nearest_centroid(
            config.metric,
            &self.entries[entry_index].vector,
            &self.centroids,
        );
        self.list_entry_indexes[list_index].push(entry_index);
        self.centroids[list_index] = centroid_for_list(
            config.dimension,
            &self.entries,
            &self.list_entry_indexes[list_index],
        );
    }
}

#[derive(Clone)]
struct TrainedLists {
    centroids: Vec<DenseVector>,
    list_entry_indexes: Vec<Vec<usize>>,
}

fn train_lists(config: &IvfIndexConfig, entries: &[IvfBuildEntry]) -> Result<TrainedLists, Status> {
    if entries.is_empty() {
        return Ok(TrainedLists {
            centroids: Vec::new(),
            list_entry_indexes: Vec::new(),
        });
    }

    let list_count = config.list_count(entries.len());
    let mut centroids = initialize_centroids(config.metric, entries, list_count);
    let mut assignments = vec![0usize; entries.len()];

    for _ in 0..config.params.training_iterations.get() {
        assignments = assign_entries(config.metric, entries, &centroids);
        centroids = recompute_centroids(config.dimension, list_count, entries, &assignments);
    }

    let mut list_entry_indexes = vec![Vec::new(); list_count];

    for (entry_index, &list_index) in assignments.iter().enumerate() {
        list_entry_indexes[list_index].push(entry_index);
    }

    Ok(TrainedLists {
        centroids,
        list_entry_indexes,
    })
}

fn assign_entries(
    metric: DistanceMetric,
    entries: &[IvfBuildEntry],
    centroids: &[DenseVector],
) -> Vec<usize> {
    let mut assignments = Vec::with_capacity(entries.len());

    for entry in entries {
        assignments.push(nearest_centroid(metric, &entry.vector, centroids));
    }

    assignments
}

fn nearest_centroid(
    metric: DistanceMetric,
    vector: &DenseVector,
    centroids: &[DenseVector],
) -> usize {
    let mut best_index = 0usize;
    let mut best_score = score_doc(metric, vector.as_slice(), centroids[0].as_slice());

    for (index, centroid) in centroids.iter().enumerate().skip(1) {
        let score = score_doc(metric, vector.as_slice(), centroid.as_slice());
        let is_better = score > best_score;

        if is_better {
            best_index = index;
            best_score = score;
        }
    }

    best_index
}

fn recompute_centroids(
    dimension: VectorDimension,
    list_count: usize,
    entries: &[IvfBuildEntry],
    assignments: &[usize],
) -> Vec<DenseVector> {
    let dimension = dimension.get();
    let mut sums = vec![vec![0.0; dimension]; list_count];
    let mut counts = vec![0usize; list_count];

    for (entry, &list_index) in entries.iter().zip(assignments) {
        counts[list_index] += 1;

        for (sum, value) in sums[list_index].iter_mut().zip(entry.vector.as_slice()) {
            *sum += *value;
        }
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

    centroids
}

fn initialize_centroids(
    metric: DistanceMetric,
    entries: &[IvfBuildEntry],
    list_count: usize,
) -> Vec<DenseVector> {
    let mut centroid_indexes = Vec::with_capacity(list_count);
    centroid_indexes.push(0);

    while centroid_indexes.len() < list_count {
        let next = next_centroid_index(metric, entries, &centroid_indexes);
        centroid_indexes.push(next);
    }

    let mut centroids = Vec::with_capacity(list_count);

    for index in centroid_indexes {
        centroids.push(entries[index].vector.clone());
    }

    centroids
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

fn centroid_for_list(
    dimension: VectorDimension,
    entries: &[IvfBuildEntry],
    entry_indexes: &[usize],
) -> DenseVector {
    let mut sums = vec![0.0; dimension.get()];

    for &entry_index in entry_indexes {
        for (sum, value) in sums.iter_mut().zip(entries[entry_index].vector.as_slice()) {
            *sum += *value;
        }
    }

    centroid_from_sums(&mut sums, entry_indexes.len())
}

fn centroid_from_sums(sums: &mut [f32], count: usize) -> DenseVector {
    let scale = count as f32;

    for value in sums.iter_mut() {
        *value /= scale;
    }

    DenseVector::parse(sums.to_vec()).expect("ivf centroid vector")
}

fn build_list_entry_indexes(
    entries: &[IvfBuildEntry],
    doc_ids_by_list: &[Vec<InternalDocId>],
) -> Result<Vec<Vec<usize>>, Status> {
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

            indexes.push(entry_index);
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

    for centroid in &stored.centroids {
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
    centroids: &[DenseVector],
    list_entry_indexes: &[Vec<usize>],
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

    let mut list_scores = Vec::with_capacity(centroids.len());

    for (list_index, centroid) in centroids.iter().enumerate() {
        list_scores.push((
            list_index,
            score_doc(
                config.metric,
                request.query_vector.as_slice(),
                centroid.as_slice(),
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
        for &entry_index in &list_entry_indexes[list_index] {
            let entry = &entries[entry_index];
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
