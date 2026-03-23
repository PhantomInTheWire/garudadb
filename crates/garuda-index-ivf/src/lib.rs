use garuda_math::score_doc;
use garuda_types::{
    DenseVector, DistanceMetric, InternalDocId, IvfIndexParams, IvfProbeCount,
    Status, StatusCode, TopK, VectorDimension,
};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct IvfBuildEntry {
    pub doc_id: InternalDocId,
    pub vector: DenseVector,
}

impl IvfBuildEntry {
    pub fn new(dimension: VectorDimension, doc_id: InternalDocId, vector: DenseVector) -> Result<Self, Status> {
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
    entries: Vec<IvfBuildEntry>,
    centroids: Vec<DenseVector>,
    list_entry_indexes: Vec<Vec<usize>>,
}

impl IvfIndex {
    pub fn empty(config: IvfIndexConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            centroids: Vec::new(),
            list_entry_indexes: Vec::new(),
        }
    }

    pub fn build(config: IvfIndexConfig, entries: Vec<IvfBuildEntry>) -> Result<Self, Status> {
        validate_entries(config.dimension, &entries)?;

        let trained = train_lists(&config, &entries)?;
        Ok(Self {
            config,
            entries,
            centroids: trained.centroids,
            list_entry_indexes: trained.list_entry_indexes,
        })
    }

    pub fn from_parts(
        config: IvfIndexConfig,
        entries: Vec<IvfBuildEntry>,
        stored: IvfStoredLists,
    ) -> Result<Self, Status> {
        validate_entries(config.dimension, &entries)?;
        validate_stored_lists(&config, &entries, &stored)?;

        let entry_indexes = build_list_entry_indexes(&entries, &stored.doc_ids_by_list)?;
        Ok(Self {
            config,
            entries,
            centroids: stored.centroids,
            list_entry_indexes: entry_indexes,
        })
    }

    pub fn search(&self, request: IvfSearchRequest<'_>) -> Result<Vec<IvfSearchHit>, Status> {
        search_entries(
            &self.config,
            &self.entries,
            &self.centroids,
            &self.list_entry_indexes,
            request,
        )
    }

    pub fn stored_lists(&self) -> IvfStoredLists {
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

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn list_count(&self) -> usize {
        self.centroids.len()
    }

    pub fn insert(&mut self, entry: IvfBuildEntry) {
        self.entries.push(entry);
        self.retrain().expect("writing ivf retrain");
    }

    fn retrain(&mut self) -> Result<(), Status> {
        let trained = train_lists(&self.config, &self.entries)?;
        self.centroids = trained.centroids;
        self.list_entry_indexes = trained.list_entry_indexes;
        Ok(())
    }
}

#[derive(Clone)]
struct TrainedLists {
    centroids: Vec<DenseVector>,
    list_entry_indexes: Vec<Vec<usize>>,
}

fn validate_entries(dimension: VectorDimension, entries: &[IvfBuildEntry]) -> Result<(), Status> {
    for entry in entries {
        if entry.vector.len() == dimension.get() {
            continue;
        }

        return Err(Status::err(
            StatusCode::InvalidArgument,
            "ivf index entry dimension does not match index dimension",
        ));
    }

    Ok(())
}

fn train_lists(config: &IvfIndexConfig, entries: &[IvfBuildEntry]) -> Result<TrainedLists, Status> {
    if entries.is_empty() {
        return Ok(TrainedLists {
            centroids: Vec::new(),
            list_entry_indexes: Vec::new(),
        });
    }

    let list_count = config.list_count(entries.len());
    let mut centroids = entries
        .iter()
        .take(list_count)
        .map(|entry| entry.vector.clone())
        .collect::<Vec<_>>();
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

fn nearest_centroid(metric: DistanceMetric, vector: &DenseVector, centroids: &[DenseVector]) -> usize {
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

        let scale = counts[list_index] as f32;
        for value in &mut sums[list_index] {
            *value /= scale;
        }

        centroids.push(DenseVector::parse(sums[list_index].clone()).expect("ivf centroid vector"));
    }

    centroids
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
            score_doc(config.metric, request.query_vector.as_slice(), centroid.as_slice()),
        ));
    }

    list_scores.sort_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
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

#[cfg(test)]
mod tests {
    use super::*;
    use garuda_types::{IvfListCount, IvfTrainingIterations};

    fn config() -> IvfIndexConfig {
        IvfIndexConfig::new(
            VectorDimension::new(2).expect("dimension"),
            DistanceMetric::Cosine,
            IvfIndexParams {
                n_list: IvfListCount::new(2).expect("n_list"),
                n_probe: IvfProbeCount::new(1).expect("n_probe"),
                training_iterations: IvfTrainingIterations::new(2).expect("iterations"),
            },
        )
    }

    fn entry(doc_id: u64, vector: [f32; 2]) -> IvfBuildEntry {
        IvfBuildEntry::new(
            VectorDimension::new(2).expect("dimension"),
            InternalDocId::new(doc_id).expect("doc_id"),
            DenseVector::parse(vector.to_vec()).expect("vector"),
        )
        .expect("entry")
    }

    #[test]
    fn search_rejects_dimension_mismatch() {
        let index = IvfIndex::build(config(), vec![entry(1, [1.0, 0.0])]).expect("build");
        let result = index.search(IvfSearchRequest::new(
            &DenseVector::parse(vec![1.0, 0.0, 0.0]).expect("vector"),
            TopK::new(1).expect("top_k"),
            IvfProbeCount::new(1).expect("nprobe"),
        ));

        assert_eq!(result.expect_err("dimension mismatch").code, StatusCode::InvalidArgument);
    }

    #[test]
    fn wider_nprobe_should_not_reduce_recall() {
        let entries = vec![
            entry(1, [1.0, 0.0]),
            entry(2, [0.9, 0.1]),
            entry(3, [0.0, 1.0]),
            entry(4, [0.1, 0.9]),
        ];
        let index = IvfIndex::build(config(), entries).expect("build");
        let query = DenseVector::parse(vec![0.95, 0.05]).expect("query");

        let narrow = index
            .search(IvfSearchRequest::new(
                &query,
                TopK::new(2).expect("top_k"),
                IvfProbeCount::new(1).expect("nprobe"),
            ))
            .expect("search");
        let wide = index
            .search(IvfSearchRequest::new(
                &query,
                TopK::new(2).expect("top_k"),
                IvfProbeCount::new(2).expect("nprobe"),
            ))
            .expect("search");

        assert_eq!(wide[0].doc_id, InternalDocId::new(1).expect("doc_id"));
        assert!(wide.len() >= narrow.len());
    }

    #[test]
    fn from_parts_rejects_duplicate_doc_ids() {
        let config = config();
        let entries = vec![entry(1, [1.0, 0.0]), entry(2, [0.0, 1.0])];
        let error = IvfIndex::from_parts(
            config,
            entries,
            IvfStoredLists {
                centroids: vec![
                    DenseVector::parse(vec![1.0, 0.0]).expect("centroid"),
                    DenseVector::parse(vec![0.0, 1.0]).expect("centroid"),
                ],
                doc_ids_by_list: vec![
                    vec![InternalDocId::new(1).expect("doc_id")],
                    vec![InternalDocId::new(1).expect("doc_id")],
                ],
            },
        )
        .expect_err("duplicate doc ids should fail");

        assert_eq!(error.code, StatusCode::Internal);
    }

    #[test]
    fn from_parts_rejects_missing_doc_ids() {
        let config = config();
        let entries = vec![entry(1, [1.0, 0.0]), entry(2, [0.0, 1.0])];
        let error = IvfIndex::from_parts(
            config,
            entries,
            IvfStoredLists {
                centroids: vec![
                    DenseVector::parse(vec![1.0, 0.0]).expect("centroid"),
                    DenseVector::parse(vec![0.0, 1.0]).expect("centroid"),
                ],
                doc_ids_by_list: vec![vec![InternalDocId::new(1).expect("doc_id")], vec![]],
            },
        )
        .expect_err("missing doc ids should fail");

        assert_eq!(error.code, StatusCode::Internal);
    }
}
