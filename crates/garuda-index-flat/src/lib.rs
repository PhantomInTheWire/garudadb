use garuda_math::score_doc;
use garuda_types::{
    DenseVector, DistanceMetric, InternalDocId, Status, StatusCode, TopK, VectorDimension,
};

#[derive(Clone, Debug, PartialEq)]
pub struct FlatIndexEntry {
    pub doc_id: InternalDocId,
    pub vector: DenseVector,
}

impl FlatIndexEntry {
    pub fn new(doc_id: InternalDocId, vector: DenseVector) -> Self {
        Self { doc_id, vector }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlatSearchHit {
    pub doc_id: InternalDocId,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlatIndex {
    dimension: VectorDimension,
    entries: Vec<FlatIndexEntry>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WritingFlatIndex {
    dimension: VectorDimension,
    entries: Vec<FlatIndexEntry>,
}

impl FlatIndex {
    pub fn build(dimension: VectorDimension, entries: Vec<FlatIndexEntry>) -> Result<Self, Status> {
        for entry in &entries {
            if entry.vector.len() == dimension.get() {
                continue;
            }

            return Err(Status::err(
                StatusCode::InvalidArgument,
                "flat index entry dimension does not match index dimension",
            ));
        }

        Ok(Self { dimension, entries })
    }

    pub fn search(
        &self,
        metric: DistanceMetric,
        query_vector: &DenseVector,
        top_k: TopK,
    ) -> Result<Vec<FlatSearchHit>, Status> {
        search_entries(self.dimension, &self.entries, metric, query_vector, top_k)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl WritingFlatIndex {
    pub fn new(dimension: VectorDimension) -> Self {
        Self {
            dimension,
            entries: Vec::new(),
        }
    }

    pub fn insert(&mut self, doc_id: InternalDocId, vector: DenseVector) {
        assert_eq!(
            vector.len(),
            self.dimension.get(),
            "writing flat index entry dimension"
        );
        self.entries.push(FlatIndexEntry::new(doc_id, vector));
    }

    pub fn search(
        &self,
        metric: DistanceMetric,
        query_vector: &DenseVector,
        top_k: TopK,
    ) -> Result<Vec<FlatSearchHit>, Status> {
        search_entries(self.dimension, &self.entries, metric, query_vector, top_k)
    }

    pub fn entries(&self) -> &[FlatIndexEntry] {
        &self.entries
    }
}

fn search_entries(
    dimension: VectorDimension,
    entries: &[FlatIndexEntry],
    metric: DistanceMetric,
    query_vector: &DenseVector,
    top_k: TopK,
) -> Result<Vec<FlatSearchHit>, Status> {
    if query_vector.len() != dimension.get() {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "query vector dimension does not match flat index dimension",
        ));
    }

    let mut hits = Vec::with_capacity(entries.len());

    for entry in entries {
        hits.push(FlatSearchHit {
            doc_id: entry.doc_id,
            score: score_doc(metric, query_vector.as_slice(), entry.vector.as_slice()),
        });
    }

    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
    hits.truncate(top_k.get());

    Ok(hits)
}
