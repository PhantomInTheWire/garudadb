use garuda_types::{InternalDocId, ScalarCompareOp, ScalarPredicate, ScalarType, ScalarValue};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarIndex {
    Bool(BoolScalarIndex),
    Int64(OrderedScalarIndex<i64>),
    Float64(OrderedScalarIndex<FloatKey>),
    String(OrderedScalarIndex<String>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarIndexData {
    Bool {
        false_doc_ids: Vec<InternalDocId>,
        true_doc_ids: Vec<InternalDocId>,
    },
    Int64(Vec<(i64, Vec<InternalDocId>)>),
    Float64(Vec<(f64, Vec<InternalDocId>)>),
    String(Vec<(String, Vec<InternalDocId>)>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoolScalarIndex {
    false_doc_ids: Vec<InternalDocId>,
    true_doc_ids: Vec<InternalDocId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrderedScalarIndex<K> {
    postings: BTreeMap<K, Vec<InternalDocId>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatKey(f64);

impl ScalarIndex {
    pub fn new(field_type: ScalarType) -> Self {
        match field_type {
            ScalarType::Bool => Self::Bool(BoolScalarIndex::new()),
            ScalarType::Int64 => Self::Int64(OrderedScalarIndex::new()),
            ScalarType::Float64 => Self::Float64(OrderedScalarIndex::new()),
            ScalarType::String => Self::String(OrderedScalarIndex::new()),
        }
    }

    pub fn insert(&mut self, doc_id: InternalDocId, value: &ScalarValue) {
        match (self, value) {
            (Self::Bool(index), ScalarValue::Bool(value)) => index.insert(doc_id, *value),
            (Self::Int64(index), ScalarValue::Int64(value)) => index.insert(doc_id, *value),
            (Self::Float64(index), ScalarValue::Float64(value)) => {
                index.insert(doc_id, FloatKey::new(*value))
            }
            (Self::String(index), ScalarValue::String(value)) => {
                index.insert(doc_id, value.clone())
            }
            _ => panic!("scalar index value type should match schema"),
        }
    }

    pub fn matching_doc_ids(&self, predicate: &ScalarPredicate) -> HashSet<InternalDocId> {
        match (self, &predicate.value) {
            (Self::Bool(index), ScalarValue::Bool(value)) => {
                index.matching_doc_ids(predicate.op, *value)
            }
            (Self::Int64(index), ScalarValue::Int64(value)) => {
                index.matching_doc_ids(predicate.op, *value)
            }
            (Self::Float64(index), ScalarValue::Float64(value)) => {
                index.matching_doc_ids(predicate.op, FloatKey::new(*value))
            }
            (Self::String(index), ScalarValue::String(value)) => {
                index.matching_doc_ids(predicate.op, value.clone())
            }
            _ => panic!("scalar predicate value type should match index type"),
        }
    }

    pub fn data(&self) -> ScalarIndexData {
        match self {
            Self::Bool(index) => ScalarIndexData::Bool {
                false_doc_ids: index.false_doc_ids.clone(),
                true_doc_ids: index.true_doc_ids.clone(),
            },
            Self::Int64(index) => ScalarIndexData::Int64(index.data()),
            Self::Float64(index) => ScalarIndexData::Float64(
                index
                    .data()
                    .into_iter()
                    .map(|(key, doc_ids)| (key.get(), doc_ids))
                    .collect(),
            ),
            Self::String(index) => ScalarIndexData::String(index.data()),
        }
    }

    pub fn from_data(data: ScalarIndexData) -> Self {
        match data {
            ScalarIndexData::Bool {
                false_doc_ids,
                true_doc_ids,
            } => Self::Bool(BoolScalarIndex {
                false_doc_ids,
                true_doc_ids,
            }),
            ScalarIndexData::Int64(postings) => {
                Self::Int64(OrderedScalarIndex::from_data(postings))
            }
            ScalarIndexData::Float64(postings) => Self::Float64(OrderedScalarIndex::from_data(
                postings
                    .into_iter()
                    .map(|(key, doc_ids)| (FloatKey::new(key), doc_ids))
                    .collect(),
            )),
            ScalarIndexData::String(postings) => {
                Self::String(OrderedScalarIndex::from_data(postings))
            }
        }
    }
}

impl BoolScalarIndex {
    fn new() -> Self {
        Self {
            false_doc_ids: Vec::new(),
            true_doc_ids: Vec::new(),
        }
    }

    fn insert(&mut self, doc_id: InternalDocId, value: bool) {
        if value {
            self.true_doc_ids.push(doc_id);
            return;
        }

        self.false_doc_ids.push(doc_id);
    }

    fn matching_doc_ids(&self, op: ScalarCompareOp, value: bool) -> HashSet<InternalDocId> {
        assert_eq!(
            op,
            ScalarCompareOp::Eq,
            "bool scalar index supports equality only"
        );

        if value {
            return self.true_doc_ids.iter().copied().collect();
        }

        self.false_doc_ids.iter().copied().collect()
    }
}

impl<K> OrderedScalarIndex<K>
where
    K: Clone + Ord,
{
    fn new() -> Self {
        Self {
            postings: BTreeMap::new(),
        }
    }

    fn from_data(postings: Vec<(K, Vec<InternalDocId>)>) -> Self {
        Self {
            postings: postings.into_iter().collect(),
        }
    }

    fn insert(&mut self, doc_id: InternalDocId, key: K) {
        self.postings.entry(key).or_default().push(doc_id);
    }

    fn data(&self) -> Vec<(K, Vec<InternalDocId>)> {
        self.postings
            .iter()
            .map(|(key, doc_ids)| (key.clone(), doc_ids.clone()))
            .collect()
    }

    fn matching_doc_ids(&self, op: ScalarCompareOp, key: K) -> HashSet<InternalDocId> {
        match op {
            ScalarCompareOp::Eq => self.eq_doc_ids(&key),
            ScalarCompareOp::Gt => {
                self.range_doc_ids((std::ops::Bound::Excluded(key), std::ops::Bound::Unbounded))
            }
            ScalarCompareOp::Gte => {
                self.range_doc_ids((std::ops::Bound::Included(key), std::ops::Bound::Unbounded))
            }
            ScalarCompareOp::Lt => {
                self.range_doc_ids((std::ops::Bound::Unbounded, std::ops::Bound::Excluded(key)))
            }
            ScalarCompareOp::Lte => {
                self.range_doc_ids((std::ops::Bound::Unbounded, std::ops::Bound::Included(key)))
            }
        }
    }

    fn eq_doc_ids(&self, key: &K) -> HashSet<InternalDocId> {
        let Some(doc_ids) = self.postings.get(key) else {
            return HashSet::new();
        };

        doc_ids.iter().copied().collect()
    }

    fn range_doc_ids<R>(&self, range: R) -> HashSet<InternalDocId>
    where
        R: std::ops::RangeBounds<K>,
    {
        let mut doc_ids = HashSet::new();

        for (_, posting_list) in self.postings.range(range) {
            doc_ids.extend(posting_list.iter().copied());
        }

        doc_ids
    }
}

impl FloatKey {
    fn new(value: f64) -> Self {
        Self(value)
    }

    fn get(self) -> f64 {
        self.0
    }
}

impl Eq for FloatKey {}

impl Ord for FloatKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for FloatKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
