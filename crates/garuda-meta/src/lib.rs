use garuda_types::{DocId, FilterExpr, ScalarValue};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IdMap {
    entries: HashMap<DocId, u64>,
}

impl IdMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn insert(&mut self, doc_id: DocId, internal_doc_id: u64) {
        self.entries.insert(doc_id, internal_doc_id);
    }

    pub fn contains(&self, doc_id: &DocId) -> bool {
        self.entries.contains_key(doc_id)
    }

    pub fn get(&self, doc_id: &DocId) -> Option<u64> {
        self.entries.get(doc_id).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&DocId, &u64)> {
        self.entries.iter()
    }
}

impl From<HashMap<DocId, u64>> for IdMap {
    fn from(entries: HashMap<DocId, u64>) -> Self {
        Self { entries }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeleteStore {
    deleted_doc_ids: HashSet<u64>,
}

impl DeleteStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.deleted_doc_ids.clear();
    }

    pub fn insert(&mut self, internal_doc_id: u64) {
        self.deleted_doc_ids.insert(internal_doc_id);
    }

    pub fn contains(&self, internal_doc_id: u64) -> bool {
        self.deleted_doc_ids.contains(&internal_doc_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &u64> {
        self.deleted_doc_ids.iter()
    }
}

impl From<HashSet<u64>> for DeleteStore {
    fn from(deleted_doc_ids: HashSet<u64>) -> Self {
        Self { deleted_doc_ids }
    }
}

pub fn evaluate_filter(expr: &FilterExpr, fields: &BTreeMap<String, ScalarValue>) -> bool {
    match expr {
        FilterExpr::Eq(field, value) => fields.get(field) == Some(value),
        FilterExpr::Ne(field, value) => fields
            .get(field)
            .is_some_and(|candidate| candidate != value),
        FilterExpr::Gt(field, value) => compare_field_value(fields, field, value, Ordering::is_gt),
        FilterExpr::Gte(field, value) => compare_field_value(fields, field, value, Ordering::is_ge),
        FilterExpr::Lt(field, value) => compare_field_value(fields, field, value, Ordering::is_lt),
        FilterExpr::Lte(field, value) => compare_field_value(fields, field, value, Ordering::is_le),
        FilterExpr::And(lhs, rhs) => evaluate_filter(lhs, fields) && evaluate_filter(rhs, fields),
        FilterExpr::Or(lhs, rhs) => evaluate_filter(lhs, fields) || evaluate_filter(rhs, fields),
    }
}

fn compare_field_value(
    fields: &BTreeMap<String, ScalarValue>,
    field: &str,
    value: &ScalarValue,
    predicate: impl FnOnce(Ordering) -> bool,
) -> bool {
    let Some(candidate) = fields.get(field) else {
        return false;
    };

    let Some(ordering) = compare_values(candidate, value) else {
        return false;
    };

    predicate(ordering)
}

fn compare_values(lhs: &ScalarValue, rhs: &ScalarValue) -> Option<Ordering> {
    match (lhs, rhs) {
        (ScalarValue::Bool(a), ScalarValue::Bool(b)) => Some(a.cmp(b)),
        (ScalarValue::Int64(a), ScalarValue::Int64(b)) => Some(a.cmp(b)),
        (ScalarValue::Float64(a), ScalarValue::Float64(b)) => a.partial_cmp(b),
        (ScalarValue::String(a), ScalarValue::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}
