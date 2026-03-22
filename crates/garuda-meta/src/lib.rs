use garuda_types::{DocId, FilterExpr, InternalDocId, ScalarValue};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IdMap {
    entries: HashMap<DocId, InternalDocId>,
}

impl IdMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn insert(&mut self, doc_id: DocId, internal_doc_id: InternalDocId) {
        self.entries.insert(doc_id, internal_doc_id);
    }

    pub fn contains(&self, doc_id: &DocId) -> bool {
        self.entries.contains_key(doc_id)
    }

    pub fn get(&self, doc_id: &DocId) -> Option<InternalDocId> {
        self.entries.get(doc_id).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&DocId, &InternalDocId)> {
        self.entries.iter()
    }
}

impl From<HashMap<DocId, InternalDocId>> for IdMap {
    fn from(entries: HashMap<DocId, InternalDocId>) -> Self {
        Self { entries }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeleteStore {
    deleted_doc_ids: HashSet<InternalDocId>,
}

impl DeleteStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.deleted_doc_ids.clear();
    }

    pub fn insert(&mut self, internal_doc_id: InternalDocId) {
        self.deleted_doc_ids.insert(internal_doc_id);
    }

    pub fn contains(&self, internal_doc_id: InternalDocId) -> bool {
        self.deleted_doc_ids.contains(&internal_doc_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &InternalDocId> {
        self.deleted_doc_ids.iter()
    }
}

impl From<HashSet<InternalDocId>> for DeleteStore {
    fn from(deleted_doc_ids: HashSet<InternalDocId>) -> Self {
        Self { deleted_doc_ids }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetadataStore {
    id_map: IdMap,
    delete_store: DeleteStore,
}

impl MetadataStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_parts(id_map: IdMap, delete_store: DeleteStore) -> Self {
        Self {
            id_map,
            delete_store,
        }
    }

    pub fn clear(&mut self) {
        self.id_map.clear();
        self.delete_store.clear();
    }

    pub fn index_live_doc(&mut self, doc_id: DocId, internal_doc_id: InternalDocId) {
        self.id_map.insert(doc_id, internal_doc_id);
    }

    pub fn mark_deleted(&mut self, internal_doc_id: InternalDocId) {
        self.delete_store.insert(internal_doc_id);
    }

    pub fn internal_doc_id(&self, doc_id: &DocId) -> Option<InternalDocId> {
        self.id_map.get(doc_id)
    }

    pub fn is_deleted(&self, internal_doc_id: InternalDocId) -> bool {
        self.delete_store.contains(internal_doc_id)
    }

    pub fn id_map_entries(&self) -> impl Iterator<Item = (&DocId, &InternalDocId)> {
        self.id_map.iter()
    }

    pub fn deleted_doc_ids(&self) -> impl Iterator<Item = &InternalDocId> {
        self.delete_store.iter()
    }

    pub fn delete_store(&self) -> &DeleteStore {
        &self.delete_store
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
