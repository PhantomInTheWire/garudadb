use crate::types::{
    FlatSearchRequest, HnswSegmentSearchRequest, PersistedSegment, RecordState, SegmentFilter,
    SegmentSearchHit, StoredRecord, WritingSegment,
};
use garuda_index_flat::FlatSearchHit;
use garuda_index_hnsw::HnswHit;
use garuda_meta::{DeleteStore, evaluate_filter};
use garuda_types::{HnswEfSearch, InternalDocId, Status, TopK};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct SearchScope<'a> {
    pub allowed_doc_ids: Option<&'a std::collections::HashSet<InternalDocId>>,
    pub delete_store: Option<&'a DeleteStore>,
}

impl SearchScope<'_> {
    fn contains(self, doc_id: InternalDocId) -> bool {
        if let Some(doc_ids) = self.allowed_doc_ids {
            if !doc_ids.contains(&doc_id) {
                return false;
            }
        }

        if let Some(delete_store) = self.delete_store {
            if delete_store.contains(doc_id) {
                return false;
            }
        }

        true
    }
}

pub fn search_writing_flat(
    segment: &WritingSegment,
    request: FlatSearchRequest<'_>,
    scope: SearchScope<'_>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    let index = segment
        .flat_index
        .as_ref()
        .expect("writing flat segment state");
    let record_indexes = live_record_indexes(&segment.records, scope);
    let hits = index.search(
        request.metric,
        request.query_vector,
        search_candidate_top_k(
            request.top_k,
            request.filter,
            &segment.records,
            &record_indexes,
        ),
    )?;
    Ok(collect_search_hits(
        &segment.records,
        hits.into_iter().map(flat_hit),
        request.filter,
        record_indexes,
    ))
}

pub fn search_writing_hnsw(
    segment: &WritingSegment,
    request: HnswSegmentSearchRequest<'_>,
    scope: SearchScope<'_>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    let index = segment
        .hnsw_index
        .as_ref()
        .expect("writing hnsw segment state");
    let record_indexes = live_record_indexes(&segment.records, scope);
    let hits = index.search(
        request.query_vector,
        search_candidate_top_k(
            request.top_k,
            request.filter,
            &segment.records,
            &record_indexes,
        ),
        request.ef_search,
    )?;
    Ok(collect_search_hits(
        &segment.records,
        hits.into_iter().map(hnsw_hit),
        request.filter,
        record_indexes,
    ))
}

pub fn search_persisted_flat(
    segment: &PersistedSegment,
    request: FlatSearchRequest<'_>,
    scope: SearchScope<'_>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    let index = segment
        .flat_index
        .as_ref()
        .expect("persisted flat segment state");
    let record_indexes = live_record_indexes(&segment.records, scope);
    let hits = index.search(
        request.metric,
        request.query_vector,
        search_candidate_top_k(
            request.top_k,
            request.filter,
            &segment.records,
            &record_indexes,
        ),
    )?;
    Ok(collect_search_hits(
        &segment.records,
        hits.into_iter().map(flat_hit),
        request.filter,
        record_indexes,
    ))
}

pub fn search_persisted_hnsw(
    segment: &PersistedSegment,
    request: HnswSegmentSearchRequest<'_>,
    scope: SearchScope<'_>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    let index = segment
        .hnsw_index
        .as_ref()
        .expect("persisted hnsw segment state");
    let record_indexes = live_record_indexes(&segment.records, scope);
    let candidate_top_k = search_candidate_top_k(
        request.top_k,
        request.filter,
        &segment.records,
        &record_indexes,
    );
    let hits = index.search(garuda_index_hnsw::HnswSearchRequest::new(
        request.query_vector,
        candidate_top_k,
        search_candidate_ef_search(request.ef_search, candidate_top_k),
    ))?;
    Ok(collect_search_hits(
        &segment.records,
        hits.into_iter().map(hnsw_hit),
        request.filter,
        record_indexes,
    ))
}

fn live_record_indexes(
    records: &[StoredRecord],
    scope: SearchScope<'_>,
) -> HashMap<InternalDocId, usize> {
    let mut record_indexes = HashMap::new();

    for (record_index, record) in records.iter().enumerate() {
        if matches!(record.state, RecordState::Deleted) || !scope.contains(record.doc_id) {
            continue;
        }

        record_indexes.insert(record.doc_id, record_index);
    }

    record_indexes
}

fn collect_search_hits(
    records: &[StoredRecord],
    hits: impl IntoIterator<Item = (InternalDocId, f32)>,
    filter: SegmentFilter<'_>,
    mut record_indexes: HashMap<InternalDocId, usize>,
) -> Vec<SegmentSearchHit> {
    let mut search_hits = Vec::new();

    for (doc_id, score) in hits {
        let Some(record_index) = record_indexes.remove(&doc_id) else {
            continue;
        };

        let record = records[record_index].clone();

        if let SegmentFilter::Matching(filter) = filter
            && !evaluate_filter(filter, &record.doc.fields)
        {
            continue;
        }

        search_hits.push(SegmentSearchHit { record, score });
    }

    search_hits
}

fn search_candidate_top_k(
    top_k: TopK,
    filter: SegmentFilter<'_>,
    records: &[StoredRecord],
    record_indexes: &HashMap<InternalDocId, usize>,
) -> TopK {
    if matches!(filter, SegmentFilter::All) && record_indexes.len() == records.len() {
        return top_k;
    }

    TopK::new(records.len().max(top_k.get())).expect("indexed docs or requested top_k")
}

fn search_candidate_ef_search(ef_search: HnswEfSearch, candidate_top_k: TopK) -> HnswEfSearch {
    let candidate_limit = candidate_top_k.get() as u32;
    let widened = ef_search.get().max(candidate_limit);
    HnswEfSearch::new(widened).expect("candidate ef_search should stay valid")
}

fn flat_hit(hit: FlatSearchHit) -> (InternalDocId, f32) {
    (hit.doc_id, hit.score)
}

fn hnsw_hit(hit: HnswHit) -> (InternalDocId, f32) {
    (hit.doc_id, hit.score)
}
