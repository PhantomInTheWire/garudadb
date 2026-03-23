use crate::types::{
    PersistedSegment, RecordState, SegmentFilter, SegmentSearchHit, SegmentSearchRequest,
    StoredRecord, WritingSegment,
};
use garuda_meta::{DeleteStore, evaluate_filter};
use garuda_types::{HnswEfSearch, InternalDocId, Status, TopK};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
struct SearchScope<'a> {
    pub allowed_doc_ids: Option<&'a std::collections::HashSet<InternalDocId>>,
    pub delete_store: Option<&'a DeleteStore>,
}

impl SearchScope<'_> {
    fn contains(self, doc_id: InternalDocId) -> bool {
        if let Some(doc_ids) = self.allowed_doc_ids
            && !doc_ids.contains(&doc_id)
        {
            return false;
        }

        if let Some(delete_store) = self.delete_store
            && delete_store.contains(doc_id)
        {
            return false;
        }

        true
    }
}

#[derive(Clone, Copy)]
enum SearchSegment<'a> {
    Writing(&'a WritingSegment),
    Persisted(&'a PersistedSegment),
}

impl SearchSegment<'_> {
    fn records(&self) -> &[StoredRecord] {
        match self {
            Self::Writing(segment) => &segment.records,
            Self::Persisted(segment) => &segment.records,
        }
    }
}

pub fn search_writing(
    segment: &WritingSegment,
    request: SegmentSearchRequest<'_>,
    allowed_doc_ids: Option<&HashSet<InternalDocId>>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    search(
        SearchSegment::Writing(segment),
        request,
        allowed_doc_ids,
        None,
    )
}

pub fn search_persisted(
    segment: &PersistedSegment,
    request: SegmentSearchRequest<'_>,
    allowed_doc_ids: Option<&HashSet<InternalDocId>>,
    delete_store: &DeleteStore,
) -> Result<Vec<SegmentSearchHit>, Status> {
    search(
        SearchSegment::Persisted(segment),
        request,
        allowed_doc_ids,
        Some(delete_store),
    )
}

fn search(
    segment: SearchSegment<'_>,
    request: SegmentSearchRequest<'_>,
    allowed_doc_ids: Option<&HashSet<InternalDocId>>,
    delete_store: Option<&DeleteStore>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    let scope = SearchScope {
        allowed_doc_ids,
        delete_store,
    };
    let records = segment.records();
    let record_indexes = live_record_indexes(records, scope);

    match request {
        SegmentSearchRequest::Flat(request) => {
            let hits = match segment {
                SearchSegment::Writing(segment) => segment
                    .flat_index
                    .as_ref()
                    .expect("writing flat index state")
                    .search(
                        request.metric,
                        request.query_vector,
                        search_candidate_top_k(
                            request.top_k,
                            request.filter,
                            records,
                            &record_indexes,
                        ),
                    )?,
                SearchSegment::Persisted(segment) => segment
                    .flat_index
                    .as_ref()
                    .expect("persisted flat index state")
                    .search(
                        request.metric,
                        request.query_vector,
                        search_candidate_top_k(
                            request.top_k,
                            request.filter,
                            records,
                            &record_indexes,
                        ),
                    )?,
            };

            Ok(collect_search_hits(
                records,
                hits.into_iter().map(|hit| (hit.doc_id, hit.score)),
                request.filter,
                record_indexes,
            ))
        }
        SegmentSearchRequest::Hnsw(request) => {
            let candidate_top_k =
                search_candidate_top_k(request.top_k, request.filter, records, &record_indexes);
            let hits = match segment {
                SearchSegment::Writing(segment) => segment
                    .hnsw_index
                    .as_ref()
                    .expect("writing hnsw index state")
                    .search(
                        request.query_vector,
                        candidate_top_k,
                        search_candidate_ef_search(request.ef_search, candidate_top_k),
                    )?,
                SearchSegment::Persisted(segment) => segment
                    .hnsw_index
                    .as_ref()
                    .expect("persisted hnsw index state")
                    .search(garuda_index_hnsw::HnswSearchRequest::new(
                        request.query_vector,
                        candidate_top_k,
                        search_candidate_ef_search(request.ef_search, candidate_top_k),
                    ))?,
            };

            Ok(collect_search_hits(
                records,
                hits.into_iter().map(|hit| (hit.doc_id, hit.score)),
                request.filter,
                record_indexes,
            ))
        }
    }
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
