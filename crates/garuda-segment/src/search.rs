use crate::types::{FlatSearchRequest, HnswSegmentSearchRequest, SegmentFilter, SegmentSearchHit};
use garuda_index_flat::FlatSearchHit;
use garuda_index_hnsw::{HnswHit, HnswIndex, HnswSearchRequest};
use garuda_meta::evaluate_filter;
use garuda_types::{DenseVector, HnswEfSearch, InternalDocId, Status, StatusCode, TopK};
use std::collections::HashMap;

pub fn search_flat(
    segment: &crate::SegmentFile,
    request: FlatSearchRequest<'_>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    let index = segment
        .flat_index
        .as_ref()
        .expect("flat segment search requires flat segment state");
    let record_indexes = live_record_indexes(segment);
    let hits = index.search(
        request.metric,
        request.query_vector,
        filtered_search_top_k(request.top_k, request.filter, record_indexes.len()),
    )?;
    collect_search_hits(
        segment,
        hits.into_iter().map(flat_hit),
        request.filter,
        record_indexes,
    )
}

pub fn search_hnsw(
    segment: &crate::SegmentFile,
    request: HnswSegmentSearchRequest<'_>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    let index = segment
        .hnsw_index
        .as_ref()
        .expect("hnsw segment search requires hnsw segment state");
    let record_indexes = live_record_indexes(segment);
    let hits = run_hnsw_search(
        index,
        request.query_vector,
        filtered_search_top_k(request.top_k, request.filter, record_indexes.len()),
        request.ef_search,
    )?;
    collect_search_hits(
        segment,
        hits.into_iter().map(hnsw_hit),
        request.filter,
        record_indexes,
    )
}

fn live_record_indexes(
    segment: &crate::SegmentFile,
) -> HashMap<garuda_types::InternalDocId, usize> {
    let mut record_indexes = HashMap::new();

    for (record_index, record) in segment.records.iter().enumerate() {
        if matches!(record.state, crate::RecordState::Deleted) {
            continue;
        }

        record_indexes.insert(record.doc_id, record_index);
    }

    record_indexes
}

fn collect_search_hits(
    segment: &crate::SegmentFile,
    hits: impl IntoIterator<Item = (InternalDocId, f32)>,
    filter: SegmentFilter<'_>,
    mut record_indexes: HashMap<garuda_types::InternalDocId, usize>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    let mut search_hits = Vec::new();

    for (doc_id, score) in hits {
        let Some(record_index) = record_indexes.remove(&doc_id) else {
            return Err(Status::err(
                StatusCode::Internal,
                "flat index hit missing backing record",
            ));
        };

        let record = segment.records[record_index].clone();

        if let SegmentFilter::Matching(filter) = filter {
            if !evaluate_filter(filter, &record.doc.fields) {
                continue;
            }
        }

        search_hits.push(SegmentSearchHit { record, score });
    }

    Ok(search_hits)
}

fn run_hnsw_search(
    index: &HnswIndex,
    query_vector: &DenseVector,
    top_k: TopK,
    ef_search: HnswEfSearch,
) -> Result<Vec<HnswHit>, Status> {
    index.search(HnswSearchRequest::new(query_vector, top_k, ef_search))
}

fn filtered_search_top_k(top_k: TopK, filter: SegmentFilter<'_>, live_doc_count: usize) -> TopK {
    if matches!(filter, SegmentFilter::All) {
        return top_k;
    }

    TopK::new(live_doc_count).expect("queryable segment must have at least one live document")
}

fn flat_hit(hit: FlatSearchHit) -> (InternalDocId, f32) {
    (hit.doc_id, hit.score)
}

fn hnsw_hit(hit: HnswHit) -> (InternalDocId, f32) {
    (hit.doc_id, hit.score)
}
