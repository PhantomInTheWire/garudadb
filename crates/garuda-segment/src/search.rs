use crate::types::{
    PersistedSegment, RecordState, SegmentExecutionRequest, SegmentFilter, SegmentFilterContext,
    SegmentSearchHit, StoredRecord, WritingSegment,
};

#[cfg(test)]
#[path = "search_candidate_nprobe_tests.rs"]
mod search_candidate_nprobe_tests;

use garuda_index_flat::FlatSearchHit;
use garuda_index_hnsw::HnswHit;
use garuda_index_ivf::{IvfIndex, IvfSearchHit, WritingIvfIndex};
use garuda_meta::evaluate_filter;
use garuda_types::{
    AnnBudgetPolicy, HnswEfSearch, InternalDocId, IvfProbeCount, RecallPlan, Status, TopK,
};
use std::collections::HashMap;

impl SegmentFilterContext<'_> {
    fn hides_deleted(self) -> bool {
        self.delete_store.is_some()
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

    fn ivf_index(&self) -> IvfSearchIndex<'_> {
        match self {
            Self::Writing(segment) => IvfSearchIndex::Writing(
                segment.ivf_index.as_ref().expect("writing ivf index state"),
            ),
            Self::Persisted(segment) => IvfSearchIndex::Persisted(
                segment
                    .ivf_index
                    .as_ref()
                    .expect("persisted ivf index state"),
            ),
        }
    }
}

enum IvfSearchIndex<'a> {
    Writing(&'a WritingIvfIndex),
    Persisted(&'a IvfIndex),
}

impl IvfSearchIndex<'_> {
    fn populated_list_count(&self) -> usize {
        match self {
            Self::Writing(index) => index.populated_list_count(),
            Self::Persisted(index) => index.populated_list_count(),
        }
    }

    fn search(
        &self,
        request: garuda_index_ivf::IvfSearchRequest<'_>,
    ) -> Result<Vec<IvfSearchHit>, Status> {
        match self {
            Self::Writing(index) => index.search(request),
            Self::Persisted(index) => index.search(request),
        }
    }
}

pub fn search_writing(
    segment: &WritingSegment,
    request: SegmentExecutionRequest<'_>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    assert!(
        request.filter.delete_store.is_none(),
        "writing delete filter"
    );
    search_segment(SearchSegment::Writing(segment), request)
}

pub fn search_persisted(
    segment: &PersistedSegment,
    request: SegmentExecutionRequest<'_>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    assert!(
        request.filter.delete_store.is_some(),
        "persisted delete filter"
    );
    search_segment(SearchSegment::Persisted(segment), request)
}

fn search_segment(
    segment: SearchSegment<'_>,
    request: SegmentExecutionRequest<'_>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    let records = segment.records();
    let stats = search_stats(records, request.filter);

    match request.recall {
        RecallPlan::Flat(recall) => {
            let candidate_top_k = TopK::new(stats.visible_doc_count.max(recall.top_k.get()))
                .expect("segment live doc count");
            let hits = match segment {
                SearchSegment::Writing(segment) => segment
                    .flat_index
                    .as_ref()
                    .expect("writing flat index state")
                    .search(request.metric, request.query_vector, candidate_top_k)?,
                SearchSegment::Persisted(segment) => segment
                    .flat_index
                    .as_ref()
                    .expect("persisted flat index state")
                    .search(request.metric, request.query_vector, candidate_top_k)?,
            };

            Ok(collect_search_hits(
                records,
                hits.into_iter().map(flat_hit),
                request.filter.residual,
                stats.record_indexes,
            ))
        }
        RecallPlan::Hnsw(recall) => {
            let candidate_top_k = ann_candidate_top_k(
                recall.top_k,
                recall.budget,
                request.filter,
                stats.candidate_doc_count,
                stats.visible_doc_count,
            );
            let hits = match segment {
                SearchSegment::Writing(segment) => segment
                    .hnsw_index
                    .as_ref()
                    .expect("writing hnsw index state")
                    .search(
                        request.query_vector,
                        candidate_top_k,
                        search_candidate_ef_search(recall.ef_search, candidate_top_k),
                    )?,
                SearchSegment::Persisted(segment) => segment
                    .hnsw_index
                    .as_ref()
                    .expect("persisted hnsw index state")
                    .search(garuda_index_hnsw::HnswSearchRequest::new(
                        request.query_vector,
                        candidate_top_k,
                        search_candidate_ef_search(recall.ef_search, candidate_top_k),
                    ))?,
            };

            Ok(collect_search_hits(
                records,
                hits.into_iter().map(hnsw_hit),
                request.filter.residual,
                stats.record_indexes,
            ))
        }
        RecallPlan::Ivf(recall) => {
            let candidate_top_k = ann_candidate_top_k(
                recall.top_k,
                recall.budget,
                request.filter,
                stats.candidate_doc_count,
                stats.visible_doc_count,
            );
            let hits = search_ivf_hits(
                segment,
                request.query_vector,
                recall,
                candidate_top_k,
                stats.candidate_doc_count,
                stats.visible_doc_count,
                stats.allowed_visible_doc_count,
            )?;

            Ok(collect_search_hits(
                records,
                hits.into_iter().map(ivf_hit),
                request.filter.residual,
                stats.record_indexes,
            ))
        }
    }
}

struct SearchStats {
    candidate_doc_count: usize,
    visible_doc_count: usize,
    allowed_visible_doc_count: usize,
    record_indexes: HashMap<InternalDocId, usize>,
}

pub(crate) struct CandidateNprobeInput {
    pub(crate) nprobe: IvfProbeCount,
    pub(crate) top_k: TopK,
    pub(crate) budget: AnnBudgetPolicy,
    pub(crate) candidate_top_k: TopK,
    pub(crate) candidate_doc_count: usize,
    pub(crate) visible_doc_count: usize,
    pub(crate) allowed_visible_doc_count: usize,
    pub(crate) populated_list_count: usize,
}

fn search_ivf_hits(
    segment: SearchSegment<'_>,
    query_vector: &garuda_types::DenseVector,
    recall: garuda_types::IvfRecallPlan,
    candidate_top_k: TopK,
    indexed_doc_count: usize,
    visible_doc_count: usize,
    allowed_visible_doc_count: usize,
) -> Result<Vec<IvfSearchHit>, Status> {
    let index = segment.ivf_index();
    let request = garuda_index_ivf::IvfSearchRequest::new(
        query_vector,
        candidate_top_k,
        search_candidate_nprobe(CandidateNprobeInput {
            nprobe: recall.nprobe,
            top_k: recall.top_k,
            budget: recall.budget,
            candidate_top_k,
            candidate_doc_count: indexed_doc_count,
            visible_doc_count,
            allowed_visible_doc_count,
            populated_list_count: index.populated_list_count(),
        }),
    );

    index.search(request)
}

fn search_stats(records: &[StoredRecord], filter: SegmentFilterContext<'_>) -> SearchStats {
    let candidate_doc_count = records.len();
    let mut visible_doc_count = 0;
    let mut allowed_visible_doc_count = 0;
    let mut record_indexes = HashMap::new();

    for (record_index, record) in records.iter().enumerate() {
        if matches!(record.state, RecordState::Deleted) {
            continue;
        }

        if let Some(delete_store) = filter.delete_store
            && delete_store.contains(record.doc_id)
        {
            continue;
        }

        visible_doc_count += 1;

        if let Some(doc_ids) = filter.allowed_doc_ids
            && !doc_ids.contains(&record.doc_id)
        {
            continue;
        }

        allowed_visible_doc_count += 1;
        record_indexes.insert(record.doc_id, record_index);
    }

    SearchStats {
        candidate_doc_count,
        visible_doc_count,
        allowed_visible_doc_count,
        record_indexes,
    }
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

fn ann_candidate_top_k(
    top_k: TopK,
    budget: AnnBudgetPolicy,
    filter: SegmentFilterContext<'_>,
    indexed_doc_count: usize,
    visible_doc_count: usize,
) -> TopK {
    if delete_visibility_hides_docs(filter, indexed_doc_count, visible_doc_count) {
        return TopK::new(visible_doc_count.max(top_k.get())).expect("segment live doc count");
    }

    if matches!(budget, AnnBudgetPolicy::Requested)
        || !filtering_can_drop_hits(filter, indexed_doc_count, visible_doc_count)
    {
        return top_k;
    }

    TopK::new(visible_doc_count.max(top_k.get())).expect("segment live doc count")
}

fn filtering_can_drop_hits(
    filter: SegmentFilterContext<'_>,
    indexed_doc_count: usize,
    visible_doc_count: usize,
) -> bool {
    if filter.allowed_doc_ids.is_some() || matches!(filter.residual, SegmentFilter::Matching(_)) {
        return true;
    }

    delete_visibility_hides_docs(filter, indexed_doc_count, visible_doc_count)
}

fn delete_visibility_hides_docs(
    filter: SegmentFilterContext<'_>,
    indexed_doc_count: usize,
    visible_doc_count: usize,
) -> bool {
    filter.hides_deleted() && visible_doc_count < indexed_doc_count
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

fn ivf_hit(hit: IvfSearchHit) -> (InternalDocId, f32) {
    (hit.doc_id, hit.score)
}

pub(crate) fn search_candidate_nprobe(input: CandidateNprobeInput) -> IvfProbeCount {
    if matches!(input.budget, AnnBudgetPolicy::Requested)
        && input.candidate_top_k == input.top_k
        && input.candidate_doc_count == input.visible_doc_count
    {
        return input.nprobe;
    }

    if input.populated_list_count <= input.candidate_top_k.get() {
        return IvfProbeCount::new(input.populated_list_count as u32)
            .expect("small list count should fit nprobe");
    }

    if input.candidate_doc_count > input.visible_doc_count {
        return IvfProbeCount::new(input.populated_list_count as u32)
            .expect("populated list count should fit nprobe");
    }

    if input.allowed_visible_doc_count == 0 {
        return IvfProbeCount::new(input.populated_list_count as u32)
            .expect("populated list count should fit nprobe");
    }

    let requested_nprobe = input.nprobe.get() as usize;
    let widened = if input.allowed_visible_doc_count < input.visible_doc_count {
        requested_nprobe
            .saturating_mul(input.visible_doc_count)
            .div_ceil(input.allowed_visible_doc_count)
    } else {
        requested_nprobe
            .saturating_mul(input.candidate_top_k.get())
            .div_ceil(input.top_k.get())
    }
    .min(input.populated_list_count) as u32;

    IvfProbeCount::new(widened).expect("candidate nprobe should stay valid")
}
