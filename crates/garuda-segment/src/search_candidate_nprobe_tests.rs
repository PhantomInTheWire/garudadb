use crate::search::{CandidateNprobeInput, search_candidate_nprobe};
use garuda_types::{AnnBudgetPolicy, IvfProbeCount, TopK};

fn top_k(value: usize) -> TopK {
    TopK::new(value).expect("top_k")
}

fn nprobe(value: u32) -> IvfProbeCount {
    IvfProbeCount::new(value).expect("nprobe")
}

#[test]
fn search_candidate_nprobe_should_keep_requested_nprobe_without_widening() {
    assert_eq!(
        search_candidate_nprobe(
            CandidateNprobeInput {
                nprobe: nprobe(2),
                top_k: top_k(4),
                budget: AnnBudgetPolicy::Requested,
                candidate_top_k: top_k(4),
                candidate_doc_count: 8,
                visible_doc_count: 8,
                allowed_visible_doc_count: 8,
                populated_list_count: 8,
            },
        ),
        nprobe(2)
    );
}

#[test]
fn search_candidate_nprobe_should_widen_proportionally_instead_of_scanning_all_lists() {
    assert_eq!(
        search_candidate_nprobe(
            CandidateNprobeInput {
                nprobe: nprobe(1),
                top_k: top_k(1),
                budget: AnnBudgetPolicy::AdaptiveFiltered,
                candidate_top_k: top_k(8),
                candidate_doc_count: 16,
                visible_doc_count: 16,
                allowed_visible_doc_count: 2,
                populated_list_count: 16,
            },
        ),
        nprobe(8)
    );
}

#[test]
fn search_candidate_nprobe_should_scan_all_lists_for_small_ivf_segments() {
    assert_eq!(
        search_candidate_nprobe(
            CandidateNprobeInput {
                nprobe: nprobe(1),
                top_k: top_k(1),
                budget: AnnBudgetPolicy::AdaptiveFiltered,
                candidate_top_k: top_k(3),
                candidate_doc_count: 3,
                visible_doc_count: 3,
                allowed_visible_doc_count: 1,
                populated_list_count: 3,
            },
        ),
        nprobe(3)
    );
}

#[test]
fn search_candidate_nprobe_should_widen_for_delete_churn_even_without_user_filter() {
    assert_eq!(
        search_candidate_nprobe(
            CandidateNprobeInput {
                nprobe: nprobe(1),
                top_k: top_k(5),
                budget: AnnBudgetPolicy::Requested,
                candidate_top_k: top_k(5),
                candidate_doc_count: 50,
                visible_doc_count: 10,
                allowed_visible_doc_count: 10,
                populated_list_count: 8,
            },
        ),
        nprobe(8)
    );
}
