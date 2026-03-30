use super::*;

fn top_k(value: usize) -> TopK {
    TopK::new(value).expect("top_k")
}

fn nprobe(value: u32) -> IvfProbeCount {
    IvfProbeCount::new(value).expect("nprobe")
}

#[test]
fn search_candidate_nprobe_should_keep_requested_nprobe_without_widening() {
    assert_eq!(
        search_candidate_nprobe(nprobe(2), top_k(4), top_k(4), 8),
        nprobe(2)
    );
}

#[test]
fn search_candidate_nprobe_should_widen_proportionally_instead_of_scanning_all_lists() {
    assert_eq!(
        search_candidate_nprobe(nprobe(1), top_k(1), top_k(8), 16),
        nprobe(8)
    );
}

#[test]
fn search_candidate_nprobe_should_scan_all_lists_for_small_ivf_segments() {
    assert_eq!(
        search_candidate_nprobe(nprobe(1), top_k(1), top_k(3), 3),
        nprobe(3)
    );
}
