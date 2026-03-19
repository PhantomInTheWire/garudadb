use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_garuda-cli"))
        .args(args)
        .output()
        .expect("run cli")
}

fn temp_path(prefix: &str) -> std::path::PathBuf {
    let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    std::env::temp_dir().join(format!("garudadb-cli-{prefix}-{ts}-{nonce}"))
}

#[test]
fn init_should_create_or_validate_a_workspace_root() {
    let tmp = temp_path("init-contract");
    assert!(!tmp.exists(), "test requires a missing root directory");

    let output = run_cli(&["--root", tmp.to_str().expect("utf8"), "init"]);

    assert!(output.status.success(), "init should succeed");
    assert!(tmp.is_dir(), "init should create the root directory");
}

#[test]
fn create_and_stats_should_form_a_basic_user_journey() {
    let tmp = temp_path("create-stats-contract");
    let create = run_cli(&["--root", tmp.to_str().expect("utf8"), "create", "docs", "4"]);
    assert!(create.status.success(), "create should succeed");

    let stats = run_cli(&["--root", tmp.to_str().expect("utf8"), "stats", "docs"]);
    assert!(stats.status.success(), "stats should succeed after create");
}
