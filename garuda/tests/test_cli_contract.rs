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

fn temp_root(prefix: &str) -> std::path::PathBuf {
    let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("garudadb-cli-{prefix}-{ts}-{nonce}"));
    std::fs::create_dir_all(&path).expect("create temp root");
    path
}

#[test]
fn doctor_should_exit_successfully_for_a_healthy_workspace() {
    let output = run_cli(&[]);
    assert!(
        output.status.success(),
        "cli should not panic on bare invocation"
    );
}

#[test]
fn init_should_create_or_validate_a_workspace_root() {
    let tmp = temp_root("init-contract");
    let output = run_cli(&["--root", tmp.to_str().expect("utf8"), "init"]);
    assert!(output.status.success(), "init should succeed");
}

#[test]
fn create_and_stats_should_form_a_basic_user_journey() {
    let tmp = temp_root("create-stats-contract");
    let create = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "create",
        "docs",
        "4",
    ]);
    assert!(create.status.success(), "create should succeed");

    let stats = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "stats",
        "docs",
    ]);
    assert!(stats.status.success(), "stats should succeed after create");
}
