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

#[test]
fn insert_jsonl_query_and_fetch_should_form_a_basic_cli_data_journey() {
    let tmp = temp_root("cli-data-journey");
    let docs_path = tmp.join("docs.jsonl");

    std::fs::write(
        &docs_path,
        concat!(
            "{\"id\":\"doc-1\",\"fields\":{\"pk\":\"doc-1\",\"rank\":1,\"category\":\"alpha\",\"score\":0.9},\"vector\":[1.0,0.0,0.0,0.0]}\n",
            "{\"id\":\"doc-2\",\"fields\":{\"pk\":\"doc-2\",\"rank\":2,\"category\":\"beta\",\"score\":0.8},\"vector\":[0.0,1.0,0.0,0.0]}\n"
        ),
    )
    .expect("write docs jsonl");

    let create = run_cli(&["--root", tmp.to_str().expect("utf8"), "create", "docs", "4"]);
    assert!(create.status.success(), "create should succeed");

    let insert = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "insert-jsonl",
        "docs",
        docs_path.to_str().expect("utf8"),
    ]);
    assert!(
        insert.status.success(),
        "insert-jsonl should succeed once the planned cli surface exists"
    );

    let query = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "query",
        "docs",
        "--vector",
        "1.0,0.0,0.0,0.0",
        "--top-k",
        "2",
    ]);
    assert!(
        query.status.success(),
        "query should succeed once the planned cli surface exists"
    );

    let fetch = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "fetch",
        "docs",
        "doc-1",
    ]);
    assert!(
        fetch.status.success(),
        "fetch should succeed once the planned cli surface exists"
    );
}

#[test]
fn create_index_command_should_exist_for_cli_contract() {
    let tmp = temp_root("cli-create-index");

    let create = run_cli(&["--root", tmp.to_str().expect("utf8"), "create", "docs", "4"]);
    assert!(create.status.success(), "create should succeed");

    let create_index = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "create-index",
        "docs",
        "embedding",
        "hnsw",
    ]);
    assert!(
        create_index.status.success(),
        "create-index should succeed once the planned cli surface exists"
    );
}
