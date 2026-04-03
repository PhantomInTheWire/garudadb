#[path = "support/cli_contract.rs"]
mod support;

use support::{run_cli, temp_path};

fn stdout_json(output: &std::process::Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).expect("stdout should be valid json")
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
fn create_stats_schema_options_flush_and_optimize_should_work() {
    let tmp = temp_path("ops-contract");

    let create = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "create",
        "docs",
        "4",
        "--metric",
        "l2",
        "--segment-max-docs",
        "16",
        "--storage-access",
        "standard-io",
    ]);
    assert!(create.status.success(), "create should succeed");

    let stats = run_cli(&["--root", tmp.to_str().expect("utf8"), "stats", "docs"]);
    assert!(stats.status.success(), "stats should succeed after create");
    let stats_json = stdout_json(&stats);
    assert_eq!(stats_json["doc_count"], 0);

    let schema = run_cli(&["--root", tmp.to_str().expect("utf8"), "schema", "docs"]);
    assert!(schema.status.success(), "schema should succeed");
    let schema_json = stdout_json(&schema);
    assert_eq!(schema_json["vector"]["metric"], "L2");

    let options = run_cli(&["--root", tmp.to_str().expect("utf8"), "options", "docs"]);
    assert!(options.status.success(), "options should succeed");
    let options_json = stdout_json(&options);
    assert_eq!(options_json["segment_max_docs"], 16);
    assert_eq!(options_json["storage_access"], "StandardIo");

    let flush = run_cli(&["--root", tmp.to_str().expect("utf8"), "flush", "docs"]);
    assert!(flush.status.success(), "flush should succeed");

    let optimize = run_cli(&["--root", tmp.to_str().expect("utf8"), "optimize", "docs"]);
    assert!(optimize.status.success(), "optimize should succeed");
}
