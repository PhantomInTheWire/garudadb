use serde_json::Value;
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

fn stdout_json(output: &std::process::Output) -> Value {
    serde_json::from_slice(&output.stdout).expect("stdout should be valid json")
}

fn write_seed_docs(path: &std::path::Path) {
    std::fs::write(
        path,
        concat!(
            "{\"id\":\"doc-1\",\"fields\":{\"pk\":\"doc-1\",\"rank\":1,\"category\":\"alpha\",\"score\":0.9},\"vector\":[1.0,0.0,0.0,0.0]}\n",
            "{\"id\":\"doc-2\",\"fields\":{\"pk\":\"doc-2\",\"rank\":2,\"category\":\"beta\",\"score\":0.8},\"vector\":[0.0,1.0,0.0,0.0]}\n"
        ),
    )
    .expect("write docs jsonl");
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

#[test]
fn insert_upsert_update_query_fetch_and_delete_commands_should_roundtrip() {
    let tmp = temp_path("cli-data-journey");
    std::fs::create_dir_all(&tmp).expect("create cli temp root");
    let docs_path = tmp.join("docs.jsonl");
    write_seed_docs(&docs_path);

    let update_path = tmp.join("updates.jsonl");
    std::fs::write(
        &update_path,
        "{\"id\":\"doc-1\",\"fields\":{\"pk\":\"doc-1\",\"rank\":11,\"category\":\"alpha\",\"score\":0.9},\"vector\":[1.0,0.0,0.0,0.0]}\n",
    )
    .expect("write updates jsonl");

    let upsert_path = tmp.join("upserts.jsonl");
    std::fs::write(
        &upsert_path,
        "{\"id\":\"doc-3\",\"fields\":{\"pk\":\"doc-3\",\"rank\":3,\"category\":\"gamma\",\"score\":0.7},\"vector\":[0.0,0.0,1.0,0.0]}\n",
    )
    .expect("write upserts jsonl");

    let create = run_cli(&["--root", tmp.to_str().expect("utf8"), "create", "docs", "4"]);
    assert!(create.status.success(), "create should succeed");

    let insert = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "insert-jsonl",
        "docs",
        docs_path.to_str().expect("utf8"),
    ]);
    assert!(insert.status.success(), "insert-jsonl should succeed");
    let insert_json = stdout_json(&insert);
    assert_eq!(insert_json.as_array().expect("insert results").len(), 2);

    let update = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "update-jsonl",
        "docs",
        update_path.to_str().expect("utf8"),
    ]);
    assert!(update.status.success(), "update-jsonl should succeed");

    let upsert = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "upsert-jsonl",
        "docs",
        upsert_path.to_str().expect("utf8"),
    ]);
    assert!(upsert.status.success(), "upsert-jsonl should succeed");

    let query = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "query",
        "docs",
        "vector",
        "--value",
        "1.0,0.0,0.0,0.0",
        "--top-k",
        "3",
        "--filter",
        "rank >= 2",
        "--include-vector",
        "--fields",
        "category,rank",
    ]);
    assert!(query.status.success(), "query should succeed");
    let query_json = stdout_json(&query);
    let hits = query_json.as_array().expect("query hits");
    assert!(!hits.is_empty(), "query should return hits");
    assert!(
        hits[0]["vector"].as_array().is_some(),
        "vector should be included"
    );
    assert!(hits[0]["fields"]["category"]["String"].is_string());
    assert!(hits[0]["fields"]["rank"]["Int64"].is_i64());

    let by_id = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "query",
        "docs",
        "by-id",
        "--value",
        "doc-1",
        "--top-k",
        "2",
    ]);
    assert!(by_id.status.success(), "query by id should succeed");

    let fetch = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "fetch",
        "docs",
        "doc-1",
    ]);
    assert!(fetch.status.success(), "fetch should succeed");
    let fetch_json = stdout_json(&fetch);
    assert_eq!(fetch_json["doc-1"]["fields"]["rank"]["Int64"], 11);

    let delete_filter = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "delete-filter",
        "docs",
        "--filter",
        "category = 'beta'",
    ]);
    assert!(
        delete_filter.status.success(),
        "delete-filter should succeed"
    );

    let delete_ids = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "delete-ids",
        "docs",
        "doc-3",
    ]);
    assert!(delete_ids.status.success(), "delete-ids should succeed");
    let delete_json = stdout_json(&delete_ids);
    assert_eq!(delete_json.as_array().expect("delete results").len(), 1);
}

#[test]
fn create_index_and_drop_index_should_support_advanced_params() {
    let tmp = temp_path("cli-create-index");

    let create = run_cli(&["--root", tmp.to_str().expect("utf8"), "create", "docs", "4"]);
    assert!(create.status.success(), "create should succeed");

    let create_index = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "create-index",
        "docs",
        "embedding",
        "hnsw",
        "--max-neighbors",
        "24",
        "--scaling-factor",
        "32",
        "--ef-construction",
        "128",
        "--prune-width",
        "20",
        "--min-neighbor-count",
        "8",
        "--ef-search",
        "40",
    ]);
    assert!(create_index.status.success(), "create-index should succeed");

    let schema = run_cli(&["--root", tmp.to_str().expect("utf8"), "schema", "docs"]);
    let schema_json = stdout_json(&schema);
    assert_eq!(
        schema_json["vector"]["indexes"]["FlatAndHnsw"]["hnsw"]["max_neighbors"],
        24
    );

    let drop_index = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "drop-index",
        "docs",
        "embedding",
        "hnsw",
    ]);
    assert!(drop_index.status.success(), "drop-index should succeed");
}

#[test]
fn add_rename_and_drop_column_should_roundtrip() {
    let tmp = temp_path("cli-ddl");
    let docs_path = tmp.join("docs.jsonl");
    std::fs::create_dir_all(&tmp).expect("create temp root");
    write_seed_docs(&docs_path);

    let create = run_cli(&["--root", tmp.to_str().expect("utf8"), "create", "docs", "4"]);
    assert!(create.status.success(), "create should succeed");

    let insert = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "insert-jsonl",
        "docs",
        docs_path.to_str().expect("utf8"),
    ]);
    assert!(insert.status.success(), "insert should succeed");

    let add_column = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "add-column",
        "docs",
        "is_public",
        "--type",
        "bool",
        "--nullability",
        "required",
        "--index",
        "none",
        "--default",
        "true",
    ]);
    assert!(add_column.status.success(), "add-column should succeed");

    let rename = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "rename-column",
        "docs",
        "is_public",
        "visible",
    ]);
    assert!(rename.status.success(), "rename-column should succeed");

    let schema = run_cli(&["--root", tmp.to_str().expect("utf8"), "schema", "docs"]);
    let schema_json = stdout_json(&schema);
    let fields = schema_json["fields"].as_array().expect("schema fields");
    assert!(fields.iter().any(|field| field["name"] == "visible"));

    let drop = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "drop-column",
        "docs",
        "visible",
    ]);
    assert!(drop.status.success(), "drop-column should succeed");
}

#[test]
fn create_from_schema_should_accept_schema_and_options_file() {
    let tmp = temp_path("cli-create-from-schema");
    std::fs::create_dir_all(&tmp).expect("create temp root");
    let schema_path = tmp.join("schema.json");

    std::fs::write(
        &schema_path,
        serde_json::json!({
            "schema": {
                "name": "docs",
                "primary_key": "pk",
                "fields": [
                    {
                        "name": "pk",
                        "field_type": "String",
                        "index": "None",
                        "nullability": "Required",
                        "default_value": null
                    },
                    {
                        "name": "rank",
                        "field_type": "Int64",
                        "index": "None",
                        "nullability": "Required",
                        "default_value": null
                    }
                ],
                "vector": {
                    "name": "embedding",
                    "dimension": 4,
                    "metric": "Cosine",
                    "indexes": "DefaultFlat"
                }
            },
            "options": {
                "access_mode": "ReadWrite",
                "storage_access": "MmapPreferred",
                "segment_max_docs": 7
            }
        })
        .to_string(),
    )
    .expect("write schema file");

    let create = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "create-from-schema",
        schema_path.to_str().expect("utf8"),
    ]);
    assert!(create.status.success(), "create-from-schema should succeed");

    let options = run_cli(&["--root", tmp.to_str().expect("utf8"), "options", "docs"]);
    assert!(options.status.success(), "options should succeed");
    let options_json = stdout_json(&options);
    assert_eq!(options_json["segment_max_docs"], 7);
}

#[test]
fn create_from_schema_requires_options_in_file() {
    let tmp = temp_path("cli-invalid-schema-file");
    std::fs::create_dir_all(&tmp).expect("create temp root");
    let schema_path = tmp.join("schema.json");

    std::fs::write(
        &schema_path,
        serde_json::json!({
            "schema": {
                "name": "docs",
                "primary_key": "pk",
                "fields": [],
                "vector": {
                    "name": "embedding",
                    "dimension": 4,
                    "metric": "Cosine",
                    "indexes": "DefaultFlat"
                }
            }
        })
        .to_string(),
    )
    .expect("write invalid schema file");

    let create = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "create-from-schema",
        schema_path.to_str().expect("utf8"),
    ]);
    assert!(!create.status.success(), "missing options should fail");
}
