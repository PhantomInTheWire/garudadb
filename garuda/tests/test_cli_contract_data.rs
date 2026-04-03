#[path = "support/cli_contract.rs"]
mod support;

use support::{run_cli, temp_path};

fn stdout_json(output: &std::process::Output) -> serde_json::Value {
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
        "--vector-projection",
        "include",
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

    let invalid_top_k = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "query",
        "docs",
        "vector",
        "--value",
        "1.0,0.0,0.0,0.0",
        "--top-k",
        "0",
    ]);
    assert!(!invalid_top_k.status.success(), "top-k zero should fail");

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
