#[path = "support/cli_contract.rs"]
mod support;

use support::{run_cli, temp_path};

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

fn assert_partial_write_fails_with_json(
    root: &std::path::Path,
    command: &str,
    docs_path: &std::path::Path,
    expected_error: &str,
) {
    let output = run_cli(&[
        "--root",
        root.to_str().expect("utf8"),
        command,
        "docs",
        docs_path.to_str().expect("utf8"),
    ]);
    assert!(!output.status.success(), "partial write should fail");

    let results = serde_json::from_slice::<serde_json::Value>(&output.stdout)
        .expect("stdout should be valid json")
        .as_array()
        .expect("write results")
        .clone();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["status"]["code"], "Ok");
    assert_eq!(results[1]["status"]["code"], expected_error);
}

#[test]
fn insert_jsonl_requires_fields_in_each_document() {
    let tmp = temp_path("cli-missing-fields");
    std::fs::create_dir_all(&tmp).expect("create temp root");
    let docs_path = tmp.join("docs.jsonl");

    std::fs::write(
        &docs_path,
        "{\"id\":\"doc-1\",\"vector\":[1.0,0.0,0.0,0.0]}\n",
    )
    .expect("write invalid docs jsonl");

    let create = run_cli(&["--root", tmp.to_str().expect("utf8"), "create", "docs", "4"]);
    assert!(create.status.success(), "create should succeed");

    let insert = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "insert-jsonl",
        "docs",
        docs_path.to_str().expect("utf8"),
    ]);
    assert!(!insert.status.success(), "missing fields should fail");
}

#[test]
fn insert_jsonl_rejects_unsigned_integer_literals_that_exceed_i64() {
    let tmp = temp_path("cli-large-unsigned-int");
    std::fs::create_dir_all(&tmp).expect("create temp root");
    let docs_path = tmp.join("docs.jsonl");

    std::fs::write(
        &docs_path,
        "{\"id\":\"doc-1\",\"fields\":{\"pk\":\"doc-1\",\"rank\":18446744073709551615,\"category\":\"alpha\",\"score\":0.9},\"vector\":[1.0,0.0,0.0,0.0]}\n",
    )
    .expect("write invalid docs jsonl");

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
        !insert.status.success(),
        "out-of-range integer literal should fail"
    );
}

#[test]
fn insert_jsonl_should_return_non_zero_when_any_write_fails() {
    let tmp = temp_path("cli-partial-write-failure");
    std::fs::create_dir_all(&tmp).expect("create temp root");
    let docs_path = tmp.join("docs.jsonl");

    std::fs::write(
        &docs_path,
        concat!(
            "{\"id\":\"doc-1\",\"fields\":{\"pk\":\"doc-1\",\"rank\":1,\"category\":\"alpha\",\"score\":0.9},\"vector\":[1.0,0.0,0.0,0.0]}\n",
            "{\"id\":\"doc-1\",\"fields\":{\"pk\":\"doc-1\",\"rank\":2,\"category\":\"beta\",\"score\":0.8},\"vector\":[0.0,1.0,0.0,0.0]}\n"
        ),
    )
    .expect("write duplicate docs jsonl");

    let create = run_cli(&["--root", tmp.to_str().expect("utf8"), "create", "docs", "4"]);
    assert!(create.status.success(), "create should succeed");

    assert_partial_write_fails_with_json(&tmp, "insert-jsonl", &docs_path, "AlreadyExists");
}

#[test]
fn upsert_jsonl_should_return_non_zero_when_any_write_fails() {
    let tmp = temp_path("cli-partial-upsert-failure");
    std::fs::create_dir_all(&tmp).expect("create temp root");
    let docs_path = tmp.join("docs.jsonl");

    std::fs::write(
        &docs_path,
        concat!(
            "{\"id\":\"doc-1\",\"fields\":{\"pk\":\"doc-1\",\"rank\":1,\"category\":\"alpha\",\"score\":0.9},\"vector\":[1.0,0.0,0.0,0.0]}\n",
            "{\"id\":\"doc-2\",\"fields\":{\"pk\":\"doc-2\",\"rank\":2,\"category\":\"beta\",\"score\":0.8},\"vector\":[0.0,1.0,0.0]}\n"
        ),
    )
    .expect("write partial upsert docs jsonl");

    let create = run_cli(&["--root", tmp.to_str().expect("utf8"), "create", "docs", "4"]);
    assert!(create.status.success(), "create should succeed");

    assert_partial_write_fails_with_json(&tmp, "upsert-jsonl", &docs_path, "InvalidArgument");
}

#[test]
fn update_jsonl_should_return_non_zero_when_any_write_fails() {
    let tmp = temp_path("cli-partial-update-failure");
    std::fs::create_dir_all(&tmp).expect("create temp root");
    let seed_docs_path = tmp.join("seed-docs.jsonl");
    let docs_path = tmp.join("docs.jsonl");

    write_seed_docs(&seed_docs_path);
    std::fs::write(
        &docs_path,
        concat!(
            "{\"id\":\"doc-1\",\"fields\":{\"pk\":\"doc-1\",\"rank\":11,\"category\":\"alpha\",\"score\":0.9},\"vector\":[1.0,0.0,0.0,0.0]}\n",
            "{\"id\":\"missing-doc\",\"fields\":{\"pk\":\"missing-doc\",\"rank\":12,\"category\":\"beta\",\"score\":0.8},\"vector\":[0.0,1.0,0.0,0.0]}\n"
        ),
    )
    .expect("write invalid update docs jsonl");

    let create = run_cli(&["--root", tmp.to_str().expect("utf8"), "create", "docs", "4"]);
    assert!(create.status.success(), "create should succeed");

    let insert = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "insert-jsonl",
        "docs",
        seed_docs_path.to_str().expect("utf8"),
    ]);
    assert!(insert.status.success(), "seed insert should succeed");

    assert_partial_write_fails_with_json(&tmp, "update-jsonl", &docs_path, "NotFound");
}
