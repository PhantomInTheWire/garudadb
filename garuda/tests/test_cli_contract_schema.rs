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
fn query_should_use_collection_vector_field_from_schema() {
    let tmp = temp_path("cli-custom-vector-field");
    std::fs::create_dir_all(&tmp).expect("create temp root");
    let schema_path = tmp.join("schema.json");
    let docs_path = tmp.join("docs.jsonl");

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
                    },
                    {
                        "name": "category",
                        "field_type": "String",
                        "index": "None",
                        "nullability": "Required",
                        "default_value": null
                    },
                    {
                        "name": "score",
                        "field_type": "Float64",
                        "index": "None",
                        "nullability": "Required",
                        "default_value": null
                    }
                ],
                "vector": {
                    "name": "feature",
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

    write_seed_docs(&docs_path);

    let create = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "create-from-schema",
        schema_path.to_str().expect("utf8"),
    ]);
    assert!(create.status.success(), "create-from-schema should succeed");

    let insert = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "insert-jsonl",
        "docs",
        docs_path.to_str().expect("utf8"),
    ]);
    assert!(insert.status.success(), "insert should succeed");

    let query = run_cli(&[
        "--root",
        tmp.to_str().expect("utf8"),
        "query",
        "docs",
        "vector",
        "--value",
        "1.0,0.0,0.0,0.0",
        "--top-k",
        "2",
    ]);
    assert!(query.status.success(), "vector query should succeed");

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
    assert!(by_id.status.success(), "by-id query should succeed");
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
