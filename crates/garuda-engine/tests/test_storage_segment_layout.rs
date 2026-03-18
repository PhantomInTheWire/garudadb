mod common;

use common::{build_doc, collection_dir, database, default_options, default_schema};

#[test]
fn rollover_creates_persisted_segment_directory_and_keeps_writing_segment_zero() {
    let (root, db) = database("storage-segment-layout");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let results = collection.insert(vec![
        build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]),
        build_doc("doc-2", 2, "alpha", 0.8, [0.9, 0.1, 0.0, 0.0]),
    ]);
    assert!(results.iter().all(|result| result.status.is_ok()));
    collection.flush().expect("flush");
    drop(collection);

    let collection_dir = collection_dir(&root, "docs");

    assert!(collection_dir.join("0").exists());
    assert!(collection_dir.join("0").join("data.seg").exists());
    assert!(collection_dir.join("0").join("data.wal").exists());
    assert!(collection_dir.join("1").exists());
    assert!(collection_dir.join("1").join("data.seg").exists());
}

#[test]
fn reopen_uses_manifest_metadata_for_persisted_segments() {
    let (root, db) = database("storage-segment-manifest");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let results = collection.insert(vec![
        build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]),
        build_doc("doc-2", 2, "alpha", 0.8, [0.9, 0.1, 0.0, 0.0]),
    ]);
    assert!(results.iter().all(|result| result.status.is_ok()));
    collection.flush().expect("flush");
    drop(collection);

    let persisted_segment_dir = collection_dir(&root, "docs").join("1");
    std::fs::rename(
        &persisted_segment_dir,
        collection_dir(&root, "docs").join("999"),
    )
    .expect("rename persisted segment dir away from manifest path");

    let reopened = db.open_collection(&common::collection_name("docs"));
    assert!(
        reopened.is_err(),
        "reopen should trust manifest-referenced segment paths instead of guessing directories"
    );
}
