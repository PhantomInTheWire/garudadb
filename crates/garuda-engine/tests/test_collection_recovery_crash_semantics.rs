mod common;

use common::{build_doc, collection_name, database, default_options, default_schema, doc_id};
use garuda_segment::{WalOp, append_wal_ops};
use garuda_storage::WRITING_SEGMENT_ID;
use garuda_types::{CollectionOptions, ScalarValue};
use std::fs;

const FNV_OFFSET_BASIS: u32 = 2_166_136_261;
const FNV_PRIME: u32 = 16_777_619;

#[test]
fn reopen_after_unflushed_writes_should_follow_a_clear_recovery_policy() {
    let (_root, db) = database("recovery-unflushed");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    let inserted = collection.insert(vec![build_doc(
        "doc-1",
        1,
        "alpha",
        0.9,
        [1.0, 0.0, 0.0, 0.0],
    )]);
    assert!(inserted[0].status.is_ok());
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen after wal-backed write");
    let fetched = reopened.fetch(vec![doc_id("doc-1")]);
    assert!(fetched.contains_key(&doc_id("doc-1")));
}

#[test]
fn reopen_after_unflushed_upsert_should_preserve_latest_visible_document() {
    let (_root, db) = database("recovery-unflushed-upsert");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let inserted = collection.insert(vec![build_doc(
        "doc-1",
        1,
        "alpha",
        0.9,
        [1.0, 0.0, 0.0, 0.0],
    )]);
    assert!(inserted[0].status.is_ok());

    let upserted = collection.upsert(vec![build_doc(
        "doc-1",
        9,
        "beta",
        0.2,
        [0.0, 1.0, 0.0, 0.0],
    )]);
    assert!(upserted[0].status.is_ok());
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen after wal-backed upsert");
    let fetched = reopened.fetch(vec![doc_id("doc-1")]);
    let doc = fetched.get(&doc_id("doc-1")).expect("doc present");

    assert_eq!(doc.fields["rank"], ScalarValue::Int64(9));
    assert_eq!(
        doc.fields["category"],
        ScalarValue::String("beta".to_string())
    );
}

#[test]
fn reopen_after_delete_then_flush_should_not_resurrect_tombstoned_docs() {
    let (_root, db) = database("recovery-delete");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    common::seed_collection(&collection);
    let deleted = collection.delete(vec![doc_id("doc-2")]);
    assert!(deleted[0].status.is_ok());
    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen");
    assert!(reopened.fetch(vec![doc_id("doc-2")]).is_empty());
}

#[test]
fn stale_wal_after_flush_does_not_duplicate_checkpointed_writes() {
    let (_root, db) = database("recovery-stale-wal");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let doc = build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]);
    let inserted = collection.insert(vec![doc.clone()]);
    assert!(inserted[0].status.is_ok());
    collection.flush().expect("flush");

    append_wal_ops(
        &collection.path(),
        WRITING_SEGMENT_ID,
        &[WalOp::Insert(doc)],
    )
    .expect("append stale wal op");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen after stale wal");
    let fetched = reopened.fetch(vec![doc_id("doc-1")]);

    assert_eq!(reopened.stats().doc_count, 1);
    assert_eq!(fetched.len(), 1);
}

#[test]
fn stale_insert_wal_after_flush_does_not_break_reopen_when_doc_was_updated() {
    let (_root, db) = database("recovery-stale-insert-after-update");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let original = build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]);
    let mut updated = original.clone();
    updated.fields.insert(
        "category".to_string(),
        ScalarValue::String("beta".to_string()),
    );

    let inserted = collection.insert(vec![original.clone()]);
    assert!(inserted[0].status.is_ok());

    let updated_result = collection.update(vec![updated.clone()]);
    assert!(updated_result[0].status.is_ok());

    collection.flush().expect("flush");
    append_wal_ops(
        &collection.path(),
        WRITING_SEGMENT_ID,
        &[WalOp::Insert(original)],
    )
    .expect("append stale insert");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen after stale insert");
    let fetched = reopened.fetch(vec![doc_id("doc-1")]);
    let doc = fetched.get(&doc_id("doc-1")).expect("doc present");

    assert_eq!(
        doc.fields.get("category"),
        Some(&ScalarValue::String("beta".to_string()))
    );
}

#[test]
fn stale_update_wal_after_flush_does_not_break_reopen_when_doc_was_deleted() {
    let (_root, db) = database("recovery-stale-update-after-delete");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let original = build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]);
    let mut updated = original.clone();
    updated.fields.insert(
        "category".to_string(),
        ScalarValue::String("beta".to_string()),
    );

    assert!(
        collection
            .insert(vec![original])
            .first()
            .expect("insert result")
            .status
            .is_ok()
    );
    assert!(
        collection
            .update(vec![updated.clone()])
            .first()
            .expect("update result")
            .status
            .is_ok()
    );
    assert!(
        collection
            .delete(vec![doc_id("doc-1")])
            .first()
            .expect("delete result")
            .status
            .is_ok()
    );

    collection.flush().expect("flush");
    append_wal_ops(
        &collection.path(),
        WRITING_SEGMENT_ID,
        &[WalOp::Update(updated)],
    )
    .expect("append stale update");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen after stale update");

    assert!(reopened.fetch(vec![doc_id("doc-1")]).is_empty());
}

#[test]
fn stale_wal_after_flush_with_rolled_segments_keeps_flushed_docs_visible() {
    let (_root, db) = database("recovery-stale-wal-rolled-segment");
    let collection = db
        .create_collection(default_schema("docs"), options_with_segment_max_docs(1))
        .expect("create collection");

    let first = build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]);
    let second = build_doc("doc-2", 2, "beta", 0.8, [0.0, 1.0, 0.0, 0.0]);

    assert!(collection.insert(vec![first])[0].status.is_ok());
    assert!(collection.insert(vec![second.clone()])[0].status.is_ok());

    collection.flush().expect("flush");
    append_wal_ops(
        &collection.path(),
        WRITING_SEGMENT_ID,
        &[WalOp::Insert(second)],
    )
    .expect("append stale wal");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen after stale wal");

    assert_eq!(reopened.stats().doc_count, 2);
    assert!(
        reopened
            .fetch(vec![doc_id("doc-1")])
            .contains_key(&doc_id("doc-1"))
    );
    assert!(
        reopened
            .fetch(vec![doc_id("doc-2")])
            .contains_key(&doc_id("doc-2"))
    );
}

#[test]
fn truncated_wal_entry_header_returns_an_error_instead_of_panicking() {
    let (_root, db) = database("recovery-truncated-wal-header");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    collection.flush().expect("flush");

    let wal_path = collection.path().join("0").join("data.wal");
    let mut wal_bytes = fs::read(&wal_path).expect("read wal");
    wal_bytes.truncate(wal_bytes.len() - std::mem::size_of::<u32>());
    wal_bytes.push(0);
    let checksum = wal_checksum(&wal_bytes);
    wal_bytes.extend_from_slice(&checksum.to_le_bytes());
    fs::write(&wal_path, wal_bytes).expect("write truncated wal");
    drop(collection);

    let reopened = db.open_collection(&collection_name("docs"));
    assert!(reopened.is_err());
}

#[test]
fn oversized_wal_payload_length_returns_an_error_instead_of_panicking() {
    let (_root, db) = database("recovery-oversized-wal-payload");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    collection.flush().expect("flush");

    let wal_path = collection.path().join("0").join("data.wal");
    let mut wal_bytes = b"GRDWAL01".to_vec();
    wal_bytes.extend_from_slice(&1u16.to_le_bytes());
    wal_bytes.push(0);
    wal_bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    let checksum = wal_checksum(&wal_bytes);
    wal_bytes.extend_from_slice(&checksum.to_le_bytes());
    fs::write(&wal_path, wal_bytes).expect("write oversized wal");
    drop(collection);

    let reopened = db.open_collection(&collection_name("docs"));
    assert!(reopened.is_err());
}

fn wal_checksum(bytes: &[u8]) -> u32 {
    let mut hash = FNV_OFFSET_BASIS;

    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

fn options_with_segment_max_docs(segment_max_docs: usize) -> CollectionOptions {
    CollectionOptions {
        segment_max_docs,
        ..common::default_options()
    }
}
