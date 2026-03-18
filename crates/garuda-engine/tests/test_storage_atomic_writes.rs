mod common;

use common::{collection_name, database, default_options, default_schema};

#[test]
fn atomic_write_keeps_only_final_file() {
    let (root, db) = database("storage-atomic-write");
    let schema = default_schema("docs");

    let collection = db
        .create_collection(schema.clone(), default_options())
        .expect("create collection");

    collection.flush().expect("flush");

    let version_path = root.join("docs").join("VERSION.json");
    let temp_path = root.join("docs").join("VERSION.json.tmp");
    assert!(version_path.exists());
    assert!(!temp_path.exists());

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    assert_eq!(reopened.schema(), schema);
}
