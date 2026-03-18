mod common;

use common::{collection_name, database, default_options, default_schema};

#[test]
fn create_and_reopen_collection_roundtrips_schema_and_options() {
    let (_root, db) = database("create-open");
    let schema = default_schema("docs");
    let options = default_options();
    let collection = db
        .create_collection(schema.clone(), options.clone())
        .expect("create collection");

    assert_eq!(collection.schema(), schema);
    assert_eq!(collection.options(), options);
    assert_eq!(collection.stats().doc_count, 0);

    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    assert_eq!(reopened.schema(), schema);
    assert_eq!(reopened.options(), options);
    assert_eq!(reopened.stats().doc_count, 0);
}
