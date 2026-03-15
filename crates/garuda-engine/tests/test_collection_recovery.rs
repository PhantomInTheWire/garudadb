mod common;

use common::{database, default_options, default_schema, seed_collection};
use garuda_types::VectorQuery;

#[test]
fn reopen_after_flush_preserves_manifest_and_query_results() {
    let (_root, db) = database("recovery");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");

    let before = collection
        .query(VectorQuery::by_vector(
            "embedding",
            vec![1.0, 0.0, 0.0, 0.0],
            3,
        ))
        .expect("query before");
    let reopened = db.open_collection("docs").expect("reopen");
    let after = reopened
        .query(VectorQuery::by_vector(
            "embedding",
            vec![1.0, 0.0, 0.0, 0.0],
            3,
        ))
        .expect("query after");

    assert_eq!(
        before.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}
