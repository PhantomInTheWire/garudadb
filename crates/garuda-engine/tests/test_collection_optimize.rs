mod common;

use common::{
    database, default_options, default_schema, dense_vector, doc_id, field_name, seed_collection,
};
use garuda_types::VectorQuery;

#[test]
fn optimize_preserves_query_results() {
    let (_root, db) = database("optimize");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let before = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            4,
        ))
        .expect("query before");
    collection.optimize().expect("optimize");
    let after = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            4,
        ))
        .expect("query after");

    assert_eq!(
        before.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn optimize_keeps_tail_segment_documents_visible() {
    let (_root, db) = database("optimize-tail");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    collection.optimize().expect("optimize");

    let fetched = collection.fetch(vec![
        doc_id("doc-1"),
        doc_id("doc-2"),
        doc_id("doc-3"),
        doc_id("doc-4"),
    ]);

    assert_eq!(fetched.len(), 4);
}
