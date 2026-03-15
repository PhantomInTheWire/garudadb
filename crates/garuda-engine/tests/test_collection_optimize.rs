mod common;

use common::{database, default_options, default_schema, seed_collection};
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
            "embedding",
            vec![1.0, 0.0, 0.0, 0.0],
            4,
        ))
        .expect("query before");
    collection.optimize().expect("optimize");
    let after = collection
        .query(VectorQuery::by_vector(
            "embedding",
            vec![1.0, 0.0, 0.0, 0.0],
            4,
        ))
        .expect("query after");

    assert_eq!(
        before.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}
