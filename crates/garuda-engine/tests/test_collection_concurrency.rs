mod common;

use common::{build_doc, database, default_options, default_schema, dense_vector, field_name};
use garuda_types::VectorQuery;
use std::thread;

#[test]
fn concurrent_insert_and_query_do_not_panic() {
    let (_root, db) = database("concurrency");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let writer = {
        let collection = collection.clone();
        thread::spawn(move || {
            for i in 0..10 {
                let _ = collection.insert(vec![build_doc(
                    &format!("doc-{i}"),
                    i as i64,
                    "alpha",
                    0.5,
                    [1.0, 0.0, 0.0, 0.0],
                )]);
            }
        })
    };

    let reader = {
        let collection = collection.clone();
        thread::spawn(move || {
            for _ in 0..10 {
                let _ = collection.query(VectorQuery::by_vector(
                    field_name("embedding"),
                    dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
                    common::top_k(5),
                ));
            }
        })
    };

    writer.join().expect("writer thread");
    reader.join().expect("reader thread");
    assert!(collection.stats().doc_count >= 1);
}
