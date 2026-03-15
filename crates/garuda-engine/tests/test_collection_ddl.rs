mod common;

use common::{database, default_options, default_schema, seed_collection};
use garuda_types::{IndexKind, ScalarFieldSchema, ScalarType, ScalarValue};

#[test]
fn create_drop_index_and_column_ddl_roundtrip() {
    let (_root, db) = database("ddl");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    collection
        .create_index("embedding", IndexKind::Hnsw)
        .expect("create hnsw index");
    assert_eq!(collection.schema().vector.index.kind(), IndexKind::Hnsw);

    collection
        .add_column(ScalarFieldSchema {
            name: "flag".to_string(),
            field_type: ScalarType::Bool,
            nullable: true,
        })
        .expect("add column");
    let fetched = collection.fetch(vec!["doc-1".to_string()]);
    assert_eq!(fetched["doc-1"].fields["flag"], ScalarValue::Null);

    collection
        .alter_column("flag", "is_flagged")
        .expect("rename column");
    let fetched = collection.fetch(vec!["doc-1".to_string()]);
    assert!(fetched["doc-1"].fields.contains_key("is_flagged"));

    collection.drop_column("is_flagged").expect("drop column");
    collection.drop_index("embedding").expect("drop index");
    assert_eq!(collection.schema().vector.index.kind(), IndexKind::Flat);
}
