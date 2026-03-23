use super::*;
use garuda_types::{
    CollectionName, CollectionSchema, DenseVector, DistanceMetric, Doc, DocId, FieldName,
    IvfIndexParams, Nullability, ScalarFieldSchema, ScalarIndexState, ScalarType, ScalarValue,
    VectorDimension, VectorFieldSchema, VectorIndexState,
};
use std::collections::BTreeMap;

#[test]
fn appending_ivf_records_should_add_each_record_once() {
    let schema = schema();
    let mut manager =
        SegmentManager::new(Vec::new(), SegmentManager::empty_writing_segment(&schema));
    let mut next_segment_id = SegmentId::new_unchecked(1);

    manager.append_new_record(
        InternalDocId::new(1).expect("valid internal doc id"),
        doc("doc-1", [1.0, 0.0, 0.0, 0.0]),
        &mut next_segment_id,
        100,
        &schema,
    );
    manager.append_new_record(
        InternalDocId::new(2).expect("valid internal doc id"),
        doc("doc-2", [0.9, 0.1, 0.0, 0.0]),
        &mut next_segment_id,
        100,
        &schema,
    );
    manager.append_new_record(
        InternalDocId::new(3).expect("valid internal doc id"),
        doc("doc-3", [0.0, 1.0, 0.0, 0.0]),
        &mut next_segment_id,
        100,
        &schema,
    );
    manager.append_new_record(
        InternalDocId::new(4).expect("valid internal doc id"),
        doc("doc-4", [0.0, 0.0, 1.0, 0.0]),
        &mut next_segment_id,
        100,
        &schema,
    );

    let index = manager
        .writing_segment()
        .ivf_index
        .as_ref()
        .expect("ivf index");
    assert_eq!(index.len(), 4);
    assert_eq!(index.list_count(), 4);
}

fn schema() -> CollectionSchema {
    CollectionSchema {
        name: CollectionName::parse("docs").expect("valid name"),
        primary_key: field_name("pk"),
        fields: vec![ScalarFieldSchema {
            name: field_name("pk"),
            field_type: ScalarType::String,
            index: ScalarIndexState::None,
            nullability: Nullability::Required,
            default_value: None,
        }],
        vector: VectorFieldSchema {
            name: field_name("embedding"),
            dimension: VectorDimension::new(4).expect("valid dimension"),
            metric: DistanceMetric::Cosine,
            indexes: VectorIndexState::IvfOnly(IvfIndexParams::default()),
        },
    }
}

fn field_name(value: &str) -> FieldName {
    FieldName::parse(value).expect("valid field name")
}

fn doc(id: &str, vector: [f32; 4]) -> Doc {
    Doc::new(
        DocId::parse(id).expect("valid doc id"),
        BTreeMap::from([("pk".to_string(), ScalarValue::String(id.to_string()))]),
        DenseVector::parse(vector.to_vec()).expect("valid vector"),
    )
}
```

Now let me update `segment_manager.rs` to replace the inline test module with the file reference, and then commit and push.
