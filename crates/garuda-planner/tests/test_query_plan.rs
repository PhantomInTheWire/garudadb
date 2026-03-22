use garuda_planner::{SegmentSearchPlan, build_query_plan};
use garuda_types::{
    DenseVector, FieldName, FilterExpr, HnswEfSearch, HnswIndexParams, IndexParams, ScalarValue,
    TopK, VectorProjection, VectorQuery,
};

#[test]
fn hnsw_query_plan_should_use_public_ef_search_override() {
    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
        TopK::new(3).expect("valid top_k"),
    );
    query.ef_search = Some(HnswEfSearch::new(8).expect("valid ef_search"));

    let plan = build_query_plan(
        query,
        Some(FilterExpr::Eq(
            "category".to_string(),
            ScalarValue::String("alpha".to_string()),
        )),
        &IndexParams::Hnsw(HnswIndexParams::default()),
    );

    assert_eq!(
        plan.search,
        SegmentSearchPlan::Hnsw {
            ef_search: HnswEfSearch::new(8).expect("valid ef_search"),
        }
    );
    assert_eq!(plan.field_name, field_name("embedding"));
    assert_eq!(plan.top_k, TopK::new(3).expect("valid top_k"));
    assert_eq!(plan.vector_projection, VectorProjection::Exclude);
}

fn field_name(value: &str) -> FieldName {
    FieldName::parse(value).expect("valid field name")
}
