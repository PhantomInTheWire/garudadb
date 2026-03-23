use garuda_planner::{SegmentFilterPlan, SegmentSearchPlan, build_query_plan};
use garuda_types::{
    CollectionName, CollectionSchema, DenseVector, DistanceMetric, FieldName, FilterExpr,
    FlatHnswDefault, FlatIvfDefault, HnswEfSearch, HnswIndexParams, IvfIndexParams, IvfProbeCount,
    LikePattern, Nullability, ScalarCompareOp, ScalarFieldSchema, ScalarIndexState,
    ScalarPredicate, ScalarPrefilter, ScalarType, ScalarValue, StringMatchExpr, TopK,
    VectorDimension, VectorFieldSchema, VectorIndexState, VectorProjection, VectorQuery,
    VectorSearch,
};

#[test]
fn hnsw_query_plan_should_use_public_ef_search_override() {
    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
        TopK::new(3).expect("valid top_k"),
    );
    query.search = VectorSearch::Hnsw {
        ef_search: HnswEfSearch::new(8).expect("valid ef_search"),
    };

    let plan = build_query_plan(
        query,
        Some(FilterExpr::Eq(
            "category".to_string(),
            ScalarValue::String("alpha".to_string()),
        )),
        &schema(VectorIndexState::FlatAndHnsw {
            default: FlatHnswDefault::Hnsw,
            hnsw: HnswIndexParams::default(),
        }),
    )
    .expect("build query plan");

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

#[test]
fn indexed_and_filter_should_split_prefilter_and_residual() {
    let query = VectorQuery::by_vector(
        field_name("embedding"),
        DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
        TopK::new(3).expect("valid top_k"),
    );
    let filter = FilterExpr::And(
        Box::new(FilterExpr::Eq(
            "category".to_string(),
            ScalarValue::String("alpha".to_string()),
        )),
        Box::new(FilterExpr::Ne("rank".to_string(), ScalarValue::Int64(2))),
    );

    let plan = build_query_plan(query, Some(filter.clone()), &indexed_schema())
        .expect("build filter plan");

    assert_eq!(
        plan.filter.scalar_prefilter,
        ScalarPrefilter::And(vec![ScalarPredicate {
            field: field_name("category"),
            op: ScalarCompareOp::Eq,
            value: ScalarValue::String("alpha".to_string()),
        }])
    );
    assert_eq!(
        plan.filter.residual,
        SegmentFilterPlan::Matching(FilterExpr::Ne("rank".to_string(), ScalarValue::Int64(2),))
    );
}

#[test]
fn or_filter_should_stay_residual() {
    let query = VectorQuery::by_vector(
        field_name("embedding"),
        DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
        TopK::new(3).expect("valid top_k"),
    );
    let filter = FilterExpr::Or(
        Box::new(FilterExpr::Eq(
            "category".to_string(),
            ScalarValue::String("alpha".to_string()),
        )),
        Box::new(FilterExpr::Eq("rank".to_string(), ScalarValue::Int64(2))),
    );

    let plan =
        build_query_plan(query, Some(filter.clone()), &indexed_schema()).expect("build or plan");

    assert_eq!(plan.filter.scalar_prefilter, ScalarPrefilter::All);
    assert_eq!(plan.filter.residual, SegmentFilterPlan::Matching(filter));
}

#[test]
fn like_contains_and_is_null_stay_residual() {
    let query = VectorQuery::by_vector(
        field_name("embedding"),
        DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
        TopK::new(3).expect("valid top_k"),
    );
    let filter = FilterExpr::And(
        Box::new(FilterExpr::StringMatch(
            "category".to_string(),
            StringMatchExpr::Like(LikePattern::PrefixSuffix {
                prefix: "alp".to_string(),
                suffix: String::new(),
            }),
        )),
        Box::new(FilterExpr::And(
            Box::new(FilterExpr::StringMatch(
                "category".to_string(),
                StringMatchExpr::Contains("pha".to_string()),
            )),
            Box::new(FilterExpr::IsNull("nickname".to_string())),
        )),
    );

    let plan =
        build_query_plan(query, Some(filter), &indexed_schema()).expect("build residual plan");

    assert_eq!(plan.filter.scalar_prefilter, ScalarPrefilter::All);
    assert_eq!(
        plan.filter.residual,
        SegmentFilterPlan::Matching(FilterExpr::And(
            Box::new(FilterExpr::And(
                Box::new(FilterExpr::StringMatch(
                    "category".to_string(),
                    StringMatchExpr::Like(LikePattern::PrefixSuffix {
                        prefix: "alp".to_string(),
                        suffix: String::new(),
                    }),
                )),
                Box::new(FilterExpr::StringMatch(
                    "category".to_string(),
                    StringMatchExpr::Contains("pha".to_string()),
                )),
            )),
            Box::new(FilterExpr::IsNull("nickname".to_string())),
        ))
    );
}

#[test]
fn ivf_query_plan_should_use_public_nprobe_override() {
    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
        TopK::new(3).expect("valid top_k"),
    );
    query.search = VectorSearch::Ivf {
        nprobe: IvfProbeCount::new(5).expect("valid nprobe"),
    };

    let plan = build_query_plan(
        query,
        None,
        &schema(VectorIndexState::FlatAndIvf {
            default: FlatIvfDefault::Ivf,
            ivf: IvfIndexParams::default(),
        }),
    )
    .expect("build ivf plan");

    assert_eq!(
        plan.search,
        SegmentSearchPlan::Ivf {
            nprobe: IvfProbeCount::new(5).expect("valid nprobe"),
        }
    );
}

fn indexed_schema() -> CollectionSchema {
    CollectionSchema {
        name: CollectionName::parse("docs").expect("valid name"),
        primary_key: field_name("pk"),
        fields: vec![
            ScalarFieldSchema {
                name: field_name("pk"),
                field_type: ScalarType::String,
                index: ScalarIndexState::None,
                nullability: Nullability::Required,
                default_value: None,
            },
            ScalarFieldSchema {
                name: field_name("rank"),
                field_type: ScalarType::Int64,
                index: ScalarIndexState::None,
                nullability: Nullability::Required,
                default_value: None,
            },
            ScalarFieldSchema {
                name: field_name("category"),
                field_type: ScalarType::String,
                index: ScalarIndexState::Indexed,
                nullability: Nullability::Required,
                default_value: None,
            },
        ],
        vector: VectorFieldSchema {
            name: field_name("embedding"),
            dimension: VectorDimension::new(4).expect("valid dimension"),
            metric: DistanceMetric::Cosine,
            indexes: VectorIndexState::DefaultFlat,
        },
    }
}

fn schema(indexes: VectorIndexState) -> CollectionSchema {
    let mut schema = indexed_schema();
    schema.vector.indexes = indexes;
    schema
}

fn field_name(value: &str) -> FieldName {
    FieldName::parse(value).expect("valid field name")
}
