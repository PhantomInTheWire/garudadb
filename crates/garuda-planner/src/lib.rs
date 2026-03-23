use garuda_types::{
    CollectionSchema, FieldName, FilterExpr, HnswEfSearch, IndexKind, IvfProbeCount,
    QueryVectorSource, ScalarCompareOp, ScalarFieldSchema, ScalarPredicate, ScalarPrefilter,
    ScalarType, ScalarValue, TopK, VectorProjection, VectorQuery, VectorSearch,
};

#[derive(Clone, Debug, PartialEq)]
pub enum SegmentFilterPlan {
    All,
    Matching(FilterExpr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilterPlan {
    pub scalar_prefilter: ScalarPrefilter,
    pub residual: SegmentFilterPlan,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SegmentSearchPlan {
    Flat,
    Hnsw { ef_search: HnswEfSearch },
    Ivf { nprobe: IvfProbeCount },
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryPlan {
    pub field_name: FieldName,
    pub source: QueryVectorSource,
    pub filter: FilterPlan,
    pub top_k: TopK,
    pub search: SegmentSearchPlan,
    pub vector_projection: VectorProjection,
    pub output_fields: Option<Vec<String>>,
}

pub fn build_query_plan(
    query: VectorQuery,
    filter: Option<FilterExpr>,
    schema: &CollectionSchema,
) -> Result<QueryPlan, garuda_types::Status> {
    Ok(QueryPlan {
        field_name: query.field_name,
        source: query.source,
        filter: build_filter_plan(filter, schema),
        top_k: query.top_k,
        search: search_plan(query.search, schema)?,
        vector_projection: query.vector_projection,
        output_fields: query.output_fields,
    })
}

fn search_plan(
    search: VectorSearch,
    schema: &CollectionSchema,
) -> Result<SegmentSearchPlan, garuda_types::Status> {
    match (schema.vector.indexes.default_kind(), search) {
        (IndexKind::Flat, VectorSearch::Default) => Ok(SegmentSearchPlan::Flat),
        (IndexKind::Flat, _) => Err(garuda_types::Status::err(
            garuda_types::StatusCode::InvalidArgument,
            "query search override does not match the default index",
        )),
        (IndexKind::Hnsw, VectorSearch::Default) => Ok(SegmentSearchPlan::Hnsw {
            ef_search: schema
                .vector
                .indexes
                .hnsw_params()
                .expect("hnsw default requires hnsw params")
                .ef_search,
        }),
        (IndexKind::Hnsw, VectorSearch::Hnsw { ef_search }) => {
            Ok(SegmentSearchPlan::Hnsw { ef_search })
        }
        (IndexKind::Hnsw, VectorSearch::Ivf { .. }) => Err(garuda_types::Status::err(
            garuda_types::StatusCode::InvalidArgument,
            "query search override does not match the default index",
        )),
        (IndexKind::Ivf, VectorSearch::Default) => Ok(SegmentSearchPlan::Ivf {
            nprobe: schema
                .vector
                .indexes
                .ivf_params()
                .expect("ivf default requires ivf params")
                .n_probe,
        }),
        (IndexKind::Ivf, VectorSearch::Ivf { nprobe }) => Ok(SegmentSearchPlan::Ivf { nprobe }),
        (IndexKind::Ivf, VectorSearch::Hnsw { .. }) => Err(garuda_types::Status::err(
            garuda_types::StatusCode::InvalidArgument,
            "query search override does not match the default index",
        )),
        (IndexKind::Scalar, _) => panic!("vector index default cannot be scalar"),
    }
}

fn build_filter_plan(filter: Option<FilterExpr>, schema: &CollectionSchema) -> FilterPlan {
    let Some(filter) = filter else {
        return FilterPlan {
            scalar_prefilter: ScalarPrefilter::All,
            residual: SegmentFilterPlan::All,
        };
    };

    let mut scalar_predicates = Vec::new();
    let mut residual_terms = Vec::new();

    if !collect_filter_terms(&filter, schema, &mut scalar_predicates, &mut residual_terms) {
        return FilterPlan {
            scalar_prefilter: ScalarPrefilter::All,
            residual: SegmentFilterPlan::Matching(filter),
        };
    }

    FilterPlan {
        scalar_prefilter: if scalar_predicates.is_empty() {
            ScalarPrefilter::All
        } else {
            ScalarPrefilter::And(scalar_predicates)
        },
        residual: if residual_terms.is_empty() {
            SegmentFilterPlan::All
        } else {
            let mut residual_terms = residual_terms;
            let first = residual_terms.remove(0);
            SegmentFilterPlan::Matching(residual_terms.into_iter().fold(first, |lhs, rhs| {
                FilterExpr::And(Box::new(lhs), Box::new(rhs))
            }))
        },
    }
}

fn collect_filter_terms(
    filter: &FilterExpr,
    schema: &CollectionSchema,
    scalar_predicates: &mut Vec<ScalarPredicate>,
    residual_terms: &mut Vec<FilterExpr>,
) -> bool {
    match filter {
        FilterExpr::And(lhs, rhs) => {
            collect_filter_terms(lhs, schema, scalar_predicates, residual_terms)
                && collect_filter_terms(rhs, schema, scalar_predicates, residual_terms)
        }
        FilterExpr::Or(_, _) => false,
        _ => {
            let Some(predicate) = pushdown_predicate(filter, schema) else {
                residual_terms.push(filter.clone());
                return true;
            };

            scalar_predicates.push(predicate);
            true
        }
    }
}

fn pushdown_predicate(filter: &FilterExpr, schema: &CollectionSchema) -> Option<ScalarPredicate> {
    match filter {
        FilterExpr::Eq(field, value) => scalar_predicate(schema, field, ScalarCompareOp::Eq, value),
        FilterExpr::Gt(field, value) => scalar_predicate(schema, field, ScalarCompareOp::Gt, value),
        FilterExpr::Gte(field, value) => {
            scalar_predicate(schema, field, ScalarCompareOp::Gte, value)
        }
        FilterExpr::Lt(field, value) => scalar_predicate(schema, field, ScalarCompareOp::Lt, value),
        FilterExpr::Lte(field, value) => {
            scalar_predicate(schema, field, ScalarCompareOp::Lte, value)
        }
        FilterExpr::Ne(_, _)
        | FilterExpr::StringMatch(_, _)
        | FilterExpr::IsNull(_)
        | FilterExpr::And(_, _)
        | FilterExpr::Or(_, _) => None,
    }
}

fn scalar_predicate(
    schema: &CollectionSchema,
    field: &str,
    op: ScalarCompareOp,
    value: &ScalarValue,
) -> Option<ScalarPredicate> {
    let field_schema = schema.scalar_field_by_name(field)?;

    if !supports_pushdown(field_schema, op, value) {
        return None;
    }

    Some(ScalarPredicate {
        field: FieldName::parse(field.to_string()).expect("validated filter field"),
        op,
        value: value.clone(),
    })
}

fn supports_pushdown(field: &ScalarFieldSchema, op: ScalarCompareOp, value: &ScalarValue) -> bool {
    if !field.is_indexed() {
        return false;
    }

    if matches!(value, ScalarValue::Null) {
        return false;
    }

    matches!(
        (field.field_type, op, value),
        (ScalarType::Bool, ScalarCompareOp::Eq, ScalarValue::Bool(_))
            | (ScalarType::Int64, _, ScalarValue::Int64(_))
            | (ScalarType::Float64, _, ScalarValue::Float64(_))
            | (ScalarType::String, _, ScalarValue::String(_))
    )
}
