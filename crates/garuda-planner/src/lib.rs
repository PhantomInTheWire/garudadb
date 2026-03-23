use garuda_types::{
    CollectionSchema, FieldName, FilterExpr, HnswEfSearch, IndexKind, QueryVectorSource,
    ScalarCompareOp, ScalarFieldSchema, ScalarPredicate, ScalarPrefilter, ScalarType, ScalarValue,
    TopK, VectorProjection, VectorQuery,
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
) -> QueryPlan {
    QueryPlan {
        field_name: query.field_name,
        source: query.source,
        filter: build_filter_plan(filter, schema),
        top_k: query.top_k,
        search: search_plan(query.ef_search, schema),
        vector_projection: query.vector_projection,
        output_fields: query.output_fields,
    }
}

fn search_plan(
    query_ef_search: Option<HnswEfSearch>,
    schema: &CollectionSchema,
) -> SegmentSearchPlan {
    match schema.vector.indexes.default_kind() {
        IndexKind::Flat => SegmentSearchPlan::Flat,
        IndexKind::Hnsw => SegmentSearchPlan::Hnsw {
            ef_search: query_ef_search.unwrap_or(
                schema
                    .vector
                    .indexes
                    .hnsw_params()
                    .expect("hnsw default requires hnsw params")
                    .ef_search,
            ),
        },
        IndexKind::Scalar => panic!("vector index default cannot be scalar"),
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
            SegmentFilterPlan::Matching(and_expr(residual_terms))
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
    let field_schema = schema
        .fields
        .iter()
        .find(|candidate| candidate.name.as_str() == field)?;

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

fn and_expr(mut filters: Vec<FilterExpr>) -> FilterExpr {
    let first = filters.remove(0);
    filters.into_iter().fold(first, |lhs, rhs| {
        FilterExpr::And(Box::new(lhs), Box::new(rhs))
    })
}
