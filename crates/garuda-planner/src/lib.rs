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
pub enum SegmentSearchPlan {
    Flat,
    Hnsw { ef_search: HnswEfSearch },
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryPlan {
    pub field_name: FieldName,
    pub source: QueryVectorSource,
    pub scalar_prefilter: ScalarPrefilter,
    pub residual_filter: SegmentFilterPlan,
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
    let (scalar_prefilter, residual_filter) = build_filter_plan(filter, schema);

    QueryPlan {
        field_name: query.field_name,
        source: query.source,
        scalar_prefilter,
        residual_filter,
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

fn build_filter_plan(
    filter: Option<FilterExpr>,
    schema: &CollectionSchema,
) -> (ScalarPrefilter, SegmentFilterPlan) {
    let Some(filter) = filter else {
        return (ScalarPrefilter::All, SegmentFilterPlan::All);
    };

    if contains_or(&filter) {
        return (ScalarPrefilter::All, SegmentFilterPlan::Matching(filter));
    }

    let mut scalar_predicates = Vec::new();
    let mut residual_terms = Vec::new();

    for term in and_terms(&filter) {
        let Some(predicate) = scalar_predicate(term, schema) else {
            residual_terms.push(term.clone());
            continue;
        };

        scalar_predicates.push(predicate);
    }

    let scalar_prefilter = if scalar_predicates.is_empty() {
        ScalarPrefilter::All
    } else {
        ScalarPrefilter::And(scalar_predicates)
    };

    let residual_filter = if residual_terms.is_empty() {
        SegmentFilterPlan::All
    } else {
        SegmentFilterPlan::Matching(and_expr(residual_terms))
    };

    (scalar_prefilter, residual_filter)
}

fn contains_or(filter: &FilterExpr) -> bool {
    match filter {
        FilterExpr::Or(_, _) => true,
        FilterExpr::And(lhs, rhs) => contains_or(lhs) || contains_or(rhs),
        _ => false,
    }
}

fn and_terms(filter: &FilterExpr) -> Vec<&FilterExpr> {
    let mut terms = Vec::new();
    collect_and_terms(filter, &mut terms);
    terms
}

fn collect_and_terms<'a>(filter: &'a FilterExpr, out: &mut Vec<&'a FilterExpr>) {
    match filter {
        FilterExpr::And(lhs, rhs) => {
            collect_and_terms(lhs, out);
            collect_and_terms(rhs, out);
        }
        _ => out.push(filter),
    }
}

fn scalar_predicate(filter: &FilterExpr, schema: &CollectionSchema) -> Option<ScalarPredicate> {
    match filter {
        FilterExpr::Eq(field, value) => {
            build_scalar_predicate(field, ScalarCompareOp::Eq, value, schema)
        }
        FilterExpr::Gt(field, value) => {
            build_scalar_predicate(field, ScalarCompareOp::Gt, value, schema)
        }
        FilterExpr::Gte(field, value) => {
            build_scalar_predicate(field, ScalarCompareOp::Gte, value, schema)
        }
        FilterExpr::Lt(field, value) => {
            build_scalar_predicate(field, ScalarCompareOp::Lt, value, schema)
        }
        FilterExpr::Lte(field, value) => {
            build_scalar_predicate(field, ScalarCompareOp::Lte, value, schema)
        }
        FilterExpr::Ne(_, _)
        | FilterExpr::StringMatch(_, _)
        | FilterExpr::IsNull(_)
        | FilterExpr::And(_, _)
        | FilterExpr::Or(_, _) => None,
    }
}

fn build_scalar_predicate(
    field: &str,
    op: ScalarCompareOp,
    value: &ScalarValue,
    schema: &CollectionSchema,
) -> Option<ScalarPredicate> {
    let field_schema = schema
        .fields
        .iter()
        .find(|candidate| candidate.name.as_str() == field)?;

    if !supports_scalar_pushdown(field_schema, op, value) {
        return None;
    }

    Some(ScalarPredicate {
        field: FieldName::parse(field.to_string()).expect("validated filter field"),
        op,
        value: value.clone(),
    })
}

fn supports_scalar_pushdown(
    field: &ScalarFieldSchema,
    op: ScalarCompareOp,
    value: &ScalarValue,
) -> bool {
    if !field.is_indexed() {
        return false;
    }

    if matches!(value, ScalarValue::Null) {
        return false;
    }

    match (field.field_type, op, value) {
        (ScalarType::Bool, ScalarCompareOp::Eq, ScalarValue::Bool(_)) => true,
        (ScalarType::Int64, _, ScalarValue::Int64(_)) => true,
        (ScalarType::Float64, _, ScalarValue::Float64(_)) => true,
        (ScalarType::String, _, ScalarValue::String(_)) => true,
        _ => false,
    }
}

fn and_expr(mut filters: Vec<FilterExpr>) -> FilterExpr {
    let first = filters.remove(0);
    filters.into_iter().fold(first, |lhs, rhs| {
        FilterExpr::And(Box::new(lhs), Box::new(rhs))
    })
}
