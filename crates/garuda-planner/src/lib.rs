//! Query planning for Garuda's vector search path.
//!
//! This crate maps a validated user query plus collection schema into a small,
//! explicit execution plan. It does not execute indexes, estimate global
//! costs, or build a general SQL plan tree.
//!
//! Inputs:
//! - `CollectionSchema`: field types, index state, and default vector backend.
//! - `VectorQuery`: vector source, `top_k`, projection knobs, and optional
//!   backend override.
//! - `Option<FilterExpr>`: the already-parsed filter tree, if any.
//!
//! Outputs:
//! - `QueryPlan`: one typed object that carries:
//!   - query-vector source
//!   - projection
//!   - filter split
//!   - vector recall plan
//! - `FilterPlan`: scalar prefilter vs residual filter.
//! - `RecallPlan`: Flat, HNSW, or IVF with a concrete budget policy.
//!
//! Planning steps:
//! 1. Choose the vector backend from schema state and query override.
//! 2. Split indexed scalar `AND` predicates into a scalar prefilter.
//! 3. Keep unsupported predicates as a residual filter.
//! 4. Choose `Requested` vs `AdaptiveFiltered` budgeting.
//! 5. Carry query-vector source and projection through unchanged.
//!
//! Key semantics:
//! - planner does not execute indexes
//! - planner does not estimate collection-global costs
//! - planner does not switch ANN families at runtime
//! - unsupported filter operators stay residual
//!
//! Non-goals:
//! - no SQL optimizer
//! - no Arrow or physical operator tree
//! - no join or aggregate planning

use garuda_types::{
    AnnBudgetPolicy, CollectionSchema, FieldName, FilterExpr, FlatRecallPlan, HnswRecallPlan,
    IvfRecallPlan, QueryVectorSource, RecallPlan, ScalarCompareOp, ScalarFieldSchema,
    ScalarPredicate, ScalarType, ScalarValue, TopK, VectorIndexState, VectorProjection,
    VectorQuery, VectorSearch,
};

#[derive(Clone, Debug, PartialEq)]
pub enum PrefilterPlan {
    All,
    ScalarAnd(Vec<ScalarPredicate>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResidualPlan {
    All,
    Filter(FilterExpr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilterPlan {
    pub prefilter: PrefilterPlan,
    pub residual: ResidualPlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionPlan {
    pub vector: VectorProjection,
    pub output_fields: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryPlan {
    pub field_name: FieldName,
    pub source: QueryVectorSource,
    pub filter: FilterPlan,
    pub recall: RecallPlan,
    pub projection: ProjectionPlan,
}

pub fn build_query_plan(
    query: VectorQuery,
    filter: Option<FilterExpr>,
    schema: &CollectionSchema,
) -> Result<QueryPlan, garuda_types::Status> {
    let filter = build_filter_plan(filter, schema);

    Ok(QueryPlan {
        field_name: query.field_name,
        source: query.source,
        recall: build_recall_plan(query.search, query.top_k, &filter, schema)?,
        projection: ProjectionPlan {
            vector: query.vector_projection,
            output_fields: query.output_fields,
        },
        filter,
    })
}

fn build_recall_plan(
    search: VectorSearch,
    top_k: TopK,
    filter: &FilterPlan,
    schema: &CollectionSchema,
) -> Result<RecallPlan, garuda_types::Status> {
    let budget = budget_policy(filter);

    match (&schema.vector.indexes, search) {
        (VectorIndexState::DefaultFlat, VectorSearch::Default)
        | (
            VectorIndexState::FlatAndHnsw {
                default: garuda_types::FlatHnswDefault::Flat,
                ..
            },
            VectorSearch::Default,
        )
        | (
            VectorIndexState::FlatAndIvf {
                default: garuda_types::FlatIvfDefault::Flat,
                ..
            },
            VectorSearch::Default,
        ) => Ok(RecallPlan::Flat(FlatRecallPlan { top_k })),
        (VectorIndexState::DefaultFlat, _)
        | (
            VectorIndexState::FlatAndHnsw {
                default: garuda_types::FlatHnswDefault::Flat,
                ..
            },
            _,
        )
        | (
            VectorIndexState::FlatAndIvf {
                default: garuda_types::FlatIvfDefault::Flat,
                ..
            },
            _,
        ) => Err(garuda_types::Status::err(
            garuda_types::StatusCode::InvalidArgument,
            "query search override does not match the default index",
        )),
        (VectorIndexState::HnswOnly(params), VectorSearch::Default)
        | (
            VectorIndexState::FlatAndHnsw {
                default: garuda_types::FlatHnswDefault::Hnsw,
                hnsw: params,
            },
            VectorSearch::Default,
        ) => Ok(RecallPlan::Hnsw(HnswRecallPlan {
            top_k,
            ef_search: params.ef_search,
            budget,
        })),
        (VectorIndexState::HnswOnly(_), VectorSearch::Hnsw { ef_search })
        | (
            VectorIndexState::FlatAndHnsw {
                default: garuda_types::FlatHnswDefault::Hnsw,
                ..
            },
            VectorSearch::Hnsw { ef_search },
        ) => Ok(RecallPlan::Hnsw(HnswRecallPlan {
            top_k,
            ef_search,
            budget,
        })),
        (VectorIndexState::HnswOnly(_), VectorSearch::Ivf { .. })
        | (
            VectorIndexState::FlatAndHnsw {
                default: garuda_types::FlatHnswDefault::Hnsw,
                ..
            },
            VectorSearch::Ivf { .. },
        ) => Err(garuda_types::Status::err(
            garuda_types::StatusCode::InvalidArgument,
            "query search override does not match the default index",
        )),
        (VectorIndexState::IvfOnly(params), VectorSearch::Default)
        | (
            VectorIndexState::FlatAndIvf {
                default: garuda_types::FlatIvfDefault::Ivf,
                ivf: params,
            },
            VectorSearch::Default,
        ) => Ok(RecallPlan::Ivf(IvfRecallPlan {
            top_k,
            nprobe: params.n_probe,
            budget,
        })),
        (VectorIndexState::IvfOnly(_), VectorSearch::Ivf { nprobe })
        | (
            VectorIndexState::FlatAndIvf {
                default: garuda_types::FlatIvfDefault::Ivf,
                ..
            },
            VectorSearch::Ivf { nprobe },
        ) => Ok(RecallPlan::Ivf(IvfRecallPlan {
            top_k,
            nprobe,
            budget,
        })),
        (VectorIndexState::IvfOnly(_), VectorSearch::Hnsw { .. })
        | (
            VectorIndexState::FlatAndIvf {
                default: garuda_types::FlatIvfDefault::Ivf,
                ..
            },
            VectorSearch::Hnsw { .. },
        ) => Err(garuda_types::Status::err(
            garuda_types::StatusCode::InvalidArgument,
            "query search override does not match the default index",
        )),
    }
}

fn budget_policy(filter: &FilterPlan) -> AnnBudgetPolicy {
    match (&filter.prefilter, &filter.residual) {
        (PrefilterPlan::All, ResidualPlan::All) => AnnBudgetPolicy::Requested,
        _ => AnnBudgetPolicy::AdaptiveFiltered,
    }
}

fn build_filter_plan(filter: Option<FilterExpr>, schema: &CollectionSchema) -> FilterPlan {
    let Some(filter) = filter else {
        return FilterPlan {
            prefilter: PrefilterPlan::All,
            residual: ResidualPlan::All,
        };
    };

    let mut scalar_predicates = Vec::new();
    let mut residual_terms = Vec::new();

    if !collect_filter_terms(&filter, schema, &mut scalar_predicates, &mut residual_terms) {
        return FilterPlan {
            prefilter: PrefilterPlan::All,
            residual: ResidualPlan::Filter(filter),
        };
    }

    FilterPlan {
        prefilter: if scalar_predicates.is_empty() {
            PrefilterPlan::All
        } else {
            PrefilterPlan::ScalarAnd(scalar_predicates)
        },
        residual: if residual_terms.is_empty() {
            ResidualPlan::All
        } else {
            ResidualPlan::Filter(and_expr(residual_terms))
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

fn and_expr(mut terms: Vec<FilterExpr>) -> FilterExpr {
    let first = terms.remove(0);
    terms.into_iter().fold(first, |lhs, rhs| {
        FilterExpr::And(Box::new(lhs), Box::new(rhs))
    })
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
