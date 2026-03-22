use garuda_types::{
    FieldName, FilterExpr, HnswEfSearch, IndexKind, QueryVectorSource, TopK, VectorIndexState,
    VectorProjection, VectorQuery,
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
    pub filter: SegmentFilterPlan,
    pub top_k: TopK,
    pub search: SegmentSearchPlan,
    pub vector_projection: VectorProjection,
    pub output_fields: Option<Vec<String>>,
}

pub fn build_query_plan(
    query: VectorQuery,
    filter: Option<FilterExpr>,
    indexes: &VectorIndexState,
) -> QueryPlan {
    let filter = match filter {
        Some(filter) => SegmentFilterPlan::Matching(filter),
        None => SegmentFilterPlan::All,
    };

    QueryPlan {
        field_name: query.field_name,
        source: query.source,
        filter,
        top_k: query.top_k,
        search: search_plan(query.ef_search, indexes),
        vector_projection: query.vector_projection,
        output_fields: query.output_fields,
    }
}

fn search_plan(
    query_ef_search: Option<HnswEfSearch>,
    indexes: &VectorIndexState,
) -> SegmentSearchPlan {
    match indexes.default_kind() {
        IndexKind::Flat => SegmentSearchPlan::Flat,
        IndexKind::Hnsw => SegmentSearchPlan::Hnsw {
            ef_search: query_ef_search.unwrap_or(
                indexes
                    .hnsw_params()
                    .expect("hnsw default requires hnsw params")
                    .ef_search,
            ),
        },
    }
}
