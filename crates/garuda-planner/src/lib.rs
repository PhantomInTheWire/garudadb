use garuda_types::{FieldName, FilterExpr, QueryVectorSource, TopK, VectorProjection, VectorQuery};

#[derive(Clone, Debug, PartialEq)]
pub enum SegmentFilterPlan {
    All,
    Matching(FilterExpr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFlatPlan {
    pub source: QueryVectorSource,
    pub filter: SegmentFilterPlan,
    pub top_k: TopK,
    pub ef_search: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SegmentExecutionPlan {
    ExactFlat(ExactFlatPlan),
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryPlan {
    pub field_name: FieldName,
    pub execution: SegmentExecutionPlan,
    pub vector_projection: VectorProjection,
    pub output_fields: Option<Vec<String>>,
}

impl QueryPlan {
    pub fn exact_flat(&self) -> &ExactFlatPlan {
        match &self.execution {
            SegmentExecutionPlan::ExactFlat(execution) => execution,
        }
    }
}

pub fn build_query_plan(query: VectorQuery, filter: Option<FilterExpr>) -> QueryPlan {
    let filter = match filter {
        Some(filter) => SegmentFilterPlan::Matching(filter),
        None => SegmentFilterPlan::All,
    };

    QueryPlan {
        field_name: query.field_name,
        execution: SegmentExecutionPlan::ExactFlat(ExactFlatPlan {
            source: query.source,
            filter,
            top_k: query.top_k,
            ef_search: query.ef_search,
        }),
        vector_projection: query.vector_projection,
        output_fields: query.output_fields,
    }
}
