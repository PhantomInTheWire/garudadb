use garuda_types::{FieldName, FilterExpr, QueryVectorSource, TopK, VectorProjection, VectorQuery};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentScanMode {
    FullScan,
    FilteredScan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryPlan {
    pub field_name: FieldName,
    pub source: QueryVectorSource,
    pub filter: Option<FilterExpr>,
    pub scan_mode: SegmentScanMode,
    pub top_k: TopK,
    pub ef_search: Option<usize>,
    pub vector_projection: VectorProjection,
    pub output_fields: Option<Vec<String>>,
}

pub fn build_query_plan(query: VectorQuery, filter: Option<FilterExpr>) -> QueryPlan {
    let scan_mode = match filter {
        Some(_) => SegmentScanMode::FilteredScan,
        None => SegmentScanMode::FullScan,
    };

    QueryPlan {
        field_name: query.field_name,
        source: query.source,
        filter,
        scan_mode,
        top_k: query.top_k,
        ef_search: query.ef_search,
        vector_projection: query.vector_projection,
        output_fields: query.output_fields,
    }
}
