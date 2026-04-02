mod common;

use common::{build_doc, database, default_options, default_schema, dense_vector, field_name};
use garuda_types::{
    IndexParams, IvfIndexParams, IvfListCount, IvfProbeCount, IvfTrainingIterations, TopK,
    VectorQuery,
};
use std::time::{Duration, Instant};

const TOTAL_DOCS: u64 = 2_000;
const DELETE_DOCS: u64 = 1_200;
const QUERY_ROUNDS: usize = 100;
const SEGMENT_MAX_DOCS: usize = 10_000;

#[test]
#[ignore = "benchmark hook"]
fn benchmark_delete_churn_hnsw() {
    eprintln!("benchmark start index=hnsw");
    let result = run_delete_churn_benchmark(IndexParams::Hnsw(Default::default()));
    print_result("hnsw", &result);
}

#[test]
#[ignore = "benchmark hook"]
fn benchmark_delete_churn_ivf() {
    eprintln!("benchmark start index=ivf");
    let result = run_delete_churn_benchmark(IndexParams::Ivf(IvfIndexParams {
        n_list: IvfListCount::new(8).expect("valid list count"),
        n_probe: IvfProbeCount::new(2).expect("valid probe count"),
        training_iterations: IvfTrainingIterations::new(2).expect("valid training iterations"),
    }));
    print_result("ivf", &result);
}

struct DeleteChurnBenchmarkResult {
    insert: Duration,
    index_create: Duration,
    delete: Duration,
    query: Duration,
}

fn run_delete_churn_benchmark(index: IndexParams) -> DeleteChurnBenchmarkResult {
    let (_root, db) = database("delete-churn-benchmark");
    let mut options = default_options();
    options.segment_max_docs = SEGMENT_MAX_DOCS;
    let collection = db
        .create_collection(default_schema("docs"), options)
        .expect("create collection");

    let insert_start = Instant::now();
    eprintln!("phase=insert start");
    let docs = build_docs();
    let inserted = collection.insert(docs);
    assert!(inserted.iter().all(|result| result.status.is_ok()));
    collection.flush().expect("flush");
    let insert = insert_start.elapsed();
    eprintln!("phase=insert done ms={}", insert.as_millis());

    let index_create_start = Instant::now();
    eprintln!("phase=index_create start");
    collection
        .create_index(&field_name("embedding"), index)
        .expect("create index");
    let index_create = index_create_start.elapsed();
    eprintln!("phase=index_create done ms={}", index_create.as_millis());

    let delete_start = Instant::now();
    eprintln!("phase=delete start");
    let deleted = collection.delete(build_delete_ids());
    assert!(deleted.iter().all(|result| result.status.is_ok()));
    let delete = delete_start.elapsed();
    eprintln!("phase=delete done ms={}", delete.as_millis());

    let query_start = Instant::now();
    eprintln!("phase=query start rounds={QUERY_ROUNDS}");
    for _ in 0..QUERY_ROUNDS {
        collection
            .query(VectorQuery::by_vector(
                field_name("embedding"),
                dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
                TopK::new(10).expect("valid top k"),
            ))
            .expect("query after delete churn");
    }
    let query = query_start.elapsed();
    eprintln!("phase=query done ms={}", query.as_millis());

    DeleteChurnBenchmarkResult {
        insert,
        index_create,
        delete,
        query,
    }
}

fn build_docs() -> Vec<garuda_types::Doc> {
    let mut docs = Vec::with_capacity(TOTAL_DOCS as usize);

    for raw in 1..=TOTAL_DOCS {
        let id = format!("doc-{raw}");
        let bucket = raw % 3;
        let vector = match bucket {
            0 => [1.0, 0.0, 0.0, 0.0],
            1 => [0.0, 1.0, 0.0, 0.0],
            _ => [0.0, 0.0, 1.0, 0.0],
        };

        docs.push(build_doc(&id, raw as i64, "alpha", 0.5, vector));
    }

    docs
}

fn build_delete_ids() -> Vec<garuda_types::DocId> {
    let mut ids = Vec::with_capacity(DELETE_DOCS as usize);
    for raw in 1..=DELETE_DOCS {
        ids.push(garuda_types::DocId::parse(format!("doc-{raw}")).expect("valid doc id"));
    }
    ids
}

fn print_result(name: &str, result: &DeleteChurnBenchmarkResult) {
    eprintln!(
        "delete-churn-benchmark index={name} insert_ms={} index_create_ms={} delete_ms={} query_ms={} query_rounds={}",
        result.insert.as_millis(),
        result.index_create.as_millis(),
        result.delete.as_millis(),
        result.query.as_millis(),
        QUERY_ROUNDS,
    );
}
