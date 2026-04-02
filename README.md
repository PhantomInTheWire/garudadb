# GarudaDB

A vector database built from scratch in Rust.

## Features

- **Multiple ANN index types** — choose between Flat (brute-force), HNSW, or IVF per collection
- **Scalar field filtering** — filter on `bool`, `int64`, `float64`, and `string` fields alongside vector queries
- **Scalar index pushdown** — indexed scalar predicates in `AND` filters are evaluated before ANN search, shrinking the candidate set before touching the vector index
- **Segmented storage** — a write-ahead writing segment plus compacted persisted segments, keeping reads fast while inserts stay live
- **Crash recovery** — periodic checkpointing and a recovery service restore consistent state on restart
- **Cosine similarity** — vectors are compared using cosine distance

## What Makes It Different

Most embeddable vector search libraries are thin wrappers around a single ANN algorithm. GarudaDB is a full database engine — it owns its own storage, its own query planner, and its own index implementations, all written in safe Rust with zero external database dependencies.

The query planner is the core differentiator: it decomposes filter expressions to push equality and range predicates on indexed scalar fields down to a prefilter pass that runs before the ANN search. Only documents that survive the prefilter are considered as ANN candidates, which keeps approximate search accurate and avoids scanning the full index when filters are selective.

## Usage

```sh
cargo build --release
# binary is at target/release/garuda
```

Initialize a database, create a collection, and insert documents:

```sh
garuda --root ./mydb init
garuda --root ./mydb create products 768
garuda --root ./mydb insert-jsonl products docs.jsonl
```

Each line of the JSONL file must follow this shape:

```json
{"id": "doc1", "fields": {"rank": 1, "category": "electronics", "score": 9.5}, "vector": [0.1, 0.2, 0.3]}
```

Run a nearest-neighbor query and fetch a document by ID:

```sh
garuda --root ./mydb query products --vector "0.1,0.2,0.3" --top-k 10
garuda --root ./mydb fetch products doc1
```

Build indexes to speed up search and filtering:

```sh
garuda --root ./mydb create-index products embedding hnsw
garuda --root ./mydb create-index products category scalar
```

Inspect a collection:

```sh
garuda --root ./mydb stats products
```

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).