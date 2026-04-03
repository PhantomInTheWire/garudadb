# GarudaDB

An embeddable vector database built from *scratch* in Rust.

GarudaDB is laid out as a small storage engine with its own query planner, segmented storage layer, WAL, checkpoints, recovery path, scalar indexes, and vector indexes in one codebase. Writes land in an active segment, sealed segments reopen from persisted state, and queries plan across scalar filters and ANN indexes instead of treating storage and search as separate systems.

The aim is to use this and build a local first RAG/memory experience, build a sync engine on top of this to replicate [what notion does with sqlite](https://www.notion.com/blog/how-we-sped-up-notion-in-the-browser-with-wasm-sqlite)

## Highlights

- Handwritten SIMD scoring
- Flat, HNSW, and IVF indexes
- Segmented storage, WAL, checkpoints, and recovery
- Hybrid query planning with scalar prefilter pushdown
- Recovery follows the saved manifest instead of guessing from snapshot filenames
- Vector search automatically looks wider when filters would otherwise hide good matches
- Saved HNSW graphs are checked on reopen, and stale or corrupt data is rejected
- Delete-aware HNSW and incremental IVF maintenance

## Layout

```text
collection/
├── manifest.N
├── idmap.N
├── del.N
├── 0/                  # active writing segment
│   ├── data.seg
│   ├── data.wal
│   ├── [flat.idx]
│   ├── [hnsw.idx]
│   ├── [ivf.idx]
│   └── [scalar/]
└── 1/                  # sealed segment
    ├── data.seg
    ├── [flat.idx]
    ├── [hnsw.idx]
    ├── [ivf.idx]
    └── [scalar/]
```

## Future work

The next major area of work is memory-efficient vector search. I want to add a range of quantization techniques, including IVF-PQ, RaBitQ, and a TurboQuant-inspired rotation approach.

After that, the next step is a dedicated sync crate so GarudaDB can support a proper local-first replication model.

Once the sync layer exists, I want to improve retrieval quality with a custom reranking model, mostly by fine-tuning on synthetic data.

That will be followed by broader retrieval infrastructure:

- a standard BM25 text search implementation built from scratch
- [Cursor's fast regex search](https://cursor.com/blog/fast-regex-search) reimplemented from scratch
- a ChromaFS-style virtual filesystem layer inspired by Mintlify's [chromaFS](https://www.mintlify.com/blog/how-we-built-a-virtual-filesystem-for-our-assistant), effectively giving the agent a fake bash/search surface

All of these pieces will eventually come together in the final local-first RAG and memory system, with the different retrieval methods feeding into the custom reranker.

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
