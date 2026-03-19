use crate::query::parse_required_filter;
use crate::state::{CollectionRuntime, WriteMode};
use garuda_meta::evaluate_filter;
use garuda_segment::{WalOp, append_wal_ops};
use garuda_storage::WRITING_SEGMENT_ID;
use garuda_types::{Doc, DocId, Status, StatusCode, WriteResult};

pub(crate) enum WriteCommand {
    Insert(Vec<Doc>),
    Upsert(Vec<Doc>),
    Update(Vec<Doc>),
    Delete(Vec<DocId>),
}

pub(crate) fn apply_write_command(
    state: &mut CollectionRuntime,
    command: WriteCommand,
) -> Vec<WriteResult> {
    match command {
        WriteCommand::Insert(docs) => apply_doc_batch(state, docs, WalOp::Insert, |state, doc| {
            state.insert_doc(doc, WriteMode::Insert)
        }),
        WriteCommand::Upsert(docs) => apply_doc_batch(state, docs, WalOp::Upsert, |state, doc| {
            state.insert_doc(doc, WriteMode::Upsert)
        }),
        WriteCommand::Update(docs) => apply_doc_batch(state, docs, WalOp::Update, |state, doc| {
            state.update_doc(doc)
        }),
        WriteCommand::Delete(ids) => apply_delete_batch(state, ids),
    }
}

pub(crate) fn replay_wal_ops(
    state: &mut CollectionRuntime,
    wal_ops: Vec<WalOp>,
) -> Result<(), Status> {
    for wal_op in wal_ops {
        apply_replayed_wal_op(state, wal_op)?;
    }

    Ok(())
}

pub(crate) fn apply_delete_by_filter(
    state: &mut CollectionRuntime,
    raw_filter: &str,
) -> Result<(), Status> {
    let filter = parse_required_filter(raw_filter, &state.catalog.schema)?;
    let ids = collect_matching_doc_ids(state, &filter);
    if ids.is_empty() {
        return Ok(());
    }

    let results = apply_delete_batch(state, ids);
    for result in results {
        if result.status.is_ok() {
            continue;
        }

        return Err(result.status);
    }

    Ok(())
}

fn apply_doc_batch(
    state: &mut CollectionRuntime,
    docs: Vec<Doc>,
    wal_op: impl Fn(Doc) -> WalOp,
    write_one: impl Fn(&mut CollectionRuntime, Doc) -> WriteResult,
) -> Vec<WriteResult> {
    let snapshot = state.clone();
    let mut results = Vec::with_capacity(docs.len());
    let mut wal_ops = Vec::with_capacity(docs.len());

    for doc in docs {
        let wal_doc = doc.clone();
        let result = write_one(state, doc);
        if result.status.is_ok() {
            wal_ops.push(wal_op(wal_doc));
        }

        results.push(result);
    }

    finish_batch(state, snapshot, &mut results, wal_ops);
    results
}

fn apply_delete_batch(state: &mut CollectionRuntime, ids: Vec<DocId>) -> Vec<WriteResult> {
    let snapshot = state.clone();
    let mut results = Vec::with_capacity(ids.len());
    let mut wal_ops = Vec::with_capacity(ids.len());

    for id in ids {
        let result = state.delete_doc(id.clone());
        if result.status.is_ok() {
            wal_ops.push(WalOp::Delete(id));
        }

        results.push(result);
    }

    finish_batch(state, snapshot, &mut results, wal_ops);
    results
}

fn collect_matching_doc_ids(
    state: &CollectionRuntime,
    filter: &garuda_types::FilterExpr,
) -> Vec<DocId> {
    let mut ids = Vec::new();

    for record in state.all_live_records() {
        if !evaluate_filter(filter, &record.doc.fields) {
            continue;
        }

        ids.push(record.doc.id);
    }

    ids
}

fn finish_batch(
    state: &mut CollectionRuntime,
    snapshot: CollectionRuntime,
    results: &mut [WriteResult],
    wal_ops: Vec<WalOp>,
) {
    let persist_result = if wal_ops.is_empty() {
        Ok(())
    } else {
        append_wal_ops(&state.path, WRITING_SEGMENT_ID, &wal_ops)
    };

    if let Err(status) = persist_result {
        *state = snapshot;
        mark_persist_failure(results, &status);
    }
}

fn mark_persist_failure(results: &mut [WriteResult], status: &Status) {
    for result in results {
        if !result.status.is_ok() {
            continue;
        }

        result.status = Status::err(status.code.clone(), status.message.clone());
    }
}

fn apply_replayed_wal_op(state: &mut CollectionRuntime, wal_op: WalOp) -> Result<(), Status> {
    if is_redundant_wal_op(state, &wal_op) {
        return Ok(());
    }

    let update_not_found = matches!(wal_op, WalOp::Update(_));
    let result = match wal_op {
        WalOp::Insert(doc) => state.insert_doc(doc, WriteMode::Insert),
        WalOp::Upsert(doc) => state.insert_doc(doc, WriteMode::Upsert),
        WalOp::Update(doc) => state.update_doc(doc),
        WalOp::Delete(doc_id) => state.delete_doc(doc_id),
    };

    if result.status.is_ok() {
        return Ok(());
    }

    if update_not_found && result.status.code == StatusCode::NotFound {
        return Ok(());
    }

    Err(Status::err(result.status.code, result.status.message))
}

fn is_redundant_wal_op(state: &CollectionRuntime, wal_op: &WalOp) -> bool {
    match wal_op {
        WalOp::Insert(doc) => state.find_live_record(&doc.id).is_some(),
        WalOp::Upsert(doc) => live_doc_matches(state, doc),
        WalOp::Update(doc) => {
            update_already_applied(state, doc) || state.find_live_record(&doc.id).is_none()
        }
        WalOp::Delete(doc_id) => state.find_live_record(doc_id).is_none(),
    }
}

fn live_doc_matches(state: &CollectionRuntime, doc: &Doc) -> bool {
    let Some(record) = state.find_live_record(&doc.id) else {
        return false;
    };

    record.doc == *doc
}

fn update_already_applied(state: &CollectionRuntime, doc: &Doc) -> bool {
    let Some(record) = state.find_live_record(&doc.id) else {
        return false;
    };

    let merged_doc = merge_docs(&record.doc, doc);
    merged_doc == record.doc
}

fn merge_docs(existing: &Doc, incoming: &Doc) -> Doc {
    let mut merged = existing.clone();

    for (key, value) in &incoming.fields {
        merged.fields.insert(key.clone(), value.clone());
    }

    if !incoming.vector.is_empty() {
        merged.vector = incoming.vector.clone();
    }

    merged
}
