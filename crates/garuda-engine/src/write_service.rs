use crate::state::{CollectionState, WriteMode};
use garuda_segment::{WalOp, append_wal_ops};
use garuda_storage::WRITING_SEGMENT_ID;
use garuda_types::{Doc, DocId, Status, WriteResult};

pub(crate) enum WriteCommand {
    Insert(Vec<Doc>),
    Upsert(Vec<Doc>),
    Update(Vec<Doc>),
    Delete(Vec<DocId>),
}

pub(crate) fn apply_write_command(
    state: &mut CollectionState,
    command: WriteCommand,
) -> Vec<WriteResult> {
    match command {
        WriteCommand::Insert(docs) => apply_doc_batch(state, docs, WalOp::Insert, |state, doc| {
            state.insert_doc(doc, WriteMode::Insert)
        }),
        WriteCommand::Upsert(docs) => apply_doc_batch(state, docs, WalOp::Upsert, |state, doc| {
            state.insert_doc(doc, WriteMode::Upsert)
        }),
        WriteCommand::Update(docs) => {
            apply_doc_batch(state, docs, WalOp::Update, |state, doc| state.update_doc(doc))
        }
        WriteCommand::Delete(ids) => apply_delete_batch(state, ids),
    }
}

fn apply_doc_batch(
    state: &mut CollectionState,
    docs: Vec<Doc>,
    wal_op: impl Fn(Doc) -> WalOp,
    write_one: impl Fn(&mut CollectionState, Doc) -> WriteResult,
) -> Vec<WriteResult> {
    let snapshot = state.clone();
    let mut results = Vec::new();
    let mut wal_ops = Vec::new();

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

fn apply_delete_batch(state: &mut CollectionState, ids: Vec<DocId>) -> Vec<WriteResult> {
    let snapshot = state.clone();
    let mut results = Vec::new();
    let mut wal_ops = Vec::new();

    for id in ids {
        let result = state.delete_doc(&id);
        if result.status.is_ok() {
            wal_ops.push(WalOp::Delete(id.clone()));
        }

        results.push(result);
    }

    finish_batch(state, snapshot, &mut results, wal_ops);
    results
}

fn finish_batch(
    state: &mut CollectionState,
    snapshot: CollectionState,
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
