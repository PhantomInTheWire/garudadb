use crate::segment_manager::SegmentManager;
use garuda_segment::{SegmentFile, StoredRecord};
use garuda_types::{FieldName, ScalarFieldSchema, ScalarValue};

pub(crate) fn backfill_new_column(segments: &mut SegmentManager, field: &ScalarFieldSchema) {
    let value = field.default_value.clone().unwrap_or(ScalarValue::Null);
    apply_to_all_segments(segments, |segment| {
        insert_field_into_records(&mut segment.records, field.name.as_str(), &value);
    });
}

pub(crate) fn rename_column(
    segments: &mut SegmentManager,
    old_name: &FieldName,
    new_name: &FieldName,
) {
    apply_to_all_segments(segments, |segment| {
        rename_field_in_records(&mut segment.records, old_name.as_str(), new_name.as_str());
    });
}

pub(crate) fn drop_column(segments: &mut SegmentManager, name: &FieldName) {
    apply_to_all_segments(segments, |segment| {
        remove_field_from_records(&mut segment.records, name.as_str());
    });
}

fn apply_to_all_segments(segments: &mut SegmentManager, mut apply: impl FnMut(&mut SegmentFile)) {
    for segment in segments.persisted_segments_mut() {
        apply(segment);
        segment.sync_meta();
    }

    let segment = segments.writing_segment_mut();
    apply(segment);
    segment.sync_meta();
}

fn insert_field_into_records(records: &mut [StoredRecord], field_name: &str, value: &ScalarValue) {
    for record in records {
        record
            .doc
            .fields
            .insert(field_name.to_string(), value.clone());
    }
}

fn rename_field_in_records(records: &mut [StoredRecord], old_name: &str, new_name: &str) {
    for record in records {
        let Some(value) = record.doc.fields.remove(old_name) else {
            continue;
        };

        record.doc.fields.insert(new_name.to_string(), value);
    }
}

fn remove_field_from_records(records: &mut [StoredRecord], field_name: &str) {
    for record in records {
        record.doc.fields.remove(field_name);
    }
}
