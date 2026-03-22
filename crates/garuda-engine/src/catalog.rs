use garuda_segment::{PersistedSegment, WritingSegment};
use garuda_types::{
    CollectionOptions, CollectionSchema, InternalDocId, Manifest, ManifestVersionId, SegmentId,
    SnapshotId,
};

#[derive(Clone)]
pub(crate) struct CollectionCatalog {
    pub(crate) schema: CollectionSchema,
    pub(crate) options: CollectionOptions,
    pub(crate) next_doc_id: InternalDocId,
    pub(crate) next_segment_id: SegmentId,
    pub(crate) id_map_snapshot_id: SnapshotId,
    pub(crate) delete_snapshot_id: SnapshotId,
    pub(crate) manifest_version_id: ManifestVersionId,
}

impl CollectionCatalog {
    pub(crate) fn from_manifest(manifest: Manifest) -> Self {
        Self {
            schema: manifest.schema,
            options: manifest.options,
            next_doc_id: manifest.next_doc_id,
            next_segment_id: manifest.next_segment_id,
            id_map_snapshot_id: manifest.id_map_snapshot_id,
            delete_snapshot_id: manifest.delete_snapshot_id,
            manifest_version_id: manifest.manifest_version_id,
        }
    }

    pub(crate) fn to_manifest(
        &self,
        writing_segment: &WritingSegment,
        persisted_segments: &[PersistedSegment],
    ) -> Manifest {
        Manifest {
            schema: self.schema.clone(),
            options: self.options.clone(),
            next_doc_id: self.next_doc_id,
            next_segment_id: self.next_segment_id,
            id_map_snapshot_id: self.id_map_snapshot_id,
            delete_snapshot_id: self.delete_snapshot_id,
            manifest_version_id: self.manifest_version_id,
            writing_segment: writing_segment.meta.clone(),
            persisted_segments: persisted_segments
                .iter()
                .map(|segment| segment.meta.clone())
                .collect(),
        }
    }
}
