use crate::state::IvfEntryIndex;
use crate::{IvfBuildEntry, IvfCentroids};
use garuda_math::score_doc;
use garuda_types::{DenseVector, DistanceMetric, VectorDimension};

impl IvfCentroids {
    pub(super) fn initialize(
        metric: DistanceMetric,
        entries: &[IvfBuildEntry],
        list_count: usize,
    ) -> Self {
        let mut centroid_indexes = Vec::with_capacity(list_count);
        centroid_indexes.push(Self::initial_centroid_index(metric, entries));

        while centroid_indexes.len() < list_count {
            let next = Self::next_centroid_index(metric, entries, &centroid_indexes);
            centroid_indexes.push(next);
        }

        let mut centroids = Vec::with_capacity(list_count);

        for index in centroid_indexes {
            centroids.push(entries[index].vector.clone());
        }

        Self::new(centroids)
    }

    pub(super) fn assign_entries(
        &self,
        metric: DistanceMetric,
        entries: &[IvfBuildEntry],
    ) -> Vec<usize> {
        let mut assignments = Vec::with_capacity(entries.len());

        for entry in entries {
            assignments.push(nearest_centroid_index(metric, &entry.vector, self.iter()));
        }

        assignments
    }

    pub(super) fn recompute(
        &self,
        dimension: VectorDimension,
        entries: &[IvfBuildEntry],
        assignments: &[usize],
    ) -> Self {
        let list_count = self.len();
        let dimension = dimension.get();
        let mut sums = vec![vec![0.0; dimension]; list_count];
        let mut counts = vec![0usize; list_count];

        for (entry, &list_index) in entries.iter().zip(assignments) {
            counts[list_index] += 1;
            add_to_sums(&mut sums[list_index], entry.vector.as_slice());
        }

        let mut centroids = Vec::with_capacity(list_count);

        for list_index in 0..list_count {
            if counts[list_index] == 0 {
                centroids.push(entries[list_index].vector.clone());
                continue;
            }

            centroids.push(centroid_from_sums(
                &mut sums[list_index],
                counts[list_index],
            ));
        }

        Self::new(centroids)
    }

    pub(super) fn into_vec(self) -> Vec<DenseVector> {
        self.0
    }

    fn initial_centroid_index(metric: DistanceMetric, entries: &[IvfBuildEntry]) -> usize {
        let mut sums = vec![0.0; entries[0].vector.len()];

        for entry in entries {
            add_to_sums(&mut sums, entry.vector.as_slice());
        }

        let mean = centroid_from_sums(&mut sums, entries.len());
        let mut best_index = 0usize;
        let mut best_score = score_doc(metric, entries[0].vector.as_slice(), mean.as_slice());

        for (entry_index, entry) in entries.iter().enumerate().skip(1) {
            let score = score_doc(metric, entry.vector.as_slice(), mean.as_slice());
            if score < best_score {
                best_index = entry_index;
                best_score = score;
            }
        }

        best_index
    }

    fn next_centroid_index(
        metric: DistanceMetric,
        entries: &[IvfBuildEntry],
        centroid_indexes: &[usize],
    ) -> usize {
        let mut next_index = 0usize;
        let mut next_score = f32::INFINITY;

        for (entry_index, entry) in entries.iter().enumerate() {
            if centroid_indexes.contains(&entry_index) {
                continue;
            }

            let nearest_score = centroid_indexes
                .iter()
                .map(|&centroid_index| {
                    score_doc(
                        metric,
                        entry.vector.as_slice(),
                        entries[centroid_index].vector.as_slice(),
                    )
                })
                .fold(f32::NEG_INFINITY, f32::max);

            if nearest_score < next_score {
                next_index = entry_index;
                next_score = nearest_score;
            }
        }

        next_index
    }
}

pub(super) fn nearest_centroid_index<'a>(
    metric: DistanceMetric,
    vector: &DenseVector,
    centroids: impl IntoIterator<Item = &'a DenseVector>,
) -> usize {
    let mut centroids = centroids.into_iter().enumerate();
    let (mut best_index, best_centroid) = centroids.next().expect("ivf centroids");
    let mut best_score = score_doc(metric, vector.as_slice(), best_centroid.as_slice());

    for (index, centroid) in centroids {
        let score = score_doc(metric, vector.as_slice(), centroid.as_slice());
        if score > best_score {
            best_index = index;
            best_score = score;
        }
    }

    best_index
}

pub(super) fn centroid_for_list(
    dimension: VectorDimension,
    entries: &[IvfBuildEntry],
    entry_indexes: &[IvfEntryIndex],
) -> DenseVector {
    let mut sums = vec![0.0; dimension.get()];

    for &entry_index in entry_indexes {
        add_to_sums(&mut sums, entries[entry_index.get()].vector.as_slice());
    }

    centroid_from_sums(&mut sums, entry_indexes.len())
}

fn add_to_sums(sums: &mut [f32], values: &[f32]) {
    sums.iter_mut()
        .zip(values)
        .for_each(|(sum, value)| *sum += *value);
}

fn centroid_from_sums(sums: &mut [f32], count: usize) -> DenseVector {
    let scale = count as f32;

    for value in sums.iter_mut() {
        *value /= scale;
    }

    DenseVector::parse(sums.to_vec()).expect("ivf centroid vector")
}

pub(super) fn assert_entries_match_dimension(
    dimension: VectorDimension,
    entries: &[IvfBuildEntry],
) {
    for entry in entries {
        assert_eq!(
            entry.vector.len(),
            dimension.get(),
            "ivf index entry dimension"
        );
    }
}
