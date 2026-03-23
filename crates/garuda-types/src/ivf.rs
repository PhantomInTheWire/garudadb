use crate::{Status, StatusCode};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

const IVF_MIN_LIST_COUNT: u32 = 1;
const IVF_MIN_PROBE_COUNT: u32 = 1;
const IVF_MIN_TRAINING_ITERATIONS: u32 = 1;
const IVF_DEFAULT_LIST_COUNT: u32 = 64;
const IVF_DEFAULT_PROBE_COUNT: u32 = 8;
const IVF_DEFAULT_TRAINING_ITERATIONS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IvfListCount(NonZeroU32);

impl IvfListCount {
    pub fn new(value: u32) -> Result<Self, Status> {
        let Some(value) = NonZeroU32::new(value) else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "ivf n_list must be greater than zero",
            ));
        };

        assert_eq!(IVF_MIN_LIST_COUNT, 1, "ivf minimum list count");
        Ok(Self(value))
    }

    pub fn from_persisted_u64(value: u64) -> Result<Self, Status> {
        let value = u32::try_from(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "ivf n_list exceeds u32::MAX in persisted manifest",
            )
        })?;

        Self::new(value)
            .map_err(|_| Status::err(StatusCode::Internal, "ivf n_list must be greater than zero"))
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IvfProbeCount(NonZeroU32);

impl IvfProbeCount {
    pub fn new(value: u32) -> Result<Self, Status> {
        let Some(value) = NonZeroU32::new(value) else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "ivf nprobe must be greater than zero",
            ));
        };

        assert_eq!(IVF_MIN_PROBE_COUNT, 1, "ivf minimum probe count");
        Ok(Self(value))
    }

    pub fn from_persisted_u64(value: u64) -> Result<Self, Status> {
        let value = u32::try_from(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "ivf nprobe exceeds u32::MAX in persisted manifest",
            )
        })?;

        Self::new(value)
            .map_err(|_| Status::err(StatusCode::Internal, "ivf nprobe must be greater than zero"))
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IvfTrainingIterations(NonZeroU32);

impl IvfTrainingIterations {
    pub fn new(value: u32) -> Result<Self, Status> {
        let Some(value) = NonZeroU32::new(value) else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "ivf training_iterations must be greater than zero",
            ));
        };

        assert_eq!(
            IVF_MIN_TRAINING_ITERATIONS, 1,
            "ivf minimum training iterations"
        );
        Ok(Self(value))
    }

    pub fn from_persisted_u64(value: u64) -> Result<Self, Status> {
        let value = u32::try_from(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "ivf training_iterations exceeds u32::MAX in persisted manifest",
            )
        })?;

        Self::new(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "ivf training_iterations must be greater than zero",
            )
        })
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IvfIndexParams {
    pub n_list: IvfListCount,
    pub n_probe: IvfProbeCount,
    pub training_iterations: IvfTrainingIterations,
}

impl Default for IvfIndexParams {
    fn default() -> Self {
        Self {
            n_list: IvfListCount::new(IVF_DEFAULT_LIST_COUNT)
                .expect("default ivf n_list should be valid"),
            n_probe: IvfProbeCount::new(IVF_DEFAULT_PROBE_COUNT)
                .expect("default ivf nprobe should be valid"),
            training_iterations: IvfTrainingIterations::new(IVF_DEFAULT_TRAINING_ITERATIONS)
                .expect("default ivf training_iterations should be valid"),
        }
    }
}
