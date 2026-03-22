use crate::{Status, StatusCode};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CollectionName(String);

impl CollectionName {
    pub fn parse(value: impl Into<String>) -> Result<Self, Status> {
        let value = value.into();

        if value.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "collection name cannot be empty",
            ));
        }

        let valid = value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-');

        if !valid {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "collection name may only use letters, numbers, '_' and '-'",
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CollectionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Borrow<str> for CollectionName {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FieldName(String);

impl FieldName {
    pub fn parse(value: impl Into<String>) -> Result<Self, Status> {
        let value = value.into();

        if value.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name cannot be empty",
            ));
        }

        let mut chars = value.chars();
        let Some(first) = chars.next() else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name cannot be empty",
            ));
        };

        if !first.is_ascii_alphabetic() && first != '_' {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name must start with a letter or '_'",
            ));
        }

        let valid = chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
        if !valid {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name may only use letters, numbers, and '_'",
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FieldName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Borrow<str> for FieldName {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DocId(String);

impl DocId {
    pub fn parse(value: impl Into<String>) -> Result<Self, Status> {
        let value = value.into();

        if value.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "document id cannot be empty",
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Borrow<str> for DocId {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct InternalDocId(u64);

impl InternalDocId {
    pub const fn new_unchecked(value: u64) -> Self {
        Self(value)
    }

    pub fn new(value: u64) -> Result<Self, Status> {
        if value == 0 {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "internal doc id must be greater than zero",
            ));
        }

        Ok(Self(value))
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SegmentId(u64);

impl SegmentId {
    pub const fn new_unchecked(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ManifestVersionId(u64);

impl ManifestVersionId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SnapshotId(u64);

impl SnapshotId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StorageFormatVersion(u16);

impl StorageFormatVersion {
    pub fn new(value: u16) -> Self {
        Self(value)
    }

    pub fn get(self) -> u16 {
        self.0
    }
}
