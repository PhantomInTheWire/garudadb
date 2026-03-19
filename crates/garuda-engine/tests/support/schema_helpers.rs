use crate::common::{default_options, default_schema, field_name};
use garuda_types::{CollectionOptions, CollectionSchema};

pub fn schema_with_dimension(name: &str, dimension: usize) -> CollectionSchema {
    let mut schema = default_schema(name);
    schema.vector.dimension = dimension;
    schema
}

pub fn schema_with_vector_name(name: &str, vector_name: &str) -> CollectionSchema {
    let mut schema = default_schema(name);
    schema.vector.name = field_name(vector_name);
    schema
}

pub fn schema_with_duplicate_field(name: &str) -> CollectionSchema {
    let mut schema = default_schema(name);
    schema.fields.push(schema.fields[0].clone());
    schema
}

pub fn schema_missing_primary_field(name: &str) -> CollectionSchema {
    let mut schema = default_schema(name);
    schema.primary_key = field_name("missing_pk");
    schema
}

pub fn read_only_options() -> CollectionOptions {
    CollectionOptions {
        read_only: true,
        ..default_options()
    }
}
