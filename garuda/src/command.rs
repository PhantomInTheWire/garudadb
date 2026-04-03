use garuda_engine::{Collection, Database};
use garuda_types::{
    CollectionOptions, CollectionSchema, DistanceMetric, Nullability, OptimizeOptions,
    ScalarFieldSchema, TopK, VectorDimension, VectorFieldSchema, VectorIndexState, VectorQuery,
};

use crate::cli::{ByIdQueryArgs, QueryOptions, QuerySource, RunnableCommand, VectorQueryArgs};
use crate::parsing::{
    field_name, parse_doc_id, parse_field_name, parse_index_params, parse_query_search,
    parse_scalar_json_literal, print_json, read_collection_file, read_jsonl_docs,
};

const PRIMARY_KEY_FIELD: &str = "pk";
const VECTOR_FIELD: &str = "embedding";

pub fn default_segment_max_docs() -> usize {
    CollectionOptions::default().segment_max_docs
}

pub fn run_command(db: &Database, command: RunnableCommand) -> Result<(), String> {
    match command {
        RunnableCommand::Create {
            name,
            dimension,
            metric,
            segment_max_docs,
            storage_access,
        } => {
            let options = CollectionOptions {
                segment_max_docs,
                storage_access: storage_access.into(),
                ..CollectionOptions::default()
            };
            db.create_collection(
                default_schema(
                    name.clone(),
                    VectorDimension::new(dimension.get()).expect("dimension is non-zero"),
                    metric.into(),
                ),
                options,
            )
            .map_err(|status| status.message)?;
            println!("{name}");
            Ok(())
        }
        RunnableCommand::CreateFromSchema { path } => {
            let (schema, options) = read_collection_file(&path)?;
            db.create_collection(schema, options)
                .map_err(|status| status.message)?;
            println!("ok");
            Ok(())
        }
        RunnableCommand::InsertJsonl { name, path } => {
            write_jsonl(&open_collection(db, &name)?, &path, Collection::insert)
        }
        RunnableCommand::UpsertJsonl { name, path } => {
            write_jsonl(&open_collection(db, &name)?, &path, Collection::upsert)
        }
        RunnableCommand::UpdateJsonl { name, path } => {
            write_jsonl(&open_collection(db, &name)?, &path, Collection::update)
        }
        RunnableCommand::DeleteIds { name, ids } => {
            let ids = ids
                .into_iter()
                .map(parse_doc_id)
                .collect::<Result<Vec<_>, _>>()?;
            print_json(&open_collection(db, &name)?.delete(ids))
        }
        RunnableCommand::DeleteFilter { name, filter } => {
            open_collection(db, &name)?
                .delete_by_filter(&filter)
                .map_err(|status| status.message)?;
            println!("ok");
            Ok(())
        }
        RunnableCommand::Query { name, source } => {
            let collection = open_collection(db, &name)?;
            let vector_field = collection.schema().vector.name.clone();
            let query = match source {
                QuerySource::Vector(args) => vector_query(vector_field.clone(), args)?,
                QuerySource::ById(args) => id_query(vector_field, args)?,
            };
            print_json(&collection.query(query).map_err(|status| status.message)?)
        }
        RunnableCommand::Fetch { name, id } => {
            print_json(&open_collection(db, &name)?.fetch(vec![parse_doc_id(id)?]))
        }
        RunnableCommand::CreateIndex { name, field, kind } => {
            open_collection(db, &name)?
                .create_index(&parse_field_name(field)?, parse_index_params(kind)?)
                .map_err(|status| status.message)?;
            println!("ok");
            Ok(())
        }
        RunnableCommand::DropIndex { name, field, kind } => {
            open_collection(db, &name)?
                .drop_index(&parse_field_name(field)?, kind.into())
                .map_err(|status| status.message)?;
            println!("ok");
            Ok(())
        }
        RunnableCommand::AddColumn {
            name,
            column,
            field_type,
            nullability,
            index,
            default,
        } => {
            open_collection(db, &name)?
                .add_column(ScalarFieldSchema {
                    name: parse_field_name(column)?,
                    field_type: field_type.into(),
                    index: index.into(),
                    nullability: nullability.into(),
                    default_value: default
                        .as_deref()
                        .map(parse_scalar_json_literal)
                        .transpose()?,
                })
                .map_err(|status| status.message)?;
            println!("ok");
            Ok(())
        }
        RunnableCommand::RenameColumn { name, old, new } => {
            open_collection(db, &name)?
                .alter_column(&parse_field_name(old)?, &parse_field_name(new)?)
                .map_err(|status| status.message)?;
            println!("ok");
            Ok(())
        }
        RunnableCommand::DropColumn { name, column } => {
            open_collection(db, &name)?
                .drop_column(&parse_field_name(column)?)
                .map_err(|status| status.message)?;
            println!("ok");
            Ok(())
        }
        RunnableCommand::Flush { name } => {
            open_collection(db, &name)?
                .flush()
                .map_err(|status| status.message)?;
            println!("ok");
            Ok(())
        }
        RunnableCommand::Optimize { name } => {
            open_collection(db, &name)?
                .optimize(OptimizeOptions)
                .map_err(|status| status.message)?;
            println!("ok");
            Ok(())
        }
        RunnableCommand::Schema { name } => print_json(&open_collection(db, &name)?.schema()),
        RunnableCommand::Options { name } => print_json(&open_collection(db, &name)?.options()),
        RunnableCommand::Stats { name } => print_json(&open_collection(db, &name)?.stats()),
    }
}

fn open_collection(
    db: &Database,
    name: &garuda_types::CollectionName,
) -> Result<Collection, String> {
    db.open_collection(name).map_err(|status| status.message)
}

fn write_jsonl(
    collection: &Collection,
    path: &std::path::Path,
    write: fn(&Collection, Vec<garuda_types::Doc>) -> Vec<garuda_types::WriteResult>,
) -> Result<(), String> {
    let results = write(collection, read_jsonl_docs(path)?);
    print_json(&results)?;

    let Some(result) = results.iter().find(|result| !result.status.is_ok()) else {
        return Ok(());
    };

    Err(result.status.message.clone())
}

fn vector_query(
    vector_field: garuda_types::FieldName,
    args: VectorQueryArgs,
) -> Result<VectorQuery, String> {
    let mut query = VectorQuery::by_vector(
        vector_field,
        args.value,
        TopK::new(args.options.top_k).map_err(|status| status.message)?,
    );
    apply_query_options(&mut query, args.options)?;
    Ok(query)
}

fn id_query(
    vector_field: garuda_types::FieldName,
    args: ByIdQueryArgs,
) -> Result<VectorQuery, String> {
    let mut query = VectorQuery::by_id(
        vector_field,
        args.value,
        TopK::new(args.options.top_k).map_err(|status| status.message)?,
    );
    apply_query_options(&mut query, args.options)?;
    Ok(query)
}

fn apply_query_options(query: &mut VectorQuery, options: QueryOptions) -> Result<(), String> {
    query.filter = options.filter;
    query.output_fields = options.fields;
    query.vector_projection = options.vector_projection.into();
    query.search = parse_query_search(options.search)?;
    Ok(())
}

fn default_schema(
    name: garuda_types::CollectionName,
    dimension: VectorDimension,
    metric: DistanceMetric,
) -> CollectionSchema {
    CollectionSchema {
        name,
        primary_key: field_name(PRIMARY_KEY_FIELD),
        fields: vec![
            ScalarFieldSchema {
                name: field_name(PRIMARY_KEY_FIELD),
                field_type: garuda_types::ScalarType::String,
                index: garuda_types::ScalarIndexState::None,
                nullability: Nullability::Required,
                default_value: None,
            },
            ScalarFieldSchema {
                name: field_name("rank"),
                field_type: garuda_types::ScalarType::Int64,
                index: garuda_types::ScalarIndexState::None,
                nullability: Nullability::Required,
                default_value: None,
            },
            ScalarFieldSchema {
                name: field_name("category"),
                field_type: garuda_types::ScalarType::String,
                index: garuda_types::ScalarIndexState::None,
                nullability: Nullability::Required,
                default_value: None,
            },
            ScalarFieldSchema {
                name: field_name("score"),
                field_type: garuda_types::ScalarType::Float64,
                index: garuda_types::ScalarIndexState::None,
                nullability: Nullability::Required,
                default_value: None,
            },
        ],
        vector: VectorFieldSchema {
            name: field_name(VECTOR_FIELD),
            dimension,
            metric,
            indexes: VectorIndexState::DefaultFlat,
        },
    }
}
