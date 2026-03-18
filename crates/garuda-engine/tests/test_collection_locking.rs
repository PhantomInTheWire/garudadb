mod common;

use common::{collection_name, database, default_options, default_schema};
use garuda_types::StatusCode;

#[test]
fn second_open_of_same_collection_fails_while_first_handle_is_alive() {
    let (_root, db) = database("locking-contention");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let result = db.open_collection(&collection_name("docs"));
    match result {
        Ok(_) => panic!("second open should fail while lock is held"),
        Err(status) => assert_eq!(status.code, StatusCode::FailedPrecondition),
    }

    drop(collection);
}

#[test]
fn lock_is_released_after_collection_is_dropped() {
    let (_root, db) = database("locking-release");

    {
        let _collection = db
            .create_collection(default_schema("docs"), default_options())
            .expect("create collection");
    }

    let reopened = db.open_collection(&collection_name("docs"));
    assert!(reopened.is_ok());
}

#[test]
fn cloned_collection_keeps_lock_alive_until_last_handle_is_dropped() {
    let (_root, db) = database("locking-clone");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    let cloned = collection.clone();

    drop(collection);

    let still_locked = db.open_collection(&collection_name("docs"));
    match still_locked {
        Ok(_) => panic!("cloned collection should keep the lock alive"),
        Err(status) => assert_eq!(status.code, StatusCode::FailedPrecondition),
    }

    drop(cloned);

    let reopened = db.open_collection(&collection_name("docs"));
    assert!(reopened.is_ok());
}
