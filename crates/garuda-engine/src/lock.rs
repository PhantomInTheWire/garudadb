use crate::storage::LOCK_FILE_NAME;
use garuda_types::{Status, StatusCode};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

#[derive(Debug)]
pub(crate) struct CollectionLock {
    collection_path: PathBuf,
    file: File,
}

impl CollectionLock {
    pub(crate) fn acquire(collection_path: impl AsRef<Path>) -> Result<Self, Status> {
        let collection_path = collection_path.as_ref().to_path_buf();
        let lock_path = collection_path.join(LOCK_FILE_NAME);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|error| {
                Status::err(
                    StatusCode::Internal,
                    format!("failed to open collection lock file: {error}"),
                )
            })?;

        register_process_lock(&collection_path)?;

        if let Err(status) = lock_file(&file) {
            unregister_process_lock(&collection_path);
            return Err(status);
        }

        Ok(Self {
            collection_path,
            file,
        })
    }
}

impl Drop for CollectionLock {
    fn drop(&mut self) {
        let _ = unlock_file(&self.file);
        unregister_process_lock(&self.collection_path);
    }
}

fn register_process_lock(collection_path: &Path) -> Result<(), Status> {
    let mut held_paths = held_lock_paths().lock().expect("lock registry poisoned");

    if held_paths.insert(collection_path.to_path_buf()) {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::FailedPrecondition,
        "collection is already open in this process",
    ))
}

fn unregister_process_lock(collection_path: &Path) {
    let mut held_paths = held_lock_paths().lock().expect("lock registry poisoned");
    held_paths.remove(collection_path);
}

fn held_lock_paths() -> &'static Mutex<HashSet<PathBuf>> {
    static HELD_LOCKS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    HELD_LOCKS.get_or_init(|| Mutex::new(HashSet::new()))
}

fn lock_file(file: &File) -> Result<(), Status> {
    let fd = file.as_raw_fd();
    let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };

    if result == 0 {
        return Ok(());
    }

    let error = std::io::Error::last_os_error();
    Err(Status::err(
        StatusCode::FailedPrecondition,
        format!("collection is already locked: {error}"),
    ))
}

fn unlock_file(file: &File) -> Result<(), Status> {
    let fd = file.as_raw_fd();
    let result = unsafe { libc::flock(fd, libc::LOCK_UN) };

    if result == 0 {
        return Ok(());
    }

    let error = std::io::Error::last_os_error();
    Err(Status::err(
        StatusCode::Internal,
        format!("failed to unlock collection: {error}"),
    ))
}
