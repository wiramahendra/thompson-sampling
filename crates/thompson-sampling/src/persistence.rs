//! First-class persistence for the bandit policy.
//!
//! Snapshots are serialisable, but *where* they go — file, database, object
//! store — is an engineering decision that should not live in the policy
//! itself. This module makes the store a trait so the same policy can run
//! in-process (memory), in a sidecar (file), or in a control plane (remote)
//! without changing `ThompsonSampling`.
//!
//! The trait is intentionally synchronous and fallible: persistence on the
//! request path should be explicit, not hidden behind an async runtime.

use crate::error::{Error, Result};
use crate::policy::Snapshot;

/// Durable store for policy snapshots.
pub trait SnapshotStore: Send + Sync + std::fmt::Debug {
    /// Persist a snapshot. Implementations should be atomic where possible.
    fn save(&self, snapshot: &Snapshot) -> Result<()>;

    /// Load the latest snapshot, if any.
    fn load(&self) -> Result<Option<Snapshot>>;
}

/// In-memory store — useful for tests and for embedding the harness.
#[derive(Debug, Default)]
pub struct MemoryStore {
    inner: std::sync::Mutex<Option<String>>,
}

impl MemoryStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SnapshotStore for MemoryStore {
    fn save(&self, snapshot: &Snapshot) -> Result<()> {
        let json = snapshot.to_json()?;
        *self.inner.lock().unwrap() = Some(json);
        Ok(())
    }

    fn load(&self) -> Result<Option<Snapshot>> {
        let guard = self.inner.lock().unwrap();
        match &*guard {
            None => Ok(None),
            Some(json) => Ok(Some(Snapshot::from_json(json)?)),
        }
    }
}

/// File-backed store — atomic via write-to-temp-then-rename.
#[derive(Debug, Clone)]
pub struct FileStore {
    path: std::path::PathBuf,
}

impl FileStore {
    /// Create a store backed by `path`.
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        FileStore { path: path.into() }
    }

    /// Path being used.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl SnapshotStore for FileStore {
    fn save(&self, snapshot: &Snapshot) -> Result<()> {
        let json = snapshot.to_json()?;
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, json.as_bytes())
            .map_err(|e| Error::Decode(format!("write {}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, &self.path)
            .map_err(|e| Error::Decode(format!("rename {}: {e}", tmp.display())))?;
        Ok(())
    }

    fn load(&self) -> Result<Option<Snapshot>> {
        match std::fs::read_to_string(&self.path) {
            Ok(json) => Ok(Some(Snapshot::from_json(&json)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::Decode(format!("read {}: {e}", self.path.display()))),
        }
    }
}

impl SnapshotStore for Box<dyn SnapshotStore> {
    fn save(&self, snapshot: &Snapshot) -> Result<()> {
        (**self).save(snapshot)
    }
    fn load(&self) -> Result<Option<Snapshot>> {
        (**self).load()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ThompsonSampling;

    #[test]
    fn memory_store_round_trips() {
        let mut policy = ThompsonSampling::with_defaults(["a", "b"]);
        // need a pull so snapshot is non-trivial
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        use rand::SeedableRng;
        let id = policy.select(&mut rng).unwrap();
        policy.record(&mut rng, &id, 0.7).unwrap();

        let snapshot = policy.snapshot();
        let store = MemoryStore::new();
        assert!(store.load().unwrap().is_none());
        store.save(&snapshot).unwrap();
        let restored = store.load().unwrap().unwrap();
        assert_eq!(restored.total_pulls, snapshot.total_pulls);
        assert_eq!(restored.arms.len(), snapshot.arms.len());
    }

    #[test]
    fn file_store_round_trips_via_tmpfile() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("thompson-test-{}.json", std::process::id()));
        let store = FileStore::new(&path);
        let _ = std::fs::remove_file(&path);

        let policy = ThompsonSampling::with_defaults(["a"]);
        let snapshot = policy.snapshot();
        store.save(&snapshot).unwrap();
        let loaded = store.load().unwrap().unwrap();
        assert_eq!(loaded.version, snapshot.version);
        let _ = std::fs::remove_file(&path);
    }
}
