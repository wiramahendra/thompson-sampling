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
/// Stores `Snapshot` directly (no JSON round-trip) to avoid double serialize
/// and ULP drift noted in `policy.rs:535`.
#[derive(Debug, Default)]
pub struct MemoryStore {
    inner: std::sync::RwLock<Option<Snapshot>>,
}

impl MemoryStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SnapshotStore for MemoryStore {
    fn save(&self, snapshot: &Snapshot) -> Result<()> {
        let mut w = self.inner.write().unwrap_or_else(|e| e.into_inner());
        *w = Some(snapshot.clone());
        Ok(())
    }

    fn load(&self) -> Result<Option<Snapshot>> {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        Ok(guard.clone())
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
        // Use pid + random suffix to avoid collision when multiple processes
        // save concurrently to the same path (with_extension("tmp") would collide).
        let tmp = self
            .path
            .with_extension(format!("tmp.{}", std::process::id()));
        // Write to tmp, fsync, then rename atomically.
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)
                .map_err(|e| Error::Decode(format!("create {}: {e}", tmp.display())))?;
            f.write_all(json.as_bytes())
                .map_err(|e| Error::Decode(format!("write {}: {e}", tmp.display())))?;
            f.sync_all()
                .map_err(|e| Error::Decode(format!("fsync {}: {e}", tmp.display())))?;
        }
        // Enforce size limit on read path — avoid unbounded allocation on untrusted snapshot.
        let meta = std::fs::metadata(&tmp)
            .map_err(|e| Error::Decode(format!("stat {}: {e}", tmp.display())))?;
        if meta.len() > 10 * 1024 * 1024 {
            let _ = std::fs::remove_file(&tmp);
            return Err(Error::Decode("snapshot too large (>10MiB)".to_string()));
        }
        std::fs::rename(&tmp, &self.path)
            .map_err(|e| Error::Decode(format!("rename {}: {e}", tmp.display())))?;
        // Fsync parent dir for durability on crash (best-effort).
        if let Some(parent) = self.path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }

    fn load(&self) -> Result<Option<Snapshot>> {
        let json = match std::fs::read_to_string(&self.path) {
            Ok(j) => j,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(Error::Decode(format!("read {}: {e}", self.path.display()))),
        };
        if json.len() > 10 * 1024 * 1024 {
            return Err(Error::Decode("snapshot too large (>10MiB)".to_string()));
        }
        Ok(Some(Snapshot::from_json(&json)?))
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
