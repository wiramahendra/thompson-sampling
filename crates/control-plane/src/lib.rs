//! Thin-waist control plane: in-memory snapshot registry.
//!
//! Keeps policy `Snapshot`s durable without owning the data path.
//! Gateway `SaveToStore` pushes snapshots; dashboard pulls via this registry.

pub mod dashboard;
pub mod server;
pub mod storage;
pub mod ui;

use std::collections::BTreeMap;
use std::sync::RwLock;
use thompson_sampling::policy::Snapshot;

/// In-memory registry — thin, no external deps.
/// `RwLock` allows concurrent dashboard reads; poison is recovered so an
/// observer panic doesn't blackhole the router. Use `storage::FileStorage`
/// if you need local durability; no S3/Postgres in this crate.
pub struct Registry {
    snapshots: RwLock<BTreeMap<String, Snapshot>>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// Create empty registry.
    pub fn new() -> Self {
        Self {
            snapshots: RwLock::new(BTreeMap::new()),
        }
    }

    fn read_map(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<String, Snapshot>> {
        self.snapshots.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write_map(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<String, Snapshot>> {
        self.snapshots.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Save snapshot for tenant `key`.
    pub fn put(&self, key: String, snapshot: Snapshot) {
        self.write_map().insert(key, snapshot);
    }

    /// Load snapshot for tenant `key`.
    pub fn get(&self, key: &str) -> Option<Snapshot> {
        self.read_map().get(key).cloned()
    }

    /// List tenant keys.
    pub fn list(&self) -> Vec<String> {
        self.read_map().keys().cloned().collect()
    }

    /// JSON dump for dashboard.
    pub fn to_json(&self) -> String {
        let map = self.read_map();
        serde_json::to_string_pretty(&*map).unwrap_or_else(|_| "{}".to_string())
    }

    /// JSON for single tenant, if present.
    pub fn to_json_for(&self, key: &str) -> Option<String> {
        let map = self.read_map();
        let snap = map.get(key)?;
        serde_json::to_string_pretty(snap).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thompson_sampling::ThompsonSampling;

    #[test]
    fn registry_put_get() {
        let reg = Registry::new();
        let policy = ThompsonSampling::with_defaults(["a"]);
        let snap = policy.snapshot();
        reg.put("tenant-1".to_string(), snap.clone());
        assert_eq!(reg.get("tenant-1").unwrap().version, snap.version);
        assert_eq!(reg.list(), vec!["tenant-1".to_string()]);
    }
}
