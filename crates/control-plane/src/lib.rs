//! Control plane scaffold: snapshot registry with versioned Config.
//!
//! Thin waist control plane keeps policy Config and Snapshots durable
//! without owning the data path. Gateway `SaveToStore` pushes snapshots;
//! dashboard pulls via this registry. Future: S3/Postgres backend.

pub mod server;

use std::collections::BTreeMap;
use std::sync::Mutex;
use thompson_sampling::policy::Snapshot;

/// In-memory registry — replace with S3/Postgres in production.
pub struct Registry {
    snapshots: Mutex<BTreeMap<String, Snapshot>>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        Self {
            snapshots: Mutex::new(BTreeMap::new()),
        }
    }

    /// Save snapshot for tenant `key`.
    pub fn put(&self, key: String, snapshot: Snapshot) {
        self.snapshots.lock().unwrap().insert(key, snapshot);
    }

    /// Load snapshot for tenant `key`.
    pub fn get(&self, key: &str) -> Option<Snapshot> {
        self.snapshots.lock().unwrap().get(key).cloned()
    }

    /// List tenant keys.
    pub fn list(&self) -> Vec<String> {
        self.snapshots.lock().unwrap().keys().cloned().collect()
    }

    /// JSON dump for dashboard.
    pub fn to_json(&self) -> String {
        let map = self.snapshots.lock().unwrap();
        serde_json::to_string_pretty(&*map).unwrap_or_else(|_| "{}".to_string())
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
