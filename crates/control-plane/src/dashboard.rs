//! Dashboard data aggregation for `control-plane`.
//!
//! Converts `Snapshot` + `Registry` into UI-ready JSON without owning HTTP.

use crate::Registry;
use thompson_sampling::policy::Snapshot;

/// UI row for one arm.
#[derive(Debug, serde::Serialize)]
pub struct ArmRow {
    pub id: String,
    pub alpha: f64,
    pub beta: f64,
    pub pulls: u64,
    pub mean: f64,
    pub warm_started: bool,
}

/// UI payload for one tenant.
#[derive(Debug, serde::Serialize)]
pub struct TenantDashboard {
    pub tenant: String,
    pub total_pulls: u64,
    pub arms: Vec<ArmRow>,
}

impl TenantDashboard {
    fn from_snapshot(tenant: &str, snap: &Snapshot) -> Self {
        let mut arms: Vec<ArmRow> = snap
            .arms
            .iter()
            .map(|a| ArmRow {
                id: a.id.clone(),
                alpha: a.posterior.alpha,
                beta: a.posterior.beta,
                pulls: a.posterior.pulls,
                mean: a.posterior.mean(),
                warm_started: a.warm_started,
            })
            .collect();
        arms.sort_by(|a, b| {
            b.mean
                .partial_cmp(&a.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self {
            tenant: tenant.to_string(),
            total_pulls: snap.total_pulls,
            arms,
        }
    }
}

/// Build dashboard payload for all tenants.
pub fn build_dashboard(registry: &Registry) -> Vec<TenantDashboard> {
    registry
        .list()
        .into_iter()
        .filter_map(|key| {
            let snap = registry.get(&key)?;
            Some(TenantDashboard::from_snapshot(&key, &snap))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use thompson_sampling::ThompsonSampling;

    #[test]
    fn dashboard_sorts_by_mean() {
        let reg = Registry::new();
        let mut policy = ThompsonSampling::with_defaults(["a", "b"]);
        // make a clearly better
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        use rand::SeedableRng;
        for _ in 0..20 {
            policy.record(&mut rng, "a", 0.9).unwrap();
        }
        reg.put("t".to_string(), policy.snapshot());
        let dash = build_dashboard(&reg);
        assert_eq!(dash[0].arms[0].id, "a");
    }
}
