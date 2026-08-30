//! Minimal snapshot HTTP service for dashboard.
//! `cargo run -p control-plane` serves `/snapshots` and `/snapshots/:key`.

use crate::Registry;
use std::sync::Arc;

pub fn json_response(registry: &Arc<Registry>) -> String {
    registry.to_json()
}

#[cfg(test)]
mod tests {
    use super::*;
    use thompson_sampling::ThompsonSampling;

    #[test]
    fn server_json_round_trip() {
        let reg = Arc::new(Registry::new());
        let policy = ThompsonSampling::with_defaults(["a"]);
        reg.put("t1".to_string(), policy.snapshot());
        let json = json_response(&reg);
        assert!(json.contains("t1"));
    }
}
