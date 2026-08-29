//! Trace replay scenario for thin-waist validation.
//!
//! Reads JSONL produced by an instrumented gateway:
//! `{"id":"openai/gpt-4","t":0,"reward":0.82}` or full `Outcome` fields.
//! Regret remains exact when `reward` is provided; when replaying `Outcome`
//! the same `RewardPolicy` used in production must be supplied.

use crate::env::{ArmSpec, RewardKind, Scenario};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// One trace record.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TraceRecord {
    /// Arm id, e.g. `openai/gpt-4`.
    pub id: String,
    /// Optional round. If absent, file order is used.
    pub t: Option<usize>,
    /// Pre-collapsed reward in `[0,1]` if available.
    pub reward: Option<f64>,
    /// Raw outcome fields — alternative to `reward`.
    pub latency_ms: Option<f64>,
    pub success: Option<bool>,
    pub cost_usd: Option<f64>,
    pub quality: Option<f64>,
}

/// Load trace records from JSONL.
pub fn load_trace(path: &Path) -> Result<Vec<TraceRecord>, String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in data.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let rec: TraceRecord = serde_json::from_str(line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), idx + 1))?;
        out.push(rec);
    }
    Ok(out)
}

/// Build a synthetic `Scenario` from trace arm ids, assigning horizon and Bernoulli rewards
/// derived from empirical means (for harness baseline). For exact replay, use `TraceReplay`.
pub fn scenario_from_trace(records: &[TraceRecord], name: &'static str) -> Scenario {
    let mut means: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let mut order: BTreeSet<String> = BTreeSet::new();
    for r in records {
        let e = means.entry(r.id.clone()).or_insert((0.0, 0));
        if let Some(v) = r.reward {
            e.0 += v;
            e.1 += 1;
        }
        order.insert(r.id.clone());
    }
    let arms: Vec<ArmSpec> = order
        .into_iter()
        .map(|id| {
            let (sum, cnt) = means.get(&id).cloned().unwrap_or((0.0, 0));
            let p = if cnt > 0 { sum / cnt as f64 } else { 0.5 };
            ArmSpec::fixed(&leak(id), p.clamp(0.0, 1.0))
        })
        .collect();
    Scenario {
        name,
        description: "Replay-derived baseline (means from trace rewards)",
        arms,
        horizon: records.len().max(1),
        reward_kind: RewardKind::Bernoulli,
    }
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// Exact replay harness: draws deterministically from the trace sequence.
/// Useful for `experiment::run` replacement when validating a policy on real data.
pub struct TraceReplay {
    /// Records indexed by `(t, id)` for O(1) draw.
    by_round: Vec<BTreeMap<String, f64>>,
    scenario: Scenario,
}

impl TraceReplay {
    /// Build from records. Records must be ordered; gaps are filled with 0.0.
    pub fn new(mut records: Vec<TraceRecord>) -> Result<Self, String> {
        if records.is_empty() {
            return Err("empty trace".to_string());
        }
        // Assign t if missing.
        for (i, r) in records.iter_mut().enumerate() {
            if r.t.is_none() {
                r.t = Some(i);
            }
            if !(0.0..=1.0).contains(&r.reward.unwrap_or(0.5)) {
                // Clamp later, but warn via error if clearly invalid
                if let Some(v) = r.reward {
                    if !v.is_finite() {
                        return Err(format!("non-finite reward at t={:?} id={}", r.t, r.id));
                    }
                }
            }
        }
        let max_t = records.iter().map(|r| r.t.unwrap()).max().unwrap();
        let mut by_round: Vec<BTreeMap<String, f64>> = vec![BTreeMap::new(); max_t + 1];
        let mut ids = BTreeSet::new();
        for r in &records {
            let v = r.reward.unwrap_or(0.5).clamp(0.0, 1.0);
            by_round[r.t.unwrap()].insert(r.id.clone(), v);
            ids.insert(r.id.clone());
        }
        let arms: Vec<ArmSpec> = ids
            .into_iter()
            .map(|id| ArmSpec::fixed(&leak(id), 0.5))
            .collect();
        let horizon = by_round.len();
        let scenario = Scenario {
            name: "trace",
            description: "Exact trace replay — reward is file, not draw",
            arms,
            horizon,
            reward_kind: RewardKind::Bernoulli,
        };
        Ok(Self { by_round, scenario })
    }

    pub fn scenario(&self) -> &Scenario {
        &self.scenario
    }

    pub fn draw(&self, id: &str, t: usize) -> f64 {
        self.by_round
            .get(t)
            .and_then(|m| m.get(id).copied())
            .unwrap_or(0.0)
    }

    pub fn horizon(&self) -> usize {
        self.by_round.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn trace_replay_round_trip() {
        let records = vec![
            TraceRecord { id: "openai/gpt-4".to_string(), t: Some(0), reward: Some(0.9), latency_ms: None, success: None, cost_usd: None, quality: None },
            TraceRecord { id: "meta/llama-3".to_string(), t: Some(0), reward: Some(0.2), latency_ms: None, success: None, cost_usd: None, quality: None },
            TraceRecord { id: "openai/gpt-4".to_string(), t: Some(1), reward: Some(0.8), latency_ms: None, success: None, cost_usd: None, quality: None },
        ];
        let replay = TraceReplay::new(records).unwrap();
        assert_eq!(replay.horizon(), 2);
        assert!((replay.draw("openai/gpt-4", 0) - 0.9).abs() < 1e-9);
        assert!((replay.draw("openai/gpt-4", 1) - 0.8).abs() < 1e-9);
    }

    #[test]
    fn load_trace_jsonl() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("trace-{}.jsonl", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"id":"a","reward":0.7}}"#).unwrap();
        writeln!(f, r#"{{"id":"b","reward":0.3}}"#).unwrap();
        let recs = load_trace(&path).unwrap();
        assert_eq!(recs.len(), 2);
        let _ = std::fs::remove_file(&path);
    }
}
