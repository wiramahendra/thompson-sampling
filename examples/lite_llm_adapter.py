"""
Thin-waist LiteLLM/Portkey adapter — 20 lines to add to any gateway.

Integrates Thompson Sampling via the 2-call waist:
    provider = policy.select()
    outcome = Outcome(latency_ms, success, cost_usd).with_quality(...)
    policy.record_outcome(provider, outcome)

This example uses the Rust core via FFI or Go sidecar HTTP. For demo,
it calls the Go gateway via localhost:8080; replace with direct library call.

Install: pip install litellm
Patch: add `thompson_select` before `litellm.completion`
"""

import time
import random

# Mock policy — replace with actual FFI: `from thompson import Policy`
class MockPolicy:
    def __init__(self, arms):
        self.arms = arms
        self.counts = {a: 0 for a in arms}

    def select(self):
        # In prod: call Rust `policy.select()` or Go `policy.Select(rng)`
        # Thompson Sampling draw — mock as uniform for demo
        choice = random.choice(self.arms)
        print(f"[thompson] select -> {choice}")
        return choice

    def record_outcome(self, provider, latency_ms, success, cost_usd, quality=None):
        # In prod: `Outcome::new(latency_ms, success, cost_usd).with_quality(quality)`
        # `policy.record_outcome(provider, outcome)`
        reward = 1.0 if success else 0.0  # simplified; real uses RewardPolicy
        print(f"[thompson] record {provider} latency={latency_ms:.0f}ms success={success} reward={reward}")
        self.counts[provider] += 1


policy = MockPolicy(["openai/gpt-4", "anthropic/claude-3-opus", "meta/llama-3"])

def litellm_completion_with_thompson(prompt: str):
    provider = policy.select()
    start = time.time()
    try:
        # Original LiteLLM call:
        # response = litellm.completion(model=provider, messages=[{"role": "user", "content": prompt}])
        # Mock latency/cost
        time.sleep(0.05)
        success = True
        cost_usd = 0.0012
        latency_ms = (time.time() - start) * 1000
        quality = 0.87  # from judge
        policy.record_outcome(provider, latency_ms, success, cost_usd, quality)
        return f"[{provider}] response to: {prompt}"
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        policy.record_outcome(provider, latency_ms, False, 0.0, None)
        raise e


if __name__ == "__main__":
    for i in range(5):
        print(litellm_completion_with_thompson(f"hello {i}"))
    print("counts:", policy.counts)
    print("\n# To integrate for real:")
    print("# 1. cargo build -p thompson-sampling --release (FFI) or run go/gateway/main.go :8080")
    print("# 2. Replace MockPolicy.select/record_outcome with FFI calls")
    print("# 3. Attach OtelObserver via PolicyObserver for Prometheus")
