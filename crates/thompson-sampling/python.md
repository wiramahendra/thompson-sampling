# Python FFI (future)

Thin-waist Python binding via `pyo3` — mirrors `examples/lite_llm_adapter.py`.

Planned:

```toml
[dependencies.pyo3]
version = "0.20"
features = ["extension-module"]
```

Expose:

```python
from thompson import Policy, Outcome
policy = Policy(["openai/gpt-4"])
provider = policy.select()
policy.record_outcome(provider, Outcome(320, True, 0.0012).with_quality(0.87))
```

Status: scaffold — use `examples/lite_llm_adapter.py:1` mock until pyo3 built.
