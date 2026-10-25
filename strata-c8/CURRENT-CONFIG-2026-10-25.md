# Saved configuration observed October 25, 2026

This is a sanitized snapshot of the saved source configuration, read on
October 25. It supplements the [October 22 qualified recipe](README.md);
it does not replace its benchmark evidence or establish a new speed result.

Use [strata-server-2026-10-25.example.json](config/strata-server-2026-10-25.example.json)
with the same [parent environment](config/service-environment.example.conf),
[runtime patch and asset dependencies](DEPENDENCIES.md). Paths are operator
examples under `/srv/qwen`; binding is loopback, the model alias describes the
custom variant, credentials are empty and compatibility aliases are removed.
Set an API key and obtain the exact assets before any deployment. An empty key
does not provide authentication.

| Setting | October 22 benchmark example | October 25 saved configuration |
| --- | --- | --- |
| Prefill cap | 16,384 | 32,768 |
| Output-budget fitting | Omitted | `fit_max_tokens=true` |
| Engine / Python API runtime | 0.1.46 / patched 0.1.42 | Unchanged |
| Resident sessions / context | 8 / 131,072 | Unchanged |
| KV / expert-cache policy | Full INT8 KV / auto cache, require all 24,576 experts | Unchanged |
| Child environment / control-vector settings | As documented in the recipe | Unchanged |

After path, credential, binding, alias and log-location sanitization, the two
settings above are the only content differences from the earlier example.
`fit_max_tokens` clamps a requested output budget to remaining context room.
It does not truncate oversized prompts or enlarge the context allocation.
No throughput gain is attributed to either change.

## Observed health and memory constraint

A read-only service check on October 25 returned `activating/start-post`
and 14 automatic restarts. The saved configuration and a healthy running
process are therefore different claims: the service had not passed its full
readiness check.

Earlier checks that day showed the auto expert cache holding approximately
22,820–22,845 slots, below the required 24,576. The readiness failure was
`Engine capacity mismatch: expert_slots`. Compared with a known-good startup
using the same arguments, available GPU memory was about 3.5 GiB lower;
concurrent desktop GPU clients accounted for approximately that difference.
These are local observations, not a universal memory requirement or a
guarantee that closing a particular application resolves every startup failure.

The example retains `--expert-cache auto` and the full-residency requirement.
Confirm adequate headroom, use the deployment's owned maintenance/recovery
procedure, and verify the actual process, eight slots, context/KV policy and
all 24,576 experts before declaring it healthy. A responsive HTTP endpoint
alone did not satisfy that deployment's readiness contract.

Publishing these files did not restart the service, close desktop applications
or run inference. The configuration is preserved as observed; recovery remains
unqualified in this snapshot.

## Strata Lens capture experiment

[Strata Lens](../research/strata-lens/README.md) uses existing Strata 0.1.46
native-prefill hooks to capture final residual rows and expert routes. Its
offline smoke changes context/prefill limits to 1,024 and requests one generated
token; those diagnostic settings are separate from this serving configuration.
The capture code has CPU/fake-engine validation but no completed live GPU
capture. It does not establish per-layer residual ladders, causal expert
functions or a throughput result.

Attribution for the saved model/vector/runtime is in
[DEPENDENCIES.md](DEPENDENCIES.md); the capture method and downstream work
are credited in [CREDITS.md](../research/strata-lens/CREDITS.md).
