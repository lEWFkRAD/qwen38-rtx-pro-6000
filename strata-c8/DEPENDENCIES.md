# Runtime and asset dependencies

This is a sanitized recipe for the measured custom IQ3_S GSQ-RCO configuration.
It is not a turnkey installer, and the JSON alone does not supply the model,
generated pack, tokenizer, MTP data, Python environment or service management.
All `/srv/qwen` paths are operator-chosen examples. No binaries or model files
are redistributed here.

## Deliberate engine/API version split

The qualified combination retains the Python/API runtime from Strata **v0.1.42**,
with the supplied server patch, and uses the official **0.1.46** decoder and
matched vision executable/CUDA 13.0 libraries from app release **v0.2.1**.
Installing the current app or running an automatic updater is a different setup.

- Python source: [v0.1.42](https://github.com/Niko1221/Strata/tree/v0.1.42), public commit
  `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`.
- Engine release: [v0.2.1](https://github.com/Niko1221/Strata/releases/tag/v0.2.1).
- Exact Linux CUDA13 archive:
  [strata-engine-0.1.46-linux-x86_64-cuda-cuda13.tar.gz](https://github.com/Niko1221/Strata/releases/download/v0.2.1/strata-engine-0.1.46-linux-x86_64-cuda-cuda13.tar.gz).
  Size: `604965163` bytes. SHA256:
  `d9e564cf9c6007067fcc93b53bf9690d7d715fe2b96f1d33106a36690c447975`.
- Signed `engine-component.json` SHA256:
  `7ff1d2840136bc64ba1a46b35114826624881bfce05911c1e15a5d2e945393f5`.
  This hashes the entire signed manifest file; its Ed25519 signature covers
  canonical JSON excluding the `signature` field.
  An exact byte-preserved copy is included as
  [engine-component.json](evidence/engine-component.json); the repository marks
  that file binary for line-ending purposes so this hash survives checkout.
- Actual package build: `a7bdbfdabc9a+cuda13.0`, engine source commit
  `a7bdbfdabc9a3e7c83e3dbc2441d0933569d773e`. The verified manifest lists
  `sm_75`, `sm_86`, `sm_89`, `sm_120`, `compute_120`, and NVIDIA driver minimum
  `580.65`. This recipe targets SM120, not every architecture in that package.

Keep the entire verified package together: `strata`, `strata-vision`,
`libstrata_kq.so`, `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`,
`BUILD.json`, `SYSTEM-LIBS.txt` and `engine-component.json`. The config puts its
directory first in `lib_dirs`; do not mix it with an older toolkit's libraries.
ELF checks on the qualified host resolved all dependencies, including system
GLIBC/libstdc++; the executable requires GLIBC up to `2.34`. The driver remains
the host's. See [engine-release.json](evidence/engine-release.json) for the complete
verified manifest and signature provenance. The release feed may change;
verify the pinned archive and manifest rather than assuming a future feed is identical.

## Retained server patch

[runtime-0142.patch](patches/runtime-0142.patch) applies to `serve/server.py` at
the public v0.1.42 commit above. Its three changes are:

1. `STRATA_SLOT_QUEUE=1`: requests beyond the engine's slot count wait at a
   bounded semaphore with queue keepalives and cancellation handling, instead
   of entering an oversubscribed admission path.
2. `STRATA_ADMIT_ROTATE=1` with `STRATA_ADMIT_ROTATE_QUANTUM=32768`: comparable
   waiting long prompts can take turns at chunk boundaries. These variables
   must be in the **Python server's environment**, not only a child engine env.
3. Optional `compatibility_api_keys`: accepts only operator-configured aliases.
   The example ships no aliases or credentials.

The supplied patch replaces one deployment-specific comment with a generic
comment; executable Python code is unchanged. Its expected postimage SHA256 is
`2e2a550736db9fba698c341aeb9267e9788f9768c24cec7b8b2d3b80135181a2`.
The original public base file SHA256 is
`ce9266730b632e919ff93809ea6f9ecd6f91f12da19597a2d95a7edb29700dfa`.
Local comparison found the other 160 pinned Python/template files identical to
that public commit. No runtime tree is copied into this recipe. Preserve the
[Strata MIT notice](LICENSES/Strata-MIT.txt) when distributing the patch.

A historical local recovery field named a different source commit that could
not be resolved publicly. The independently checked v0.1.42 file hashes and patch
above provide the reproduction basis instead. See
[runtime-provenance.json](evidence/runtime-provenance.json). A complete Python
package lock/environment export is not included; obtain the pinned source's
dependencies and qualify the resulting environment separately. The pinned
source records llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`
for its related tokenizer/vision/build tooling.

## The model is a served custom variant

The measured weights are **IQ3_S GSQ-RCO**, combined with the separate
`Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf` control vector. The vector is
applied at **strength 1.0, layers 4-44, `project`, `per-layer`**. This changes
model behavior, including refusal behavior; it is not a numerical kernel speed
optimization or a vanilla-Qwen configuration. The stored sampling default and
benchmarks use `experimental_speed_projection=true`, which enables the loaded
vector; a request setting it to `false` changes the served variant for that request.

- Model package:
  [alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF),
  pinned revision `d4af3be5701549435e0770d625f79dd2def1358f`.
- The retained model card attributes the unchanged quantized weights to
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
- Both GGUF shards, the projector and control vector need to match
  [model-assets.json](evidence/model-assets.json). Their hashes come from the
  retained verified-download record and publisher manifest; this documentation
  update did not rehash the large weight files.

The first GGUF shard supplies the native model; the second is supplied to
`--ple-gguf` for the PLE/n-gram table. The example keeps `--ple-io ram`.
Prepare the matching Strata pack, tokenizer and MTP runtime data, and keep the
expert profile from the pinned source. Their generated pack/MTP artifact hashes
are not provided here: reproducing the exact generated artifacts remains a
dependency, and substituting another quantization or tokenizer is a separate
experiment. Model/vector/projector licenses are separate from Strata's MIT
license; the retained model card identifies Qwen Community License 1.0.

## Launch layout and inherited environment

Copy/edit the example as `/srv/qwen/config/strata-server.json`, set your own API
key, and provide all referenced assets and writable log directories first.
The example binds `127.0.0.1:8002`, has an empty `api_key` requiring operator
configuration, and ships `compatibility_api_keys: []`. A listening service
without a supplied key has no authentication; do not assume the blank field
creates a secret. Override any CLI host/port consistently with the JSON.

Use working directory `/srv/qwen/runtime-0.1.42` and the environment in
[service-environment.example.conf](config/service-environment.example.conf).
The qualified module invocation, with paths/binding sanitized, is:

```text
/srv/qwen/venv/bin/python -m serve.server --engine strata --config /srv/qwen/config/strata-server.json --host 127.0.0.1 --port 8002
```

The served alias comes from `model_name` in the JSON. Benchmark/client commands
that accept a model argument must use
`--model qwen3.8-flash-next-custom-iq3-s` with this example; the original
measurements used a different endpoint alias for these same custom model assets.
Do not append `--model` to the Python server command: that version has no such
server CLI argument.

That Python server appends `--batch 8` to the configured decoder arguments when
`parallel` is 8. Do not start a second engine beside an existing model server.
The six config `env` entries are inherited by its engine children. Server-side
queue/rotation variables must also be present before the Python process starts.
`STRATA_VERIFY_ALL_RESIDENT=1` and `STRATA_ARENA_MMAP=1` were inherited in the
qualified service and are included for fidelity.
The example omits `fit_max_tokens`, matching the qualified benchmark snapshot.
A later live-config read showed an explicit `fit_max_tokens=true`; that API
overflow policy was not part of the speed qualification and is not a speed
setting. It fits requested output limits into remaining context space and
does not change the 131,072-token context allocation.

Eight full-context INT8 slots, full main KV, all 24,576 experts resident and a
450 W GPU policy describe the qualified 96 GB card profile. `--kv-resident 131072`
preserves the drafter ring; metrics report full main/slot KV residency as
`kv_resident=0`. `--spec 4`/MTP are configured for the solo path; ordinary C8
target decoding in the cited measurement had no per-slot draft proposals.
The four batch flags were qualified together with the new engine package.
On SM120 the default Daily attention can retain per-row attention even with
`QSA_SLOTS=1`; scores/top-k still batch. `PF_FUSED=0` avoids the new differently
rounded native IQ3_S prefill default.

The original service used `LimitMEMLOCK=infinity`, swap disabled for its cgroup,
and host-specific `MemoryHigh=54G`/`MemoryMax=58G`; these are operational limits,
not portable RAM requirements. Provide adequate RAM for the PLE table, runtime
and caches. The 450 W policy and process-bound readiness/recovery controls are
external to Strata's JSON and are not installed by this example. No service,
GPU power setting, model download, firewall rule or production path is changed
by these files.
