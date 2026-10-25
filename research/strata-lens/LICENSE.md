# License scope

Copyright 2026 lEWFkRAD

The original downstream Strata Lens scripts, tests, example configuration,
probe corpus and documentation in this directory are licensed under the
Apache License, Version 2.0, consistent with the repository's existing
[downstream license scope](../../strata-c8/LICENSE.md). The complete terms are
in [APACHE-2.0.txt](../../strata-c8/LICENSES/APACHE-2.0.txt).

This grant covers the original downstream work only. It does not relicense
Strata, Python, `regex`, the Qwen model, GSQ/RCO tooling or quantized weights,
the Huihui-derived control vector, tokenizer assets, generated model packs,
MTP weights, or NVIDIA/CUDA components. Those dependencies retain their own
licenses and are obtained separately.

Strata is MIT-licensed; see the existing
[Strata MIT copyright and permission notice](../../strata-c8/LICENSES/Strata-MIT.txt).
The capture harness calls its installed executable and imports its installed
tokenizer. No Strata source or engine binaries are vendored in this directory.

The model package and control vector have their own Qwen Community License
1.0 terms. Refer to their pinned upstream sources in [CREDITS.md](CREDITS.md).
No model or vector license is replaced by this Apache grant.

Neuron-explorer is credited as methodological inspiration. Its code, assets
and data are not included here, and no license to them is asserted.

Third-party names and marks belong to their respective owners. This is an
independent experiment; no upstream endorsement is claimed.
