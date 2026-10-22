# Schematic provenance

The [schematic](qwen-c8-firm-throughput-v2.png) was created with built-in image
generation on October 22, 2026 and checked against the accompanying
[measurement receipt](../evidence/results-2026-10-22.json). It is a conceptual
illustration, not a screenshot of instrumentation or a proportional VRAM chart.

This corrected version labels the GPU-resident **expert weights**, the PLE
table in **host RAM**, and the measured C8 path as **batched target decode**.
The configured native drafter did not propose tokens in those C8 measurements.
The full model variant and dependencies are stated in
[DEPENDENCIES.md](../DEPENDENCIES.md).
