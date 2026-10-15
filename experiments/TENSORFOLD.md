# TensorFold tuning — October 3, 2026

Evidence: [six-configuration benchmark receipt](receipts/tensorfold-comparison.json)
and [four-stream qualification](receipts/tensorfold-four-stream.json).

The initial TensorFold 0.6.3 attempt with the existing NVIDIA-style checkpoint
failed memory admission. The successful trial used the supported MLX 4-bit/MTP
checkpoint, revision `2b170fa6309d5d1ee380b35636075fac7945f286`.
TensorFold 0.6.4 source was
`6ea5ade26c4335491c50275af0be32b81f75f525`.

This changes weight representation relative to the SGLang signed-W4 deployment.
The following are configuration comparisons, not equal-quality engine A/Bs.

## From text throughput to a qualified vision profile

The 0.6.3 balanced text trial measured approximately 423 aggregate tok/s for
four short requests. Version 0.6.4 preserved CUDA graphs with only one live
request in a multistream server: warm code decode reached about 376 tok/s and
warm prose about 224 tok/s. Those are individual decode estimates, not batch
throughput or first-call guarantees.

Enabling vision exposed memory-admission limits. A two-stream BF16-KV profile
was qualified before four-stream INT8 work. The stock four-stream INT8 trial
delivered 388.08/396.57/395.41 aggregate tok/s on short bursts but **failed**
long concurrent cache growth. Those fast rejected results are not a qualified
alternative.

The isolated growth fix allows geometric cache growth only while truly solo;
concurrent requests retain existing 8K increments. It avoids about 0.978 GiB
of unnecessary retained INT8 cache at the 64K boundary. Patch SHA-256:
`0e5cc95808c65f510d68e26412d4764b1e46b27a4c696b2b782830f539df56fe`.

| Selected four-stream profile | Recorded value |
| --- | --- |
| Context per request | 131,072 |
| KV cache | INT8 |
| MTP / confidence | 6 / 0.5 |
| Prefill rows / GPU reserve | 1,024 / 4 GiB |
| Vision | Enabled |
| Short four-request aggregate samples | 378.76, 383.08, 386.42 tok/s |
| Median | 383.08 tok/s |
| Warm single code / prose decode estimates | 305.70 / 214.78 tok/s |
| Cold approximately 120K TTFT | 18.9659 s |

For context, the original SGLang baseline had one 200.28 aggregate tok/s burst
and 10.8462 s approximately 120K TTFT. Different checkpoints, cache precision,
vision settings and sample counts prevent an isolated engine speedup claim.
The later two-stream baseline varied substantially and was reused from a prior
trial; the receipt preserves that variability.

## What the long-context checks proved

The patched profile repeated six bounded correctness cases, four vision
checks, nine API checks and seven transition/cancellation checks, plus
drafted/serial and fresh/cached parity. Four approximately 63K prompts each
finished 8,192 output tokens, including one image, but the original all-four
boundary proof was inconclusive.

A further four approximately 66K prompts each completed 4,096 tokens, with
health evidence that all four were decoding after every prompt exceeded 64K.
This establishes the tested boundary overlap. **Four simultaneous full 128K
histories and six active generations were not qualified.**

The selected profile was promoted on October 3, then superseded by the Strata
experiment. This page records that date's result, not a current deployment recipe.
