# Upstream CPU baseline failures — retained evidence

Exact source: `906be195c1052aaa74f95852ad1917ce1926ba64`; unchanged upstream lock; Python 3.12.3 / torch 2.13.0+cu129 / Triton 3.7.1 on Onyx CPU, two threads and CUDA hidden.

Selected upstream suite: **144 passed, 3 failed**. These results are not a complete upstream CI run. The failing tests were not skipped, weakened, or patched. See `receipts/upstream-cpu-tests.log` and XML.

## Clustering: material reference issue

`test_ward_never_stops_short_at_a_cap_of_twice_the_bucket_size`, pool-size cases 1 and 4, fails for the near-copy vectors at seed 1. The reference stops with 18 buckets instead of 16 in one group, and 68 instead of 64 in another. It raises rather than silently returning a truncated valid result.

Instrumentation of the same merge loop shows allowed pairs still exist at the stop (160 and 2760 directed allowed pairs). The FP32 Gram matrix and merge costs are slightly asymmetric; maximum cost asymmetry at the stops is about 1.91e-6 and 9.54e-7. Running the unchanged reference on the same inputs cast to FP64 reaches the target in every group for these two cases.

Inference: floating-point asymmetry is a plausible cause of the mutual-nearest-pair selection stalling. This is supporting evidence, not a full proof or a GPU reproduction. Tiny numerical errors can have a discrete clustering consequence; do not dismiss this as harmless rounding. A symmetric-cost or higher-precision reference change is a candidate to investigate separately, not an applied fix.

The exact published source is preserved for reproducibility. Any eventual repair should have its own patch/revision, retain the failing regression, and compare the Triton path on the affected near-copy cases before claiming qualification.

## Attention: bitwise comparison is stronger than observed agreement

`test_query_blocking_does_not_change_the_answer` uses `torch.equal` across two query-block sizes. A separate four-seed diagnostic observes finite results with maximum absolute differences of 2.53e-7 to 3.58e-7. All four agree at atol 1e-6 / rtol 1e-5, while bitwise equality fails.

This supports a floating-point rounding explanation for this test, but the original failing result remains recorded. No assertion was changed. The diagnostic is not evidence that every attention case or a CUDA implementation agrees.

## Reproduction and effect on the GPU window

`diagnose_reference.py` reruns the 12 small Ward cases and four attention comparisons entirely on CPU. It records `receipts/reference-diagnostics.json` and never modifies upstream source.

The test kit can still be prepared and a later explicitly opened window can investigate CUDA compatibility. A successful synthetic smoke test must not be presented as a clean upstream baseline or general LSA correctness. Review or resolve the clustering defect before accepting performance numbers for integration decisions. Flash-Next/Strata and model-level quality remain unqualified.
