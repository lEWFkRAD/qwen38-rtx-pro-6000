# Configuration-integrity incident — November 4, 2026 UTC

Status: recovery verified on November 4 UTC. The approved credential configuration passed managed readiness and an exact canary at **02:27:10.620721 UTC**. New independent and remote-client checks subsequently passed; `production_final_ready=true`. See [recovery-summary-public.json](recovery-summary-public.json) for the credential-free evidence summary.

At approximately 02:13 UTC, a later edit changed only the top-level API credential. A direct service restart then failed the strict pre-start configuration hash check: the protected contract still pinned the approved configuration. Serving arguments, environment and the 262144-context/eight-slot/prefill-8192 profile were unchanged. The verified cause was the credential-only configuration hash mismatch, not a 262K memory failure.

At 02:19:03 UTC, the root operator privately preserved the edited configuration and restored the byte-exact approved backup. Credentials, actual configurations, private backup paths, configuration fingerprints and operational recovery scripts are excluded from this publication.

The automatic recovery policy permits two starts per hour and would otherwise wait until **02:29:57.977 UTC**. The user then explicitly instructed approval of the new credential and an immediate start rather than waiting for that window. This was an authorized, one-time manual exception, not an automatic-policy change.

The root operator approved the new credential as an additive compatibility-key entry, retaining the primary and existing keys, then resealed the configuration, protected contract and dependent personal pins under locks, compare-and-swap checks and owned maintenance control. A comma-joined primary-key value was not used. Actual credentials and private pin values are excluded here.

One manual recovery began at **02:26:17.188249 UTC**, before the automatic window. All previous attempts were preserved and this single attempt was appended; automatic decision-limit code and readiness checks were unchanged. Managed readiness and an exact canary passed at **02:27:10.620721 UTC**. This was an explicitly authorized manual exception, not an override-free restart or a change to the automatic policy.

A separate recovery audit at 02:27:42.349494 UTC passed **19/19 checks**, including 262144 context, eight slots, ordinary fully resident INT8 KV, all 24576 GPU-resident experts, matched vision and prefill 8192. The three approved credentials all passed authentication, while an invalid credential returned 401. No credential values or fingerprints are published.

At **02:27:57.7167785 UTC**, the remote client passed four GET checks for health, status, models and metrics, and one synthetic completion returned exactly `ONYX_RECOVERY_OK` with six output tokens (about 229 ms). Its provider catalog bytes were unchanged and no client restart was performed. Sunshine remained stopped with enablement unchanged.

The earlier promotion's 19/19 independent audit and 7/7 production canaries are unchanged historical evidence from before this incident. Current recovery readiness is established by the separate new evidence above, not by relabeling those earlier results. All research failures and the unqualified eight-deep simultaneous-decoding limit also remain unchanged.

This record describes a later availability incident, separate from the completed capacity and retrieval experiments. It does not change the approved production profile or claim a new model-quality or concurrency qualification.
