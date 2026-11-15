# Contributing

Use Python 3.12 and `requirements.txt`. Before proposing a change, run `python verify.py --retrain` on the unchanged release to establish the reference behavior. Keep generated experiments under `outputs/` and preserve `models/`, `data/` and `reference/` as frozen artifacts.

For algorithm changes, report the exact split, source-only preprocessing, decision threshold, both class errors, Accuracy and Macro-F1. Keep multiple years of a physical field together in split analyses. Do not choose the best seed or tune on test labels and describe the result as a blind evaluation.

Document intended changes to reference outputs explicitly. A checksum failure after intentionally editing source code is expected; do not regenerate release checksums merely to hide changed predictions. Changes to the scientific model should receive a separate version and fresh validation evidence.

Do not commit virtual environments, credentials, downloaded image caches or transient output directories. Retain paper citations and source-data attribution when extending benchmarks.
