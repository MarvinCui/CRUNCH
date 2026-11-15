# Reproduce the published-method adaptations

These optional experiments do not affect CRUNCH's released checkpoint. The fixed-threshold protocol uses only the packaged WA train and MD test observations, not NL selection. The 24 reference runs are saved under `reference/`. See [the comparison document](../docs/BENCHMARKS.md) for papers and adaptation details.

```bash
# From the repository root, with the CRUNCH CPU environment active:
python -m pip install -r benchmarks/requirements-cpu.txt
OPENBLAS_NUM_THREADS=1 python benchmarks/prepare.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python benchmarks/rocket.py

# TempCNN: activate a GPU environment with benchmarks/requirements-gpu.txt.
# On a cluster, run inside an allocated Slurm GPU job; no host-specific partition is assumed.
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python benchmarks/tempcnn.py

# Return to the CPU environment to summarize the rerun:
python benchmarks/summarize.py

# Or verify metrics from shipped predictions, with no GPU or retraining:
python benchmarks/summarize.py --reference
```

CPU environment: Python 3.12, aeon 1.6.0 and the root numerical dependencies. Original GPU environment: Python 3.10.20, PyTorch 2.11.0+cu128, NumPy 2.2.6, pandas 2.3.3, NVIDIA RTX PRO 6000 Blackwell. Use an appropriate official PyTorch wheel for the allocated GPU. CRUNCH itself has no GPU dependency.

The adapters were derived from the earlier broader comparison by removing NL inference only. WA sequence ordering, source-only normalization, source inner split, seeds and architectures are preserved. All class-weighting modes remain visible. `prepare.py` and the CPU benchmark were checked against the original frozen predictions. No baseline weights are needed to use CRUNCH; retraining generates them under `results/`.

The scripts are local wrappers/ports, not vendored copies of the authors' complete research repositories. Timings include implementation-specific work and should not be used for a CPU-vs-GPU efficiency ranking.
