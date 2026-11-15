# Attribution and third-party components

CRUNCH's architecture, implementation and model release are authored by **TIANYANG LIU and BORAN CUI**. The team's original dual-branch method is part of CRUNCH, not an external competitor's model.

Credit upstream field labels to Lower Yakima / WSDA / producer records and Hack the Harvest / USDA BARC, and satellite observations to their upstream Landsat/Sentinel sources. These datasets are not represented as originally collected or owned by the model authors. Data license scope is documented separately in `DATA_LICENSE.md`.

NumPy, SciPy, pandas, scikit-learn and joblib are dependencies distributed by their own projects under their own licenses. Optional benchmark dependencies aeon and PyTorch are likewise separate projects. No dependency source tree or binaries are vendored here.

ROCKET, MiniRocket, MultiRocket and TempCNN are credited to the original paper authors in `docs/BENCHMARKS.md` and `docs/references.bib`. Benchmark scripts are local adapters/implementations; they are not presented as those authors' official complete experiments. The TempCNN architecture port does not claim exact numerical equivalence to the original Keras code.
