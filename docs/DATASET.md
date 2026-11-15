# Dataset and fixed split

| Partition | Field-years | Physical fields | Grain / silage | Index observations |
|---|---:|---:|---:|---:|
| `data/train` | 599 | 599 | 37 / 562 | 5,066 |
| `data/test` | 44 | 34 | 32 / 12 | 1,774 |

Train: Yakima, Washington, 2015. Test: BARC, Maryland, 2021–2023. No field-year or physical-field overlap exists. No MD archive, CA, NE, PA or NL samples are included in CRUNCH training. These are the optical-supported samples from a frozen prior collection, not all available agricultural fields; Yakima originally had 620 labels, of which 599 met optical support criteria.

## Files and schema

Each partition contains `fields.csv`, `observations.csv` and `ids.txt`. Fields contain `field_year_id` (unique join key), `physical_field` (repeat-year grouping), `region`, `year`, `label` (`grain`/`silage`), `source_label`, `label_source` and `cohort`. Source labels and provenance columns are metadata, never predictors. The physical ID may differ from the numeric field-year suffix due to the frozen source crosswalk; use the provided mapping.

Observations contain `field_year_id,date,ndvi,lswi,ndti`. Values are unscaled index values on irregular dates. These are the raw model inputs, already reduced from quality-filtered imagery; this package does not contain raw Landsat/Sentinel raster scenes, polygons, weather or new imagery. Multiple observations within a season are not separate labeled training examples.

`train.py` reads only the train partition. `evaluate.py` reads only the requested evaluation partition and an already fitted checkpoint. The metadata year specifies the harmonic time origin. Input predictor columns cannot override features freshly computed by `predict.py`.

## Provenance

- WA labels: [Lower Yakima GWMA layer](https://services3.arcgis.com/9Qz94N8Zml9hnG84/ArcGIS/rest/services/GWMA_Project_2/FeatureServer/15), including WSDA and producer entries. The upstream categories distinguish grain and silage corn. Frozen WA observations came from Landsat 8 surface reflectance.
- MD labels and effective index observations: the user-supplied Hack the Harvest / USDA BARC competition snapshot. [Competition resources](https://hack-the-harvest.devpost.com/resources). The local snapshot contains 44 field-years and is distinct from the 49-field-year [USDA BARC archive](https://doi.org/10.15482/USDA.ADC/28950746); the latter's license or sample count must not automatically be substituted for the competition subset.
- Input source hashes: `reference/source_hashes.json`. Within-release checksums: `CHECKSUMS.sha256`.

See [DATA_LICENSE.md](../DATA_LICENSE.md) for source-specific license status. Neither label collection nor underlying satellite data authorship is claimed by CRUNCH's authors.

## Evaluation interpretation

MD is excluded from parameter fitting, but has been inspected in earlier model development. Do not present the released result as an untouched blind test. The 44 MD field-years represent 34 physical fields and are not 44 independent locations. The training population is heavily silage-dominated, whereas the test set is grain-dominated. Report Macro-F1, both class errors, and coverage along with Accuracy.
