# Results layout

- `results/demo/` — outputs from coco8 / synthetic / early prototype runs. **Not final.**
- `results/coco_person/` — final validated COCO person-subset outputs:
  - `tables/val_per_image.csv`: 11,400 rows, 0 failures, 0 duplicates
  - `tables/val_aggregate.csv`: 76 groups, 150 images per group
  - `final_results_manifest.json`: merged final-result provenance and invariants
  - `figures/`: validated source-data CSVs derived from `val_aggregate.csv`;
    PNG/PDF files are rendered by `scripts/06_make_plots.py`
  - `qualitative/`: verified sample-selection metadata
- `results/archive_failed_local/` — archived failed local run; not used.

Only `results/coco_person/` is used for the final report. Detection results are
F1/precision/recall metrics, not AP50. Motion-blur enhancement is stable
unsharp masking.
