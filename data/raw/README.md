# Local CSE-CIC-IDS2018 data setup

The dataset is intentionally not distributed with this repository. Obtain CSE-CIC-IDS2018 from an authorized source and copy the following files into `data/raw/csv/`:

```text
02-14-2018.csv
02-15-2018.csv
02-16-2018.csv
02-20-2018.csv
02-21-2018.csv
02-22-2018.csv
02-23-2018.csv
02-28-2018.csv
03-01-2018.csv
03-02-2018.csv
```

The CSV copy is the primary temporal input because it contains `Timestamp`. If available, place the optional Parquet comparison copy under `data/raw/parquet/`.

Both directories are Git-ignored. Scripts read the raw inputs but write generated outputs only under `data/processed/`, `results/`, `logs/`, or other output directories. Do not modify the raw files in place.

From the repository root, verify discovery with:

```powershell
python scripts/inspect_dataset.py --debug
```

Run the frozen Stage C pipeline only after all 10 expected CSVs are present. Generated Stage C arrays are local artifacts and are not fetched from GitHub.
