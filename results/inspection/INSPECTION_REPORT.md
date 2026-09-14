# CIC-IDS2018 Dataset Inspection Report

**Scope:** INSPECT → MEASURE → REPORT only. No dataset file was modified, renamed, deleted, merged or converted. No preprocessing, graph, model, pruning or training code was written.
**Method:** read-only streaming scans (`pyarrow.csv.open_csv`, every column forced to `string` so no dtype inference can hide malformed values) + Parquet metadata/row-group statistics + single columnar reads. All numbers below are measured, not estimated. Where a figure is derived or assumed, it is labelled as such.
**Raw measurements:** `results/inspection/*.json` (30 files).
**Disposable scripts used:** `scratch/inspect_csv.py`, `scratch/inspect_csv_extra.py`, `scratch/inspect_parquet.py` — safe to delete.

---

## 1. Workspace and data structure

Project root: `C:\Project\ComputerNetworks` (note: the earlier OneDrive path is now empty; all Phase-1 work and the data live here).

```
ComputerNetworks/
├── configs/default.yaml            # Phase 1 config (T=5, 0.70/0.10/0.20, seed 42)
├── src/{config.py, preprocessing/, features/, graph/, models/, pruning/, training/, evaluation/, utils/}
├── scripts/inspect_dataset.py
├── tests/{test_setup.py, conftest.py}
├── scratch/                        # DISPOSABLE inspection scripts (3 files)
├── results/inspection/             # 30 measurement JSONs produced by this pass
├── logs/                           # scan logs
├── data/
│   ├── raw/csv/                    # 10 files, 6,886,649,507 B (6.41 GiB)
│   ├── raw/parquet/                # 10 files,   725,047,382 B (0.68 GiB)
│   └── processed/                  # empty
├── extracted_report.txt            # PRIMARY SOURCE OF TRUTH
└── README.md, requirements*.txt, environment.yml, pyproject.toml
```

Total raw data on disk: **7,611,696,889 B ≈ 7.09 GiB**.

---

## 2. CSV inventory (`data/raw/csv/`)

| File | Bytes | GiB | Data rows | Cols | Labels present |
|---|---:|---:|---:|---:|---|
| 02-14-2018.csv | 358,223,333 | 0.33 | 1,048,575 | 80 | Benign 667,626 · FTP-BruteForce 193,360 · SSH-Bruteforce 187,589 |
| 02-15-2018.csv | 375,945,899 | 0.35 | 1,048,575 | 80 | Benign 996,077 · DoS-GoldenEye 41,508 · DoS-Slowloris 10,990 |
| 02-16-2018.csv | 333,723,605 | 0.31 | 1,048,575 | 80 | DoS-Hulk 461,912 · Benign 446,772 · DoS-SlowHTTPTest 139,890 · **"Label" ×1** |
| 02-20-2018.csv | 4,054,925,350 | 3.78 | 7,948,748 | **84** | Benign 7,372,557 · DDoS-LOIC-HTTP 576,191 |
| 02-21-2018.csv | 328,893,673 | 0.31 | 1,048,575 | 80 | DDOS-HOIC 686,012 · Benign 360,833 · DDOS-LOIC-UDP 1,730 |
| 02-22-2018.csv | 382,636,202 | 0.36 | 1,048,575 | 80 | Benign 1,048,213 · BF-Web 249 · BF-XSS 79 · SQLi 34 |
| 02-23-2018.csv | 382,840,456 | 0.36 | 1,048,575 | 80 | Benign 1,048,009 · BF-Web 362 · BF-XSS 151 · SQLi 53 |
| 02-28-2018.csv | 209,249,758 | 0.19 | 613,104 | 80 | Benign 544,200 · Infilteration 68,871 · **"Label" ×33** |
| 03-01-2018.csv | 107,842,858 | 0.10 | 331,125 | 80 | Benign 238,037 · Infilteration 93,063 · **"Label" ×25** |
| 03-02-2018.csv | 352,368,373 | 0.33 | 1,048,575 | 80 | Benign 762,384 · Bot 286,191 |
| **TOTAL** | **6,886,649,507** | **6.41** | **16,233,002** | | |

`"Label" ×N` = repeated header rows embedded in the data (see §8).

**Seven of ten files contain exactly 1,048,575 rows = 2²⁰ − 1**, the Excel worksheet row limit. This is an artefact, and it strongly suggests those seven files were truncated when passed through a spreadsheet. Their last data rows are not the chronological end of the day (e.g. 02-14 ends at `10:53:28`, 02-23 ends mid-sequence at `09:04:11`). The three files that are *not* at the cap — 02-20 (7,948,748), 02-28 (613,104), 03-01 (331,125) — are presumably complete.

---

## 3. Parquet inventory (`data/raw/parquet/`)

All filenames follow `<Scenario>-<Weekday>-<DD-MM-YYYY>_TrafficForML_CICFlowMeter.parquet`. All: 78 columns, 1 row group, `created_by = parquet-cpp-arrow version 8.0.0`.

| File | Bytes | Rows | Labels |
|---|---:|---:|---|
| Botnet-Friday-02-03-2018 | 83,481,543 | 771,587 | Benign 627,052 · Bot 144,535 |
| Bruteforce-Wednesday-14-02-2018 | 76,430,026 | 619,346 | Benign 525,245 · SSH-Bruteforce 94,048 · FTP-BruteForce **53** |
| DDoS1-Tuesday-20-02-2018 | 89,492,377 | 954,846 | DDoS-LOIC-HTTP 575,364 · Benign 379,482 |
| DDoS2-Wednesday-21-02-2018 | 33,448,003 | 561,396 | Benign 360,805 · DDOS-HOIC 198,861 · DDOS-LOIC-UDP 1,730 |
| DoS1-Thursday-15-02-2018 | 98,618,434 | 794,812 | Benign 743,498 · DoS-GoldenEye 41,406 · DoS-Slowloris 9,908 |
| DoS2-Friday-16-02-2018 | 57,992,315 | 591,873 | Benign 446,619 · DoS-Hulk 145,199 · DoS-SlowHTTPTest **55** |
| Infil1-Wednesday-28-02-2018 | 50,954,894 | 456,873 | Benign 400,424 · Infilteration 56,449 |
| Infil2-Thursday-01-03-2018 | 29,282,940 | 249,170 | Benign 187,136 · Infilteration 62,034 |
| Web1-Thursday-22-02-2018 | 103,957,293 | 830,224 | Benign 829,883 · BF-Web 228 · BF-XSS 79 · SQLi 34 |
| Web2-Friday-23-02-2018 | 101,389,557 | 829,405 | Benign 828,864 · BF-Web 340 · BF-XSS 150 · SQLi 51 |
| **TOTAL** | **725,047,382** | **6,659,532** | |

Every weekday in the filenames is correct for its date, and each file's date maps 1:1 onto a CSV of the same day.

---

## 4. CSV vs Parquet comparison

### 4A. Confirmed differences (measured)

**a) Different column counts and different columns.**
CSV (9 files) = 80 columns; CSV 02-20 = 84; Parquet = 78.

| | CSV (80-col) | CSV 02-20 (84-col) | Parquet (78) |
|---|---|---|---|
| Flow ID, Src IP, Src Port, Dst IP | ✗ | ✓ | ✗ |
| Dst Port | ✓ | ✓ | **✗** |
| Timestamp | ✓ | ✓ | **✗** |
| Protocol | ✓ | ✓ | ✓ |
| 76 flow statistics + Label | ✓ | ✓ | ✓ |

**The Parquet files contain no timestamp column at all.** This is decisive for this project (see §9).

**b) Column-name style differs, but the 78 shared columns align positionally and exactly.**
Removing `Dst Port` and `Timestamp` from the CSV header leaves 78 names that map 1:1 in order onto the Parquet schema. 50 of 78 names match after normalisation; the other 28 are pure short-form/long-form renames, e.g.:

| CSV | Parquet |
|---|---|
| `TotLen Fwd Pkts` | `Fwd Packets Length Total` |
| `Fwd Pkt Len Max` | `Fwd Packet Length Max` |
| `Pkt Len Var` | `Packet Length Variance` |
| `Pkt Size Avg` | `Avg Packet Size` |
| `Fwd Byts/b Avg` | `Fwd Avg Bytes/Bulk` |
| `Fwd Blk Rate Avg` | `Fwd Avg Bulk Rate` |

Position 1 → `Protocol` and position 78 → `Label` in both, confirming the alignment is not accidental.

**c) Row counts differ substantially, always Parquet < CSV.**

| Day | CSV rows | Parquet rows | Δ |
|---|---:|---:|---:|
| 14 Feb | 1,048,575 | 619,346 | −429,229 |
| 15 Feb | 1,048,575 | 794,812 | −253,763 |
| 16 Feb | 1,048,575 | 591,873 | −456,702 |
| 20 Feb | 7,948,748 | 954,846 | −6,993,902 |
| 21 Feb | 1,048,575 | 561,396 | −487,179 |
| 22 Feb | 1,048,575 | 830,224 | −218,351 |
| 23 Feb | 1,048,575 | 829,405 | −219,170 |
| 28 Feb | 613,104 | 456,873 | −156,231 |
| 01 Mar | 331,125 | 249,170 | −81,955 |
| 02 Mar | 1,048,575 | 771,587 | −276,988 |
| **Total** | **16,233,002** | **6,659,532** | **−9,573,470** |

**d) Class distributions differ, in some cases catastrophically.**

| Class | CSV total | Parquet total |
|---|---:|---:|
| FTP-BruteForce | 193,360 | **53** |
| DoS attacks-SlowHTTPTest | 139,890 | **55** |
| DDOS attack-HOIC | 686,012 | 198,861 |
| DoS attacks-Hulk | 461,912 | 145,199 |
| Bot | 286,191 | 144,535 |
| Infilteration | 161,934 | 118,483 |
| SSH-Bruteforce | 187,589 | 94,048 |
| Benign | 13,484,708 | 5,329,008 |

Two attack classes are effectively destroyed in the Parquet copy. **The CSV and Parquet sets are not interchangeable.**

**e) Parquet dtypes are inconsistent between files.**
31 of 78 columns have a different physical type in different Parquet files, e.g. `Flow Duration` is `int32` in most files but `int64` in Bruteforce; `Total Backward Packets` appears as `int8`, `int16` and `int32`; `Flow Bytes/s` as `float` and `double`. Concatenating the ten Parquet files requires an explicit schema harmonisation step.

### 4B. Confirmed equivalence

Tested by comparing the exact multiset of 16-column integer keys (`Flow Duration`, `Tot Fwd/Bwd Pkts`, `TotLen Fwd/Bwd Pkts`, `Flow IAT Max/Min`, `Fwd/Bwd Header Len`, `Init Fwd/Bwd Win Byts`, `Idle Max/Min`, `Pkt Len Max/Min`, `Label`) between each Parquet file and its CSV.

For all five pairs tested (14 Feb, 15 Feb, 16 Feb, 20 Feb, 01 Mar):

- **Subset violations (a key occurring more often in Parquet than in the CSV): 0.**
- **Keys present in Parquet but absent from the CSV: 0** (after excluding CSV rows with `Flow Duration == 0`).

So: **the Parquet rows are a genuine subset of the CSV rows, and the feature values agree exactly.** The Parquet copy is a filtered derivative of the CSV, not an independent extraction. The column semantics of the positional alignment in §4A(b) are therefore confirmed, not merely assumed.

### 4C. Likely equivalence (not fully verified)

- The five untested day-pairs (21 Feb, 22 Feb, 23 Feb, 28 Feb, 02 Mar) very likely follow the same subset relationship, given 5/5 confirmations and identical schemas. Not verified.
- CSV rows with `Flow Duration == 0` appear to be excluded from the Parquet copy (`Flow Duration` min is ≥ 1 in 8 of 10 Parquet files). Consistent with all evidence, but it is not the *only* filter applied.

### 4D. Cannot be determined without further work

- **The exact rule that produced the Parquet subset is unknown.** For 14 Feb: the CSV has 1,044,751 rows with `Flow Duration != 0`, of which 819,126 are distinct; the Parquet has 619,346. Neither "drop zero-duration" nor "drop duplicates" nor both explains the count, and neither explains FTP-BruteForce collapsing from 39,346 distinct rows to 53. Some additional filter or subsample was applied that I cannot recover from the files.
- Whether the Parquet float columns are bit-identical to the CSV decimal strings for all 6.66 M rows (would need a full-precision full-file comparison).
- Whether the 5 untested day-pairs contain any Parquet-only rows.

---

## 5. Actual schema

### CSV — 80 columns (9 of 10 files; header md5 `64a4bae24658fa55bc9f55889af725da`, byte-identical across all nine)

```
Dst Port, Protocol, Timestamp, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts,
TotLen Fwd Pkts, TotLen Bwd Pkts, Fwd Pkt Len Max, Fwd Pkt Len Min,
Fwd Pkt Len Mean, Fwd Pkt Len Std, Bwd Pkt Len Max, Bwd Pkt Len Min,
Bwd Pkt Len Mean, Bwd Pkt Len Std, Flow Byts/s, Flow Pkts/s, Flow IAT Mean,
Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Tot, Fwd IAT Mean,
Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std,
Bwd IAT Max, Bwd IAT Min, Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags,
Bwd URG Flags, Fwd Header Len, Bwd Header Len, Fwd Pkts/s, Bwd Pkts/s,
Pkt Len Min, Pkt Len Max, Pkt Len Mean, Pkt Len Std, Pkt Len Var,
FIN Flag Cnt, SYN Flag Cnt, RST Flag Cnt, PSH Flag Cnt, ACK Flag Cnt,
URG Flag Cnt, CWE Flag Count, ECE Flag Cnt, Down/Up Ratio, Pkt Size Avg,
Fwd Seg Size Avg, Bwd Seg Size Avg, Fwd Byts/b Avg, Fwd Pkts/b Avg,
Fwd Blk Rate Avg, Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg,
Subflow Fwd Pkts, Subflow Fwd Byts, Subflow Bwd Pkts, Subflow Bwd Byts,
Init Fwd Win Byts, Init Bwd Win Byts, Fwd Act Data Pkts, Fwd Seg Size Min,
Active Mean, Active Std, Active Max, Active Min,
Idle Mean, Idle Std, Idle Max, Idle Min, Label
```

Structure: 1 identifier-ish column (`Dst Port`), 1 protocol column, 1 timestamp, **76 numeric flow features**, 1 label.

### CSV — 84 columns (02-20-2018.csv only; header md5 `8e916b7d2b522d9158c2d533cd436171`)

Identical to the above with `Flow ID, Src IP, Src Port, Dst IP` prepended. Verified by diffing the headers after removing those four names — the remaining 80 are byte-identical.

### Parquet — 78 columns (all 10 files, identical names)

`Protocol` + 76 numeric flow features + `Label`. `Label` is stored as `dictionary<values=string, indices=int8>`. **No `Timestamp`, no `Dst Port`, no IP/port/Flow-ID fields.** Physical integer/float widths vary per file (§4A(e)).

### Feature-count reconciliation

Whichever file you use, the number of *usable flow features* is **76** (excluding `Dst Port`, `Timestamp`, `Label`, and the four 02-20-only identifier columns). "80" only holds if you count `Dst Port` + `Timestamp` + `Label` alongside the 76 features, or `Protocol` + 76 features + `Label` + one more. See §11.

---

## 6. Actual labels and class distribution

### CSV corpus (16,233,002 rows across 10 files)

| Label | Count | Share |
|---|---:|---:|
| Benign | 13,484,708 | 83.07 % |
| DDOS attack-HOIC | 686,012 | 4.23 % |
| DDoS attacks-LOIC-HTTP | 576,191 | 3.55 % |
| DoS attacks-Hulk | 461,912 | 2.85 % |
| Bot | 286,191 | 1.76 % |
| FTP-BruteForce | 193,360 | 1.19 % |
| SSH-Bruteforce | 187,589 | 1.16 % |
| Infilteration | 161,934 | 1.00 % |
| DoS attacks-SlowHTTPTest | 139,890 | 0.86 % |
| DoS attacks-GoldenEye | 41,508 | 0.26 % |
| DoS attacks-Slowloris | 10,990 | 0.07 % |
| DDOS attack-LOIC-UDP | 1,730 | 0.011 % |
| Brute Force -Web | 611 | 0.0038 % |
| Brute Force -XSS | 230 | 0.0014 % |
| SQL Injection | 87 | 0.0005 % |
| *`Label` (header echo, not a class)* | 59 | — |

**15 real classes** (1 benign + 14 attack types). Imbalance ratio Benign : SQL Injection = **155,000 : 1**.

### Parquet corpus (6,659,532 rows)

Same 15 classes; Benign 5,329,008 (80.02 %), and the distortions listed in §4A(d).

### Label-string hazards
- Inconsistent casing: `DDOS attack-HOIC` / `DDOS attack-LOIC-UDP` vs `DDoS attacks-LOIC-HTTP`.
- Inconsistent casing: `FTP-BruteForce` vs `SSH-Bruteforce`.
- Leading space in the value: `Brute Force -Web`, `Brute Force -XSS`.
- Misspelling in the source data: **`Infilteration`** (not "Infiltration").
- The literal string `Label` appears 59 times as a value, from embedded header rows.
- No `NaN`, empty, or null labels in any file.

---

## 7. Timestamp findings

Timestamps exist **only in the CSV files**. Format is `%d/%m/%Y %H:%M:%S`, resolved by trying seven candidate formats and picking the best-parsing one; 0 rows fail to parse in 8 of 10 files.

| File | Parsed min | Parsed max | Distinct values | Adjacent-decreasing pairs (file order) | AM/PM markers |
|---|---|---|---:|---:|---:|
| 02-14 | 1970-01-05 03:01:17 | 2018-02-14 12:59:59 | 32,043 | 289,765 | 0 |
| 02-15 | 2018-02-15 01:00:00 | 2018-02-15 12:59:59 | 33,862 | 382,967 | 0 |
| 02-16 | 2018-02-16 01:00:32 | 2018-02-16 12:58:24 | 3,178 | 268,941 | 0 |
| 02-20 | 2018-02-20 01:00:00 | 2018-02-20 12:59:59 | 38,478 | 2,977,911 | 0 |
| 02-21 | 2018-02-21 01:55:46 | 2018-02-21 10:43:21 | 3,256 | 4,056 | 0 |
| 02-22 | 1970-01-10 03:04:26 | 2018-02-22 12:59:59 | 32,891 | 380,319 | 0 |
| 02-23 | 2018-02-23 01:00:00 | 2018-02-23 12:59:59 | 33,325 | 378,363 | 0 |
| 02-28 | 2018-02-28 01:00:00 | 2018-02-28 12:59:59 | 31,855 | 237,535 | 0 |
| 03-01 | 2018-03-01 01:00:00 | 2018-03-01 12:59:59 | 32,069 | 37,824 | 0 |
| 03-02 | 2018-03-02 01:00:00 | 2018-03-02 12:59:59 | 31,425 | 292,106 | 0 |

### 7a. Timestamps are on a 12-hour clock with the AM/PM marker stripped — measured

- **The hour field never exceeds 12 in any file.** Full hour histogram for 02-14: `{01:65412, 02:170152, 03:102759, 04:28426, 05:11640, 07:2, 08:67293, 09:105574, 10:168069, 11:225353, 12:103895}`. Hours 13–23 and hour 00 are absent from all ten files.
- **Zero rows carry an AM/PM marker** (regex-checked across all distinct values in all files).

Read literally, every capture day begins at 01:00 and ends at 12:59:59 — a 12-hour span starting at 1 AM, which does not match a business-hours capture. Under the reading "hours 01–07 are PM", the days become:

| File | Reconstructed span |
|---|---|
| 02-14 | 08:28:20 → 17:30:48 |
| 02-23 | 08:15:09 → 17:30:46 |
| 03-01 | 08:16:28 → 17:45:28 |

which is a coherent ~09:00 workday. **This reconstruction is an inference, not a measurement** — the data contains no marker that proves which rows are PM. Hours 06 and 07 are the boundary cases and are ambiguous (02-20 has 6,278 rows at hour 06 and 3,100 at hour 07).

### 7b. File order is only approximately chronological

Run-length encoding the hour sequence in file order for 02-14 gives **340,456 runs over 1,048,570 rows, median run length 1**, with the hour sequence oscillating `8, 9, 8, 9, 8, 9, 10, 9, 10, 9, 10, 11, 10, 11 …`. The rows are locally interleaved throughout; a global chronological sort is mandatory before any temporal windowing.

### 7c. Timestamp granularity is 1 second, with massive tie groups

| File | Distinct seconds | Mean rows/sec | Median | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| 02-14 | 32,038 | 32.7 | 26 | 87 | 133 | 292 |
| 03-01 | 32,068 | 10.3 | 5 | 35 | 70 | 2,050 |

There is no sub-second field. Within a second, the true ordering of flows is unrecoverable.

### 7d. Corrupted 1970 timestamps

02-14 has 5 rows and 02-22 has 9 rows dated 1970. Located exactly (02-14, file lines 410,958–410,961 and 412,186):

```
Dst Port=0, Protocol=0, Timestamp=05/01/1970 03:01:17, Flow Duration=-11873000000,  Label=Benign
Dst Port=0, Protocol=0, Timestamp=08/01/1970 07:32:33, Flow Duration=-681402000000, Label=Benign
Dst Port=0, Protocol=0, Timestamp=12/01/1970 07:17:56, Flow Duration=-919011000000, Label=Benign
Dst Port=0, Protocol=0, Timestamp=12/01/1970 09:15:10, Flow Duration=-273850000000, Label=Benign
Dst Port=0, Protocol=0, Timestamp=12/01/1970 09:44:12, Flow Duration=-529801000000, Label=Benign
```

These have `Dst Port = 0`, `Protocol = 0` and large **negative** `Flow Duration` — CICFlowMeter integer-overflow artefacts. **These corrupt rows are retained in the Parquet copy**: `Bruteforce-…-14-02-2018.parquet` has `Flow Duration` min = −919,011,000,000, matching row 410,960 exactly, and `Web1-…-22-02-2018.parquet` has min = −828,220,000,000 from 02-22.

---

## 8. Data-quality findings (measured — not fixed)

### 8a. Repeated header rows embedded in the data — 59 rows

| File | Header rows in data | Locations |
|---|---:|---|
| 02-16-2018.csv | 1 | line 1,000,001 |
| 02-28-2018.csv | 33 | 21,840; 43,119; 63,293; … |
| 03-01-2018.csv | 25 | 415; 19,763; 19,908; … |

Every one of the 80 fields equals its own column name. These are the source of the 59 `Label`-valued labels in §6, and they make every numeric column non-numeric on those rows.

### 8b. `Infinity` and `NaN` tokens in `Flow Byts/s` and `Flow Pkts/s`

| File | `Flow Byts/s` = +Inf | `Flow Pkts/s` = +Inf | `Flow Byts/s` = NaN |
|---|---:|---:|---:|
| 02-14 | 1,547 | 3,824 | 2,277 |
| 02-15 | 3,106 | 8,027 | 4,921 |
| 02-20 | 22,686 | 59,453 | 36,767 |
| 02-22 | 2,041 | 5,610 | 3,569 |
| 02-23 | 1,954 | 5,708 | 3,754 |
| 02-28 | 2,128 | 6,169 | 4,041 |
| 03-01 | 1,085 | 2,919 | 1,834 |
| 03-02 | 1,492 | 4,050 | 2,558 |
| 02-16, 02-21 | 0 | 0 | 0 |

Zero negative infinities anywhere. These arise from zero-duration flows. **Naïve `pd.read_csv` will type these two columns as `object`, silently breaking downstream numeric code.**

### 8c. Duplicate rows — measured over all 10 CSVs

64-bit hash of the joined raw field strings; collision probability at n = 16.2 M is ~1e-6.

| File | Rows | Exact duplicates (all cols) | % | Duplicates ignoring Label | Label-conflicting |
|---|---:|---:|---:|---:|---:|
| 02-14 | 1,048,575 | 225,628 | **21.52 %** | 225,628 | 0 |
| 02-15 | 1,048,575 | 2,421 | 0.23 % | 2,421 | 0 |
| 02-16 | 1,048,575 | 147,586 | **14.07 %** | 147,586 | 0 |
| 02-20 | 7,948,748 | 2 | 0.00 % | 2 | 0 |
| 02-21 | 1,048,575 | 17,557 | 1.67 % | 17,557 | 0 |
| 02-22 | 1,048,575 | 3,278 | 0.31 % | 3,278 | 0 |
| 02-23 | 1,048,575 | 2,614 | 0.25 % | 2,614 | 0 |
| 02-28 | 613,104 | 6,121 | 1.00 % | 12,722 | **6,601** |
| 03-01 | 331,125 | 97 | 0.03 % | 102 | **5** |
| 03-02 | 1,048,575 | 5,459 | 0.52 % | 5,459 | 0 |
| **TOTAL** | **16,233,002** | **410,763** | **2.53 %** | | |

02-20 is nearly duplicate-free precisely because it retains `Flow ID / Src IP / Src Port / Dst IP`; the other files' duplicates are largely an artefact of dropping those identifier columns.

**Label conflicts** (identical feature vectors carrying different labels) occur in the two Infiltration CSVs and are far worse in the Parquet copies:

| Parquet file | Rows | Duplicates ignoring Label | Feature groups with >1 label | Rows involved |
|---|---:|---:|---:|---:|
| Infil1-…-28-02-2018 | 456,873 | 8,187 | **8,187** | **16,374** |
| Infil2-…-01-03-2018 | 249,170 | 8,577 | — | — |
| Botnet-…-02-03-2018 | 771,587 | 127 | — | — |
| Web1 / Web2 / Bruteforce / DoS1 | — | 3 / 4 / 7 / 49 | — | — |

Verified directly on Infil1: 456,873 rows → 448,686 distinct feature vectors → 8,187 groups where the same feature vector appears with more than one label. **This is irreducible label noise: no classifier can separate those 16,374 rows.** It also caps achievable accuracy on the Infiltration task.

### 8d. Negative values

| Column | Files affected | Distinct negative values | Interpretation |
|---|---|---|---|
| `Init Fwd Win Byts` | all 10 (118,201 – 2,653,104 rows each) | **only −1** | CICFlowMeter "not observed" sentinel |
| `Init Bwd Win Byts` | all 10 (159,458 – 4,209,795 rows each) | **only −1** | same |
| `Flow Duration`, `Flow IAT *`, `Fwd IAT *`, `Flow Pkts/s` | 02-14 (5 rows), 02-22 (9), 02-23 (1) | large negatives | overflow artefacts, §7d |

The `-1` sentinels are legitimate and must not be treated as measurement errors, but they will distort any scaler that assumes non-negative window sizes.

### 8e. Nulls, empties, malformed rows

- **0 empty strings** in any column of any CSV.
- **0 malformed rows** (wrong field count) in any CSV — Arrow's invalid-row handler fired zero times across all 16.2 M rows.
- **Parquet: 0 nulls, 0 NaN, 0 ±Inf in every column of every file** (verified from row-group statistics and a full numeric pass). The Parquet copy is clean on this axis because the offending rows were filtered out.

---

## 9. T = 5 temporal-window feasibility

**Architecture assessed (not implemented):** sort flows chronologically → cut into non-overlapping consecutive groups of 5 → each group is one graph → graph label = label of the 5th flow.

I measured this on three CSVs after (a) removing header-echo rows, (b) applying the hours-01–07-are-PM reconstruction, (c) a stable chronological sort.

| | 02-14 | 02-16 | 03-01 |
|---|---:|---:|---:|
| Usable rows | 1,048,570 | 1,048,574 | 331,100 |
| **Windows at T=5** | **209,714** | **209,714** | **66,220** |
| Rows dropped at tail | 0 | 4 | 0 |
| Label-pure windows | 191,414 (91.27 %) | 199,764 (95.26 %) | 59,451 (89.78 %) |
| Mixed-label windows | 18,300 (8.73 %) | 9,950 (4.74 %) | 6,769 (10.22 %) |
| **Windows spanning a single second** | **185,256 (88.34 %)** | **206,968 (98.69 %)** | **44,602 (67.35 %)** |
| Window time span: median / p95 / max | 0 s / 1 s / 11 s | 0 s / 0 s / 676 s | 0 s / 2 s / 46 s |
| 5th-flow label ≠ window majority | 0 / 20,000 (0.00 %) | 8 / 20,000 (0.04 %) | 674 / 20,000 (3.37 %) |

Graph-label distribution (02-14): Benign 133,502 · FTP-BruteForce 38,674 · SSH-Bruteforce 37,538 — proportional to the row-level distribution, so class balance is preserved.

Estimated corpus-wide window count at T=5, summing ⌊rows/5⌋ per file over the measured row counts (header echoes removed): **≈ 3,246,586 windows**.

### Verdict: the architecture is **feasible but not yet valid as specified.** Three measured problems.

**Problem 1 — "consecutive" is undefined for most windows.** Between 67 % and 99 % of windows fall entirely within a single 1-second timestamp. Within a tie group of up to 292 flows (02-14) or 2,050 (03-01), the sort order is decided by the tie-break rule, not by time. The "5th flow" is therefore an arbitrary choice among ties for the large majority of windows, and the graph label inherits that arbitrariness. **Two different sort implementations will produce different labels for the same data.** The measured p50 window span of 0 seconds is the core issue.

**Problem 2 — labelling by the 5th flow discards information and is unstable.** 4.7–10.2 % of windows are label-mixed. On 03-01 the 5th flow disagrees with the window majority in 3.37 % of cases. Combined with Problem 1, the label of a mixed window is effectively determined by tie-break order.

**Problem 3 — the timestamp itself is reconstructed, not read.** The 12-hour ambiguity (§7a) means the chronological sort depends on an assumption. If it is wrong for hours 06–07, those flows land ~12 hours away from their true position.

**Additional constraint:** Parquet has **no timestamp column**, so **the T=5 temporal architecture cannot be built from the Parquet files at all.** Only the CSVs support it.

**What the data does support:** the mechanics work. 3.2 M windows is ample; class balance survives windowing; the tail loss is ≤ 4 rows per file; every window is fully populated. Windows must not cross day boundaries (each file is one capture day; the gap between files is ~15 hours).

**What would need to be decided before implementing:** a deterministic, documented tie-break for same-second flows; whether the graph label should be the 5th flow's, the majority, or "attack if any flow is an attack"; and whether to accept or repair the 12-hour reconstruction.

---

## 10. Recommended development subset

### Group A — basic schema work and pipeline testing

**`data/raw/csv/03-01-2018.csv`** (107,842,858 B · 331,125 rows · 80 cols)
Smallest CSV; fits in memory; carries every hazard you need to handle — 25 embedded header rows, 1,085 + 2,919 Infinity tokens, 1,834 NaN tokens, 5 label conflicts, both `-1` window sentinels, 12-hour timestamps, 37,824 out-of-order pairs. Scan time 12 s. Balanced enough (Benign 238,037 / Infilteration 93,063) that a smoke test is meaningful.

**`data/raw/parquet/Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet`** (29,282,940 B · 249,170 rows)
Smallest Parquet, same day. Use it only to exercise the Parquet reader and to test the schema-harmonisation path — not for temporal work.

### Group B — actual multiclass pipeline development

Five CSVs giving 8 attack classes plus Benign, ~4.6 GiB, ~12.4 M rows:

| File | Rows | Adds |
|---|---:|---|
| 02-14-2018.csv | 1,048,575 | FTP-BruteForce, SSH-Bruteforce |
| 02-15-2018.csv | 1,048,575 | DoS-GoldenEye, DoS-Slowloris |
| 02-16-2018.csv | 1,048,575 | DoS-Hulk, DoS-SlowHTTPTest |
| 02-21-2018.csv | 1,048,575 | DDOS-HOIC, DDOS-LOIC-UDP |
| 03-02-2018.csv | 1,048,575 | Bot |
| 03-01-2018.csv | 331,125 | Infilteration |

Why CSV and not Parquet: the temporal architecture requires timestamps, which Parquet lacks; and Parquet destroys FTP-BruteForce (193,360 → 53) and DoS-SlowHTTPTest (139,890 → 55). Why these files: each contributes a distinct attack family, all are ≤ 0.36 GiB, all share the identical 80-column header, and together they cover 9 of the 15 classes with reasonable minority counts. Note 02-14 carries 21.5 % duplicates and 02-16 carries 14.1 % — worth measuring the effect of deduplication early rather than late.

### Group C — save for later, larger experiments

| File | Rows | Reason to defer |
|---|---:|---|
| **02-20-2018.csv** | 7,948,748 (3.78 GiB, **84 cols**) | Nearly half the corpus by itself; different schema needing a separate code path; 3.6 min just to scan. Its 4 identifier columns are valuable later (real duplicate-free flows, host-level graph edges), which is exactly why it deserves a dedicated phase. |
| **02-22-2018.csv, 02-23-2018.csv** | 1,048,575 each | The web-attack classes are extreme minorities (SQL Injection 34 and 53 rows). Anything learned here will be dominated by sampling noise. Bring in once the pipeline handles imbalance properly. |
| **02-28-2018.csv** | 613,104 | Second Infiltration day; adds 6,601 label conflicts and 33 header rows. Redundant with 03-01 for now. |
| **All 10 Parquet files** | 6,659,532 | No timestamps; per-file dtype inconsistency across 31 columns; two classes effectively destroyed. Useful later as a fast-loading, pre-cleaned tabular baseline to compare against the graph model. |

---

## 11. Report alignment

### Claim 1 — dataset volume

| | |
|---|---|
| **A. Report says** | Table 3.4.2, line 472: "About 16 million flow records" |
| **B. Files show** | CSV corpus: **16,233,002** data rows across 10 files. Parquet corpus: **6,659,532** rows. |
| **C. Discrepancy** | **The CSV figure matches the report almost exactly.** The Parquet figure is 2.4× smaller and does *not* match. The report's number is only reproducible from the CSVs. Additionally, 7 of 10 CSVs are truncated at exactly 2²⁰−1 rows, so the 16.2 M is itself a truncated count — the true CIC-IDS2018 corpus is larger. The agreement with "about 16 million" is therefore partly coincidental. |

### Claim 2 — feature count

| | |
|---|---|
| **A. Report says** | Line 194: "CIC-IDS2018 provides flows with 80 raw features"; Table 3.4.2 line 474: "80 CICFlowMeter-V3 features (bidirectional flow stats: durations, packet counts/sizes, bulk rates, window, flags, etc.)" |
| **B. Files show** | CSV: **80 total columns** = `Dst Port` + `Protocol` + `Timestamp` + **76 numeric features** + `Label`. CSV 02-20: 84 columns. Parquet: **78 columns** = `Protocol` + 76 features + `Label`. |
| **C. Discrepancy** | **The "80" is a column count, not a feature count.** The actual number of numeric flow features is **76**. Feeding all 80 columns to a model would mean feeding it a raw timestamp string and a destination-port integer as if they were flow statistics. The report's phrasing "80 … features" needs correcting to "80 columns, of which 76 are numeric flow features". The named feature families in the report (durations, packet counts/sizes, bulk rates, window, flags) all exist and are correctly described. |

### Claim 3 — attack scenarios

| | |
|---|---|
| **A. Report says** | Line 473: "Seven attack scenarios: Brute-force, Heartbleed, Botnet, DoS, DDoS, Web application attacks (e.g. SQLi, XSS), insider infiltration" |
| **B. Files show** | **14 attack labels in 6 families**: Brute-force (FTP-BruteForce, SSH-Bruteforce), Botnet (Bot), DoS (Hulk, GoldenEye, Slowloris, SlowHTTPTest), DDoS (HOIC, LOIC-HTTP, LOIC-UDP), Web (Brute Force -Web, Brute Force -XSS, SQL Injection), Infiltration (Infilteration). |
| **C. Discrepancy** | **Heartbleed is absent.** No Heartbleed label exists in any of the 20 files. Six of the seven named scenarios are present; the seventh is not. This is not a fault in the report's description of CIC-IDS2018 as published — it is a fact about *our* copy: the Heartbleed capture day is not in `data/raw/`. Any multiclass evaluation must be stated over 6 attack families / 14 attack classes, not 7. **I am flagging this rather than reconciling it — you may want to obtain the missing capture day, or amend the report.** |

### Claim 4 — dataset size on disk

| | |
|---|---|
| **A. Report says** | Line 197: "Approximately 50 GB is estimated for datasets" |
| **B. Files show** | CSV 6.41 GiB + Parquet 0.68 GiB = **7.09 GiB total**. |
| **C. Discrepancy** | Actual is **~14 % of the estimate**. This is a budget over-estimate, not an error, and it is favourable — but note it partly reflects the truncated CSVs and the missing capture days rather than efficient storage. |

### Claim 5 — ingestion method

| | |
|---|---|
| **A. Report says** | Line 216: "Load CSE-CIC-IDS2018 flow CSVs and labels via chunked streaming with per-class intake caps." |
| **B. Files show** | Chunked streaming is confirmed necessary and sufficient: 02-20 (3.78 GiB) scanned in 216 s at 64 MB blocks without memory pressure. Per-class caps are confirmed necessary: Benign is 83.07 % and the Benign : SQL-Injection ratio is 155,000 : 1. |
| **C. Discrepancy** | **None.** The report's stated ingestion strategy is validated by the data. Note the report says "CSVs", which is consistent with the finding that only the CSVs carry timestamps. |

### Claim 6 — temporal features

| | |
|---|---|
| **A. Report says** | Table 3.4.2: "Contains explicit temporal features (flow duration, inter-arrival, active/idle times)" |
| **B. Files show** | Present and correctly named: `Flow Duration`, `Flow IAT Mean/Std/Max/Min`, `Fwd IAT Tot/Mean/Std/Max/Min`, `Bwd IAT Tot/Mean/Std/Max/Min`, `Active Mean/Std/Max/Min`, `Idle Mean/Std/Max/Min` — 24 temporal features in both CSV and Parquet. |
| **C. Discrepancy** | **None on the feature list.** But an unstated gap: the report's temporal-graph design also needs a per-flow *wall-clock* timestamp, and that exists only in the CSVs, is on a 12-hour clock with no AM/PM marker, and has 1-second granularity with tie groups up to 2,050 flows. The report does not mention this. |

### Claim 7 — node types

| | |
|---|---|
| **A. Report says** | Line 205: five node types "Time, Packets, Binary, Bulk, Window" |
| **B. Files show** | Columns exist for all five groupings — Time (24 IAT/Active/Idle/Duration), Packets (packet counts and lengths), Binary (8 flag-count columns), Bulk (6 `*Byts/b Avg` / `*Blk Rate Avg`), Window (`Init Fwd/Bwd Win Byts`). |
| **C. Discrepancy** | **None in principle**, but two facts affect the grouping: the 6 Bulk columns are frequently all-zero in CICFlowMeter output (not yet quantified — see §12), and the 2 Window columns are dominated by the `-1` sentinel (up to 4.2 M rows in 02-20). A "Window" node built from two mostly-sentinel features may carry little signal. |

---

## 12. Risks and unknowns

**Risks (measured)**

1. **Parquet has no timestamps** — the entire temporal architecture is unavailable on that copy. Highest-impact finding in this report.
2. **12-hour clock ambiguity** — every temporal ordering decision rests on an unverifiable assumption about which rows are PM. Affects all 16.2 M CSV rows.
3. **1-second granularity with 26–2,050 flows per tie group** — makes "consecutive flows" ill-defined for 67–99 % of T=5 windows.
4. **Seven CSVs truncated at 2²⁰−1 rows** — the class distributions in those files are of a truncated prefix, not of the full capture day. Any prior computed from them is biased.
5. **Irreducible label noise** — 16,374 rows in Infil1 alone carry contradictory labels for identical feature vectors. Caps achievable accuracy on Infiltration.
6. **Extreme imbalance** — SQL Injection has 87 rows in 16.2 M (155,000:1). At T=5 windowing, entire classes may vanish from a validation split.
7. **410,763 duplicate CSV rows (2.53 %)**, concentrated in 02-14 (21.5 %) and 02-16 (14.1 %) — will leak between train and test splits if not handled before splitting.
8. **Infinity / NaN tokens in two columns** — will silently produce `object` dtype under naive `pd.read_csv`.
9. **59 embedded header rows** — will produce a 16th phantom class and non-numeric values.
10. **31 of 78 Parquet columns have inconsistent dtypes across files** — blocks naive concatenation.
11. **Heartbleed missing** — one of the seven scenarios the report names is not in our copy.
12. **02-20 has a different schema (84 cols)** — needs its own code path or column-subsetting.

**Unknowns (explicitly not determined)**

1. **The exact rule that produced the Parquet subset.** Neither zero-duration filtering nor deduplication (nor both) reproduces the row counts. Not recoverable from the files alone.
2. **Whether the 5 untested day-pairs (21, 22, 23 Feb, 28 Feb, 02 Mar) obey the same Parquet ⊂ CSV relationship.** 5 of 10 pairs verified; the rest assumed by analogy.
3. **Whether CSV and Parquet floats are bit-identical.** Would require a full-precision comparison of 6.66 M × 76 values — an expensive full scan that I did not run.
4. **Per-column constant/all-zero rate.** I did not measure how many of the 76 features are constant (the Bulk columns are the usual suspects). This needs a full pass and I have not run it — I am not going to state a number I have not measured.
5. **Per-column min/max/mean for the CSVs.** Measured for Parquet only. Requires a full numeric scan of 6.41 GiB.
6. **Whether the truncated files' missing rows are recoverable.** Depends on whether the original untruncated CSVs exist upstream.
7. **The correct AM/PM assignment for hours 06 and 07** (9,378 rows in 02-20). Genuinely ambiguous.
8. **Whether `Dst Port` should be a feature, a node identity, or dropped.** A schema question, not a data question.

---

## 13. Recommended next step

**Write a single, tested, read-only CSV loader for `data/raw/csv/03-01-2018.csv` that produces a clean in-memory DataFrame — and nothing else.**

Concretely, one function that:
1. reads the 80 columns with all dtypes forced explicitly (no inference),
2. drops the 25 embedded header rows,
3. coerces `Flow Byts/s` and `Flow Pkts/s` to float with `Infinity`/`NaN` preserved as `inf`/`nan` rather than strings,
4. parses `Timestamp` under an explicitly documented, configurable 12-hour reconstruction rule,
5. returns `(features_df, labels_series)` with the 76 numeric features and the label,
6. is covered by tests asserting the exact measured counts from this report — 331,125 raw rows, 25 header rows removed, 331,100 remaining, 3 label values, 1,085 + 2,919 infinities, 118,201 + 159,458 `-1` sentinels.

**Why this and not something larger:** every downstream component — feature extraction, HTGraph construction, windowing, the model — consumes the output of this one function. Building it against the smallest file, with assertions pinned to numbers measured here, means every later stage inherits verified inputs. It also forces the two decisions that everything else depends on (the timestamp reconstruction rule and the tie-break rule) to be made explicitly and in one place, rather than being buried as implicit behaviour in a sort call.

I have **not** started this. Confirm before I proceed.
