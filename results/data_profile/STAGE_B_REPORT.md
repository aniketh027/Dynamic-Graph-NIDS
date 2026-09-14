# Stage B - Multi-Day Intake and Global Dataset Profiling

Generated 2026-09-05T14:48:56.196846+00:00 | measurement digest `f7869d922df1d1d6` | elapsed 206.4s

Every number below was measured from the raw CSV files. Nothing was cleaned, replaced, deduplicated, encoded or written back: this stage **measures only**.

## 1. Corpus at a glance

```
csv dir              : C:\Project\ComputerNetworks\data\raw\csv
files profiled       : 10 (6.41 GiB)
capture dates        : 2018-02-14, 2018-02-15, 2018-02-16, 2018-02-20, 2018-02-21, 2018-02-22, 2018-02-23, 2018-02-28, 2018-03-01, 2018-03-02
raw data rows        : 16,233,002
embedded header rows : 59
usable rows          : 16,232,943
rows profiled        : 16,232,943
label vocabulary     : 15 classes
schema variants      : extended_84=1, standard_80=9
truncated files      : 7 (at the 1,048,575-row Excel cap)
elapsed              : 206.4s
```

<details><summary>Configuration used</summary>

```
config file          : C:\Project\ComputerNetworks\configs\default.yaml
project              : dynamic-graph-nids
seed                 : 42
dataset dir (input)  : C:\Project\ComputerNetworks\data\raw
output dir (processed): C:\Project\ComputerNetworks\data\processed
results / plots      : C:\Project\ComputerNetworks\results | C:\Project\ComputerNetworks\plots
checkpoints / logs   : C:\Project\ComputerNetworks\checkpoints | C:\Project\ComputerNetworks\logs
dataset file filter  : <all matching files>
debug mode           : False (row limit 100,000)
temporal window T    : 5
split ratios         : train=0.70 val=0.10 test=0.20
loader primary csv   : csv/03-01-2018.csv
loader cleaning      : nan->0.0 inf->column_finite_extremum
profiling            : raw/csv chunk=200,000 duplicates=True/global=True
```

</details>

## 2. File discovery and the deterministic capture-order proxy

Files were discovered recursively under `C:\Project\ComputerNetworks\data\raw\csv` by glob, not by hard-coded name. Each file's capture date comes from the `MM-DD-YYYY` digits in its own filename, cross-checked against the modal date in its own `Timestamp` column. Files are then sorted by `(capture_date, source_file)`, so the sequence never depends on the order the filesystem happens to return.

**Ordering contract**: deterministic capture-order proxy = source-file capture date, then Original_Index (original row position within that file).

This is a *deterministic capture-order proxy*, not perfect physical chronology - see section 10. Rows are addressed as `(source_file, source_file_date, original_index)`, where `original_index` is the 0-based position among the source file's data rows, assigned before any row is excluded.

| seq | file | capture date (filename) | date source | modal Timestamp date | size (bytes) | GiB |
|---|---|---|---|---|---|---|
| 0 | 02-14-2018.csv | 2018-02-14 | filename | 2018-02-14 | 358,223,333 | 0.33 |
| 1 | 02-15-2018.csv | 2018-02-15 | filename | 2018-02-15 | 375,945,899 | 0.35 |
| 2 | 02-16-2018.csv | 2018-02-16 | filename | 2018-02-16 | 333,723,605 | 0.31 |
| 3 | 02-20-2018.csv | 2018-02-20 | filename | 2018-02-20 | 4,054,925,350 | 3.78 |
| 4 | 02-21-2018.csv | 2018-02-21 | filename | 2018-02-21 | 328,893,673 | 0.31 |
| 5 | 02-22-2018.csv | 2018-02-22 | filename | 2018-02-22 | 382,636,202 | 0.36 |
| 6 | 02-23-2018.csv | 2018-02-23 | filename | 2018-02-23 | 382,840,456 | 0.36 |
| 7 | 02-28-2018.csv | 2018-02-28 | filename | 2018-02-28 | 209,249,758 | 0.19 |
| 8 | 03-01-2018.csv | 2018-03-01 | filename | 2018-03-01 | 107,842,858 | 0.10 |
| 9 | 03-02-2018.csv | 2018-03-02 | filename | 2018-03-02 | 352,368,373 | 0.33 |

## 3. Schema validation

Standard layout: **80 columns** = `Dst Port` + `Protocol` + `Timestamp` + 76 numeric flow features + `Label`. Model features = **78** (the 76 flow features plus `Dst Port` and `Protocol`); `Timestamp` and `Label` are metadata and target, not features.

- `extended_84`: 1 file(s) - 02-20-2018.csv
- `standard_80`: 9 file(s) - 02-14-2018.csv, 02-15-2018.csv, 02-16-2018.csv, 02-21-2018.csv, 02-22-2018.csv, 02-23-2018.csv, 02-28-2018.csv, 03-01-2018.csv, 03-02-2018.csv

| file | cols | variant | EOL | md5 (stripped) | md5 (raw bytes) | extra columns | missing columns |
|---|---|---|---|---|---|---|---|
| 02-14-2018.csv | 80 | standard_80 | LF | `d8434b53de72` | `64a4bae24658` | - | - |
| 02-15-2018.csv | 80 | standard_80 | LF | `d8434b53de72` | `64a4bae24658` | - | - |
| 02-16-2018.csv | 80 | standard_80 | LF | `d8434b53de72` | `64a4bae24658` | - | - |
| 02-20-2018.csv | 84 | extended_84 | CRLF | `7a20eb9a113f` | `8062b6f49c24` | Flow ID, Src IP, Src Port, Dst IP | - |
| 02-21-2018.csv | 80 | standard_80 | LF | `d8434b53de72` | `64a4bae24658` | - | - |
| 02-22-2018.csv | 80 | standard_80 | LF | `d8434b53de72` | `64a4bae24658` | - | - |
| 02-23-2018.csv | 80 | standard_80 | LF | `d8434b53de72` | `64a4bae24658` | - | - |
| 02-28-2018.csv | 80 | standard_80 | LF | `d8434b53de72` | `64a4bae24658` | - | - |
| 03-01-2018.csv | 80 | standard_80 | LF | `d8434b53de72` | `64a4bae24658` | - | - |
| 03-02-2018.csv | 80 | standard_80 | CRLF | `d8434b53de72` | `e6e611954444` | - | - |

The two md5 columns answer different questions. The *stripped* md5 hashes the header line with the line terminator removed, so it says whether the column **names** match. The *raw bytes* md5 includes the terminator, so it also distinguishes LF from CRLF files.

**84-column layout** (02-20-2018.csv): the standard 80 columns with `Flow ID`, `Src IP`, `Src Port`, `Dst IP` prepended - a strict superset, verified positionally. All 78 model features are present there unchanged, so the file is read by column *name*. The 4 extra columns are **reported, not silently discarded**: they are excluded from the model feature set by explicit decision (a flow ID and raw endpoint identity would leak host identity and cannot generalise), and they remain untouched in the raw file.

## 4. Multi-day intake and row accounting

Each file is read twice and never held whole in memory. **Pass 1** reads only the `Label` column to get the data-row count, the exact positions of the embedded header rows and the label counts. **Pass 2** then reads the 80 standard columns in chunks of 200,000 rows, with the 78 feature dtypes forced to `float64` and the known header lines excluded via `skiprows`.

Peak memory is one chunk (~119 MB at 200,000 x 78 float64) plus the duplicate-hash accumulators (26 bytes per corpus row), instead of the roughly 10 GB a concatenated feature frame for the whole corpus would need. The corpus is never concatenated into a single DataFrame.

Forcing the dtypes is not only about memory. Under inference the same column can come back as `int64` in one chunk and `float64` in another, and `hash(5) != hash(5.0)` - which would silently corrupt the duplicate counts at chunk boundaries. A chunk that contained an embedded header row would also turn all 78 columns into Python objects (roughly 1 GB per chunk). Locating those rows in pass 1 removes both hazards.

Row counts are cross-checked three ways: the parsed data-row count against the file's physical line count, pass 2's row count against pass 1's prediction, and pass 2's label counts against pass 1's.

| file | capture date | raw data rows | physical lines | embedded headers | usable rows | rows profiled | at Excel cap |
|---|---|---|---|---|---|---|---|
| 02-14-2018.csv | 2018-02-14 | 1,048,575 | 1,048,576 | 0 | 1,048,575 | 1,048,575 | YES |
| 02-15-2018.csv | 2018-02-15 | 1,048,575 | 1,048,576 | 0 | 1,048,575 | 1,048,575 | YES |
| 02-16-2018.csv | 2018-02-16 | 1,048,575 | 1,048,576 | 1 | 1,048,574 | 1,048,574 | YES |
| 02-20-2018.csv | 2018-02-20 | 7,948,748 | 7,948,749 | 0 | 7,948,748 | 7,948,748 | no |
| 02-21-2018.csv | 2018-02-21 | 1,048,575 | 1,048,576 | 0 | 1,048,575 | 1,048,575 | YES |
| 02-22-2018.csv | 2018-02-22 | 1,048,575 | 1,048,576 | 0 | 1,048,575 | 1,048,575 | YES |
| 02-23-2018.csv | 2018-02-23 | 1,048,575 | 1,048,576 | 0 | 1,048,575 | 1,048,575 | YES |
| 02-28-2018.csv | 2018-02-28 | 613,104 | 613,105 | 33 | 613,071 | 613,071 | no |
| 03-01-2018.csv | 2018-03-01 | 331,125 | 331,126 | 25 | 331,100 | 331,100 | no |
| 03-02-2018.csv | 2018-03-02 | 1,048,575 | 1,048,576 | 0 | 1,048,575 | 1,048,575 | YES |
| **TOTAL** | 10 files | **16,233,002** |  | 59 | **16,232,943** | **16,232,943** | 7 file(s) |

Embedded header rows are data rows whose `Label` field is the literal string `"Label"` - repeated header lines concatenated into the data. They are excluded from the profiled rows but **not removed from the raw files**, which are never modified.

## 5. Global label vocabulary

**15 distinct labels** across 16,232,943 usable rows. Original spelling and casing are preserved exactly - including the dataset's own `Infilteration` misspelling. No label is encoded and no class mapping is persisted in this stage.

| label | rows | share | capture dates | first seen | last seen |
|---|---|---|---|---|---|
| Benign | 13,484,708 | 83.0700% | 10 | 2018-02-14 | 2018-03-02 |
| DDOS attack-HOIC | 686,012 | 4.2260% | 1 | 2018-02-21 | 2018-02-21 |
| DDoS attacks-LOIC-HTTP | 576,191 | 3.5495% | 1 | 2018-02-20 | 2018-02-20 |
| DoS attacks-Hulk | 461,912 | 2.8455% | 1 | 2018-02-16 | 2018-02-16 |
| Bot | 286,191 | 1.7630% | 1 | 2018-03-02 | 2018-03-02 |
| FTP-BruteForce | 193,360 | 1.1912% | 1 | 2018-02-14 | 2018-02-14 |
| SSH-Bruteforce | 187,589 | 1.1556% | 1 | 2018-02-14 | 2018-02-14 |
| Infilteration | 161,934 | 0.9976% | 2 | 2018-02-28 | 2018-03-01 |
| DoS attacks-SlowHTTPTest | 139,890 | 0.8618% | 1 | 2018-02-16 | 2018-02-16 |
| DoS attacks-GoldenEye | 41,508 | 0.2557% | 1 | 2018-02-15 | 2018-02-15 |
| DoS attacks-Slowloris | 10,990 | 0.0677% | 1 | 2018-02-15 | 2018-02-15 |
| DDOS attack-LOIC-UDP | 1,730 | 0.0107% | 1 | 2018-02-21 | 2018-02-21 |
| Brute Force -Web | 611 | 0.0038% | 2 | 2018-02-22 | 2018-02-23 |
| Brute Force -XSS | 230 | 0.0014% | 2 | 2018-02-22 | 2018-02-23 |
| SQL Injection | 87 | 0.0005% | 2 | 2018-02-22 | 2018-02-23 |

Per-file label counts are in `class_by_day.csv` and `per_file_summary.json`; the wide per-label view is in `class_distribution.csv`.

## 6. Class-by-time analysis

CIC-IDS2018 is time-segregated by design: each attack was executed on a particular day, so a class's rows are concentrated in one or two capture dates rather than spread across the corpus.

- Classes present on exactly one capture date: **10** of 15 - Bot, DDOS attack-HOIC, DDOS attack-LOIC-UDP, DDoS attacks-LOIC-HTTP, DoS attacks-GoldenEye, DoS attacks-Hulk, DoS attacks-SlowHTTPTest, DoS attacks-Slowloris, FTP-BruteForce, SSH-Bruteforce
- Classes whose first appearance is in the final third of the date sequence: **2** - Bot, Infilteration

| label | 2018-02-14 | 2018-02-15 | 2018-02-16 | 2018-02-20 | 2018-02-21 | 2018-02-22 | 2018-02-23 | 2018-02-28 | 2018-03-01 | 2018-03-02 |
|---|---|---|---|---|---|---|---|---|---|---|
| Benign | 667,626 | 996,077 | 446,772 | 7,372,557 | 360,833 | 1,048,213 | 1,048,009 | 544,200 | 238,037 | 762,384 |
| DDOS attack-HOIC | 0 | 0 | 0 | 0 | 686,012 | 0 | 0 | 0 | 0 | 0 |
| DDoS attacks-LOIC-HTTP | 0 | 0 | 0 | 576,191 | 0 | 0 | 0 | 0 | 0 | 0 |
| DoS attacks-Hulk | 0 | 0 | 461,912 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 286,191 |
| FTP-BruteForce | 193,360 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SSH-Bruteforce | 187,589 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Infilteration | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 68,871 | 93,063 | 0 |
| DoS attacks-SlowHTTPTest | 0 | 0 | 139,890 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DoS attacks-GoldenEye | 0 | 41,508 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DoS attacks-Slowloris | 0 | 10,990 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DDOS attack-LOIC-UDP | 0 | 0 | 0 | 0 | 1,730 | 0 | 0 | 0 | 0 | 0 |
| Brute Force -Web | 0 | 0 | 0 | 0 | 0 | 249 | 362 | 0 | 0 | 0 |
| Brute Force -XSS | 0 | 0 | 0 | 0 | 0 | 79 | 151 | 0 | 0 | 0 |
| SQL Injection | 0 | 0 | 0 | 0 | 0 | 34 | 53 | 0 | 0 | 0 |

### Naive chronological 70/10/20 split - coverage check

Cut at exact row indices in the deterministic sequence (70% / 10% / 20% of 16,232,943 usable rows): train `[0, 11,363,060)`, validation `[11,363,060, 12,986,354)`, test `[12,986,354, 16,232,943)`.

- `train_end` falls in 02-21-2018.csv @ row 268,588
- `validation_end` falls in 02-22-2018.csv @ row 843,307

**5 class(es) would be absent from the training period**: Bot, Brute Force -Web, Brute Force -XSS, Infilteration, SQL Injection. A model trained on that split cannot learn them, and any metric that averages over classes would be reporting on classes it never saw.

- Absent from validation: Bot, DDOS attack-LOIC-UDP, DDoS attacks-LOIC-HTTP, DoS attacks-GoldenEye, DoS attacks-Hulk, DoS attacks-SlowHTTPTest, DoS attacks-Slowloris, FTP-BruteForce, Infilteration, SSH-Bruteforce
- Absent from test: DDOS attack-HOIC, DDOS attack-LOIC-UDP, DDoS attacks-LOIC-HTTP, DoS attacks-GoldenEye, DoS attacks-Hulk, DoS attacks-SlowHTTPTest, DoS attacks-Slowloris, FTP-BruteForce, SSH-Bruteforce
- Present only in train (never evaluated): DDOS attack-LOIC-UDP, DDoS attacks-LOIC-HTTP, DoS attacks-GoldenEye, DoS attacks-Hulk, DoS attacks-SlowHTTPTest, DoS attacks-Slowloris, FTP-BruteForce, SSH-Bruteforce

**This is an analysis only.** Stage B does not change the split strategy and does not use random stratification; the chronological split is kept because a temporal intrusion-detection model must not be evaluated on rows that precede its training data.

## 7. Non-finite values

Corpus totals: **59,721 NaN**, **131,799 +inf**, **0 -inf** across the 78 model feature columns.

| file | NaN | +inf | -inf | affected columns | outside the known rate columns |
|---|---|---|---|---|---|
| 02-14-2018.csv | 2,277 | 5,371 | 0 | Flow Byts/s, Flow Pkts/s | none |
| 02-15-2018.csv | 4,921 | 11,133 | 0 | Flow Byts/s, Flow Pkts/s | none |
| 02-16-2018.csv | 0 | 0 | 0 | - | none |
| 02-20-2018.csv | 36,767 | 82,139 | 0 | Flow Byts/s, Flow Pkts/s | none |
| 02-21-2018.csv | 0 | 0 | 0 | - | none |
| 02-22-2018.csv | 3,569 | 7,651 | 0 | Flow Byts/s, Flow Pkts/s | none |
| 02-23-2018.csv | 3,754 | 7,662 | 0 | Flow Byts/s, Flow Pkts/s | none |
| 02-28-2018.csv | 4,041 | 8,297 | 0 | Flow Byts/s, Flow Pkts/s | none |
| 03-01-2018.csv | 1,834 | 4,004 | 0 | Flow Byts/s, Flow Pkts/s | none |
| 03-02-2018.csv | 2,558 | 5,542 | 0 | Flow Byts/s, Flow Pkts/s | none |

`Flow Byts/s`, `Flow Pkts/s` are the expected sources: CICFlowMeter divides by the flow duration, so a zero-duration flow yields `Infinity` (bytes moved in no time) or `NaN` (0/0).

**No non-finite value was replaced in Stage B.** The replacement values must be derived from the TRAIN split alone and then applied unchanged to validation and test; deriving them from the whole corpus would leak information from the evaluation period into training. That is a Stage C task.

## 8. Negative and sentinel values

| file | negative values | per column | outside the known sentinel columns |
|---|---|---|---|
| 02-14-2018.csv | 566,675 | Flow Duration (5), Flow IAT Max (2), Flow IAT Mean (5), Flow IAT Min (5), Flow Pkts/s (5), Fwd IAT Max (2), Fwd IAT Mean (5), Fwd IAT Min (5), Fwd IAT Tot (5), Init Bwd Win Byts (347,370), Init Fwd Win Byts (219,266) | Flow Duration (5), Flow IAT Max (2), Flow IAT Mean (5), Flow IAT Min (5), Flow Pkts/s (5), Fwd IAT Max (2), Fwd IAT Mean (5), Fwd IAT Min (5), Fwd IAT Tot (5) |
| 02-15-2018.csv | 951,673 | Init Bwd Win Byts (587,584), Init Fwd Win Byts (364,089) | none |
| 02-16-2018.csv | 438,522 | Init Bwd Win Byts (438,345), Init Fwd Win Byts (177) | none |
| 02-20-2018.csv | 6,862,899 | Init Bwd Win Byts (4,209,795), Init Fwd Win Byts (2,653,104) | none |
| 02-21-2018.csv | 529,705 | Init Bwd Win Byts (525,993), Init Fwd Win Byts (3,712) | none |
| 02-22-2018.csv | 900,187 | Flow Duration (9), Flow IAT Max (1), Flow IAT Mean (9), Flow IAT Min (9), Flow Pkts/s (9), Fwd IAT Max (1), Fwd IAT Mean (9), Fwd IAT Min (9), Fwd IAT Tot (9), Init Bwd Win Byts (577,969), Init Fwd Win Byts (322,153) | Flow Duration (9), Flow IAT Max (1), Flow IAT Mean (9), Flow IAT Min (9), Flow Pkts/s (9), Fwd IAT Max (1), Fwd IAT Mean (9), Fwd IAT Min (9), Fwd IAT Tot (9) |
| 02-23-2018.csv | 881,635 | Flow IAT Min (1), Fwd IAT Min (1), Init Bwd Win Byts (573,165), Init Fwd Win Byts (308,468) | Flow IAT Min (1), Fwd IAT Min (1) |
| 02-28-2018.csv | 548,845 | Init Bwd Win Byts (321,490), Init Fwd Win Byts (227,355) | none |
| 03-01-2018.csv | 277,659 | Init Bwd Win Byts (159,458), Init Fwd Win Byts (118,201) | none |
| 03-02-2018.csv | 730,463 | Init Bwd Win Byts (514,380), Init Fwd Win Byts (216,083) | none |

`Init Fwd Win Byts`, `Init Bwd Win Byts` use `-1` as CICFlowMeter's *"not observed"* sentinel: the TCP window size was never seen in that direction. That is genuine information ('no traffic in this direction'), semantically distinct from a window size of 0.

**Nothing was replaced or clipped.** Per-column detail, including whether every negative in a column is exactly the sentinel, is in `negative_summary.csv`.

## 9. Duplicate analysis

Three categories, measured on the 78 model feature columns:

- **A. exact** - identical features *and* raw `Timestamp` *and* `Label`.
- **B. temporal** - identical features and label, different `Timestamp` (the same flow signature recurring at another moment).
- **C. conflicting** - one feature vector carrying more than one label. These are the label-noise cases: no classifier can separate them.

| scope | rows | distinct feature vectors | A exact | B temporal | A+B | A+B share | C groups | C rows |
|---|---|---|---|---|---|---|---|---|
| 02-14-2018.csv | 1,048,575 | 674,808 | 225,628 | 148,132 | 373,760 | 35.645% | 7 | 169,775 |
| 02-15-2018.csv | 1,048,575 | 884,658 | 2,421 | 161,496 | 163,917 | 15.632% | 0 | 0 |
| 02-16-2018.csv | 1,048,574 | 591,901 | 147,586 | 309,087 | 456,673 | 43.552% | 0 | 0 |
| 02-20-2018.csv | 7,948,748 | 6,120,535 | 22,490 | 1,805,723 | 1,828,213 | 23.000% | 0 | 0 |
| 02-21-2018.csv | 1,048,575 | 561,405 | 17,557 | 469,613 | 487,170 | 46.460% | 0 | 0 |
| 02-22-2018.csv | 1,048,575 | 906,923 | 3,278 | 138,371 | 141,649 | 13.509% | 3 | 8 |
| 02-23-2018.csv | 1,048,575 | 910,658 | 2,614 | 135,299 | 137,913 | 13.152% | 4 | 16 |
| 02-28-2018.csv | 613,071 | 508,780 | 6,089 | 88,812 | 94,901 | 15.480% | 9,390 | 51,995 |
| 03-01-2018.csv | 331,100 | 282,656 | 73 | 40,144 | 40,217 | 12.146% | 8,227 | 44,625 |
| 03-02-2018.csv | 1,048,575 | 823,732 | 5,459 | 219,384 | 224,843 | 21.443% | 0 | 0 |
| __CORPUS__ | 16,232,943 | 11,967,162 | 433,195 | 3,810,208 | 4,243,403 | 26.141% | 22,373 | 423,390 |

Corpus-wide label combinations found on a single feature vector:

| label combination | feature groups | rows |
|---|---|---|
| DoS attacks-SlowHTTPTest + FTP-BruteForce + SSH-Bruteforce | 5 | 225,705 |
| Benign + Infilteration | 22,348 | 170,849 |
| DoS attacks-SlowHTTPTest + FTP-BruteForce | 7 | 14,051 |
| FTP-BruteForce + SSH-Bruteforce | 2 | 12,742 |
| Brute Force -Web + SQL Injection | 5 | 24 |
| Benign + Brute Force -Web | 3 | 11 |
| Benign + SQL Injection | 1 | 4 |
| Benign + Brute Force -XSS | 1 | 2 |
| Brute Force -XSS + SQL Injection | 1 | 2 |

**No duplicate or conflicting row was removed.** The exclusion policy belongs in Stage C, and it has to be decided with the temporal split in mind: removing category B would destroy the recurrence structure a temporal model is meant to learn, and any removal must not delete rows from the evaluation period on the basis of rows in the training period.

## 10. Timestamp analysis

| file | parsed | unparsed | distinct | min | max | AM/PM markers | largest tie group | mean tie group | decreasing steps |
|---|---|---|---|---|---|---|---|---|---|
| 02-14-2018.csv | 1,048,575 | 0 | 32,043 | 1970-01-05T03:01:17 | 2018-02-14T12:59:59 | 0 | 292 | 32.7 | 289765 |
| 02-15-2018.csv | 1,048,575 | 0 | 33,862 | 2018-02-15T01:00:00 | 2018-02-15T12:59:59 | 0 | 728 | 31.0 | 382967 |
| 02-16-2018.csv | 1,048,574 | 0 | 3,177 | 2018-02-16T01:00:32 | 2018-02-16T12:58:24 | 0 | 8403 | 330.1 | 268941 |
| 02-20-2018.csv | 7,948,748 | 0 | 38,478 | 2018-02-20T01:00:00 | 2018-02-20T12:59:59 | 0 | 850 | 206.6 | 2977911 |
| 02-21-2018.csv | 1,048,575 | 0 | 3,256 | 2018-02-21T01:55:46 | 2018-02-21T10:43:21 | 0 | 815 | 322.0 | 4056 |
| 02-22-2018.csv | 1,048,575 | 0 | 32,891 | 1970-01-10T03:04:26 | 2018-02-22T12:59:59 | 0 | 449 | 31.9 | 380319 |
| 02-23-2018.csv | 1,048,575 | 0 | 33,325 | 2018-02-23T01:00:00 | 2018-02-23T12:59:59 | 0 | 387 | 31.5 | 378363 |
| 02-28-2018.csv | 613,071 | 0 | 31,854 | 2018-02-28T01:00:00 | 2018-02-28T12:59:59 | 0 | 3055 | 19.2 | 237535 |
| 03-01-2018.csv | 331,100 | 0 | 32,068 | 2018-03-01T01:00:00 | 2018-03-01T12:59:59 | 0 | 2050 | 10.3 | 37824 |
| 03-02-2018.csv | 1,048,575 | 0 | 31,425 | 2018-03-02T01:00:00 | 2018-03-02T12:59:59 | 0 | 362 | 33.4 | 292106 |

Format `%d/%m/%Y %H:%M:%S`. Hours observed across the corpus: `01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12`. The hour field only ever holds 01-12 and **no AM/PM marker appears anywhere**, so a value like `01:15:00` could be 01:15 or 13:15. That ambiguity is real and is **not reconstructed here** - inventing an AM/PM rule would fabricate an ordering the data does not contain.

Timestamps are also heavily tied: the largest same-second group holds 8,403 rows, so the timestamp alone cannot order rows within a second even if the clock were unambiguous.

**14 row(s) carry a timestamp outside their file's capture date** and explain the pre-2018 minimums in the table above: 02-14-2018.csv 5 row(s), e.g. `05/01/1970 03:01:17`; 02-22-2018.csv 9 row(s), e.g. `10/01/1970 03:04:26`. These are a rounding error against 16,232,943 rows, but they are **not** dropped or corrected here - section 8 shows the same rows also carry negative durations and inter-arrival times, so they are one corrupt-flow population for Stage C to rule on, not two separate anomalies.

**`Timestamp` is therefore diagnostic metadata only.** Rows are never sorted by it; ordering uses the capture-order proxy from section 2. The raw values are preserved unmodified.

## 11. Dataset size and truncation

7 of 10 files hold exactly **1,048,575** data rows - that is 2^20 - 1, the Excel worksheet row limit (1,048,576 rows including the header). Files landing on that exact number were almost certainly opened and re-saved in a spreadsheet, silently discarding every row past the limit.

Affected: `02-14-2018.csv`, `02-15-2018.csv`, `02-16-2018.csv`, `02-21-2018.csv`, `02-22-2018.csv`, `02-23-2018.csv`, `03-02-2018.csv`

**No attempt is made to repair or re-obtain the missing rows in this stage.** The consequence to carry forward is that per-day row counts are capped, not complete, so a day's true class balance may differ from what is measured here.

## 12. Parquet copy (metadata only)

10 Parquet files, 691 MB, 6,659,532 rows total. Read from the Parquet footers only - no column data was loaded.

| file | capture date | rows | cols | has Timestamp | has Label | cols missing vs CSV |
|---|---|---|---|---|---|---|
| Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet | 2018-02-14 | 619,346 | 78 | **no** | yes | 50 |
| DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet | 2018-02-15 | 794,812 | 78 | **no** | yes | 50 |
| DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter.parquet | 2018-02-16 | 591,873 | 78 | **no** | yes | 50 |
| DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet | 2018-02-20 | 954,846 | 78 | **no** | yes | 50 |
| DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet | 2018-02-21 | 561,396 | 78 | **no** | yes | 50 |
| Web1-Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet | 2018-02-22 | 830,224 | 78 | **no** | yes | 50 |
| Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter.parquet | 2018-02-23 | 829,405 | 78 | **no** | yes | 50 |
| Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet | 2018-02-28 | 456,873 | 78 | **no** | yes | 50 |
| Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet | 2018-03-01 | 249,170 | 78 | **no** | yes | 50 |
| Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet | 2018-03-02 | 771,587 | 78 | **no** | yes | 50 |

The Parquet copy holds 6,659,532 rows against the CSV corpus's 16,232,943 usable rows, and **no Parquet file carries the `Timestamp` column**.

**The CSV corpus is the primary temporal source for this project.** Without `Timestamp` the Parquet copy cannot support the temporal HTGraph pipeline at all, and it also covers fewer rows. It is compared here on metadata only, as required, and is never read by the pipeline. Both copies are left untouched - nothing was converted in either direction.

## 13. Alignment with the project report

Each row separates **A. what the report requires**, **B. what the files actually contain**, and **C. the engineering decision**. Discrepancies are recorded, not silently fixed in either direction.

| topic | A. report requirement | B. measured fact | C. decision |
|---|---|---|---|
| Corpus size | approximately 16 million flow records | 16,233,002 raw data rows, 16,232,943 usable after excluding 59 embedded header rows | consistent with the report; the usable count is what every later stage quotes |
| Raw feature count | 80 raw features | 80 CSV **columns**, of which 76 are numeric flow features; `Timestamp` is metadata and `Label` is the target | the report's '80 features' is the column count, not the feature count - the report wording should say 80 columns |
| Model feature count | 76 flow features plus Dst Port and Protocol | 78 model feature columns, all present in every usable file | 78 model features confirmed |
| Schema uniformity | a single uniform schema across day files | 9 file(s) with 80 columns, 1 with 84 | read by column name; the extended file's 4 extra identifier columns are reported and excluded from the feature set, not dropped from the raw file |
| Timestamp availability | timestamps available for temporal graph construction | present in every CSV file; 16,232,943 of 16,232,943 rows parse (0 fail), but the clock is 12-hour with no AM/PM marker and heavily tied | Timestamp is diagnostic metadata; ordering uses the deterministic capture-order proxy, and the report must not claim exact wall-clock chronology |
| Class coverage | the CSE-CIC-IDS2018 attack taxonomy | 15 distinct labels measured | labels kept in the dataset's own spelling, including the `Infilteration` misspelling, so counts stay traceable to the files |
| Heartbleed | listed among the attack scenarios | **absent from all files** | scope reduction must be stated in the report; no samples are fabricated |
| Chronological 70/10/20 split | sequential 70/10/20 split, no random shuffling | 5 class(es) absent from the training period, 9 absent from test | the split strategy is unchanged in this stage; the coverage gap is recorded for Stage C to address without resorting to random stratification |
| File completeness | full daily captures | 7 file(s) hold exactly 1,048,575 rows (the Excel row cap) | treated as truncated; not repaired in this stage, and per-day counts are reported as measured |

### Attack-scenario coverage

| report attack family | status | measured labels | rows |
|---|---|---|---|
| Brute Force (FTP/SSH) | present | FTP-BruteForce, SSH-Bruteforce | 380,949 |
| DoS | present | DDoS attacks-LOIC-HTTP, DoS attacks-GoldenEye, DoS attacks-Hulk, DoS attacks-SlowHTTPTest, DoS attacks-Slowloris | 1,230,491 |
| DDoS | present | DDOS attack-HOIC, DDOS attack-LOIC-UDP, DDoS attacks-LOIC-HTTP | 1,263,933 |
| Web attacks | present | Brute Force -Web, Brute Force -XSS, SQL Injection | 928 |
| Infiltration | present | Infilteration | 161,934 |
| Botnet | present | Bot | 286,191 |
| Heartbleed | **ABSENT** | - | 0 |

### Heartbleed

Searched every label in all 10 discovered CSV files. **Heartbleed is ABSENT from every label in every discovered CSV file. No Heartbleed sample is fabricated and no claim of full attack-scenario coverage is made.**

The report's Heartbleed scenario therefore cannot be evaluated with this corpus. No Heartbleed rows are fabricated, and no claim is made that all report attack scenarios are covered. The honest options are to state the reduced scope explicitly in the report, or to obtain the missing capture file - a decision for you, not for this stage.

## 14. Open issues

- 03-01-2018.csv: the last usable row sits at Original_Index 331111, not 331124 - the final data row is an embedded header line.

## 15. Artifacts

Written to `C:\Project\ComputerNetworks\results\data_profile`:

- `per_file_summary.json`
- `global_dataset_summary.json`
- `schema_summary.json`
- `class_distribution.csv`
- `class_by_day.csv`
- `duplicate_summary.csv`
- `nonfinite_summary.csv`
- `negative_summary.csv`
- `STAGE_B_REPORT.md`

No cleaned dataset copy, feature matrix or train/validation/test array was written, and nothing was written under `data/raw`.
