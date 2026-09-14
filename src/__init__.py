"""Dynamic Graph Optimization and Automated Feature Extraction for Efficient
Intrusion Detection in Network Systems.

Package layout
--------------
``config``        typed configuration loader (YAML + env overrides)
``utils``         logging / reproducibility helpers
``preprocessing`` dataset inspection now; cleaning + splitting after the real
                  CIC-IDS2018 schema has been inspected
``features``      automated feature extraction (later phase)
``graph``         heterogeneous temporal graph construction (later phase)
``models``        detection models (later phase)
``pruning``       attention-based dynamic graph pruning (later phase)
``training``      training loops (later phase)
``evaluation``    metrics, efficiency measurements, reporting (later phase)
"""

__version__ = "0.1.0"
