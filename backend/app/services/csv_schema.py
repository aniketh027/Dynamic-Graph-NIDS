"""Read-only C3B descriptor derived from the frozen schema and C2A parser.

No model/state loading and no mutation of csv.field_size_limit. The HTTP
parser's process-global current limit is part of the descriptor identity.
"""
import csv

from backend.app.services import csv_replay
from src.graph.schema import canonical_sha256
from src.preprocessing.schema import CSV_COLUMNS, CSV_COLUMNS_EXTENDED, MODEL_FEATURE_COLUMNS

CONTRACT_VERSION = "graphnids-c3b-browser-csv-v1"


def csv_descriptor():
    record = {
        "contract_version": CONTRACT_VERSION,
        "canonical_columns": list(MODEL_FEATURE_COLUMNS),
        "profiles": {
            name: {
                "columns": list(columns),
                "projection_indices": [columns.index(feature) for feature in MODEL_FEATURE_COLUMNS],
                "removed_metadata_fields": [column for column in columns if column not in MODEL_FEATURE_COLUMNS],
            }
            for name, columns in (("standard_80", CSV_COLUMNS), ("extended_84", CSV_COLUMNS_EXTENDED))
        },
        "feature_order_sha256": canonical_sha256(MODEL_FEATURE_COLUMNS),
        "numeric": {
            "grammar_identity": "c2a-decimal-ieee-full-token-v1",
            "pattern": csv_replay.NUMBER.pattern,
            "case_insensitive": True,
            "float32_max_finite": repr(csv_replay.FLOAT32_MAX),
            "finite_check": "binary64 conversion; abs(value) <= float32_max_finite; underflow allowed",
            "whitespace_semantics": "Python str.strip code points; original decoded cell preserved",
            "strip_code_points": [point for point in range(0x110000) if chr(point).isspace()],
        },
        "limits": {
            "source_bytes": csv_replay.MAX_BYTES, "output_bytes": csv_replay.MAX_BYTES,
            "flow_records": csv_replay.MAX_ROWS,
            "prediction_windows": csv_replay.MAX_ROWS // csv_replay.WINDOW,
            "window": csv_replay.WINDOW, "inference_batch_windows": csv_replay.BATCH_SIZE,
            "retained_field_characters": csv.field_size_limit(),
            "field_limit_semantics": "current process csv.field_size_limit; decoded Unicode characters",
        },
        "serialization": "UTF-8 without BOM; QUOTE_ALL; LF records; decoded retained cells unchanged",
        "source_encoding": "strict UTF-8; optional single initial BOM",
        "header_matching": "canonical unique names in any order; exact ordered standard_80/extended_84",
    }
    return {**record, "descriptor_sha256": canonical_sha256(record)}
