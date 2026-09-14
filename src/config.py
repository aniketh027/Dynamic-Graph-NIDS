"""Project configuration loading and validation.

The configuration is a YAML file (``configs/default.yaml`` by default) mapped
onto typed dataclasses, with optional environment-variable overrides.

Deliberately dataset-agnostic: this module knows about *paths*, *debug limits*,
the *temporal window*, *split ratios* and the *seed* only. No CIC-IDS2018
column names or schema details appear here; they are added after the real
dataset has been inspected.

Usage
-----
    from src.config import Config

    cfg = Config.load()                       # configs/default.yaml
    cfg = Config.load("configs/custom.yaml")  # explicit file
    cfg.ensure_directories()                  # create output dirs
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Repository root = parent of the ``src`` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"

#: Ratios are compared against 1.0 with this absolute tolerance.
RATIO_TOLERANCE: float = 1e-6

#: Environment variables that override the YAML values (all optional).
ENV_PREFIX = "NIDS_"


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or invalid."""


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class ProjectConfig:
    name: str = "dynamic-graph-nids"
    seed: int = 42


@dataclass
class PathsConfig:
    """All paths are resolved to absolute paths against :data:`PROJECT_ROOT`."""

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    results_dir: Path = PROJECT_ROOT / "results"
    plots_dir: Path = PROJECT_ROOT / "plots"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    logs_dir: Path = PROJECT_ROOT / "logs"
    dataset_file: Optional[str] = None

    @property
    def output_dirs(self) -> List[Path]:
        """Directories that the project is allowed to create and write to."""
        return [
            self.processed_data_dir,
            self.results_dir,
            self.plots_dir,
            self.checkpoints_dir,
            self.logs_dir,
        ]

    @property
    def required_dirs(self) -> List[Path]:
        """Directories that must exist for the project to be usable."""
        return [self.raw_data_dir, *self.output_dirs]


@dataclass
class DebugConfig:
    enabled: bool = False
    row_limit: int = 100_000

    @property
    def row_limit_or_none(self) -> Optional[int]:
        """Row cap to apply to readers: the limit in debug mode, else ``None``."""
        return self.row_limit if self.enabled else None


@dataclass
class TemporalConfig:
    #: Temporal window T.
    window_size: int = 5


@dataclass
class SplitConfig:
    train: float = 0.70
    validation: float = 0.10
    test: float = 0.20

    @property
    def total(self) -> float:
        return self.train + self.validation + self.test

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.train, self.validation, self.test)


@dataclass
class InspectionConfig:
    chunk_size: int = 500_000
    file_patterns: List[str] = field(default_factory=lambda: ["*.csv", "*.csv.gz"])
    timestamp_sample_size: int = 200
    max_label_cardinality: int = 50


#: Accepted values for :attr:`LoaderConfig.infinity_policy`.
INFINITY_POLICIES: Tuple[str, ...] = ("column_finite_extremum", "constant")


@dataclass
class LoaderConfig:
    """Cleaning policy for the raw CIC-IDS2018 CSV loader.

    Only the day-CSV loader reads this section. Column names live in
    :mod:`src.preprocessing.schema`, not here.
    """

    #: Day file used for loader development and tests (inside ``raw_data_dir``).
    primary_csv: str = "csv/03-01-2018.csv"
    #: Value substituted for NaN in the model feature columns.
    nan_replacement: float = 0.0
    #: How +/-inf is replaced. ``column_finite_extremum`` substitutes the largest
    #: (smallest) finite value observed in that column, preserving the ordering
    #: "inf was above everything else"; ``constant`` uses
    #: :attr:`infinity_replacement` for both signs.
    infinity_policy: str = "column_finite_extremum"
    #: Substitute used when ``infinity_policy == "constant"``.
    infinity_replacement: float = 0.0
    #: Measure (never remove) exact / temporal / conflicting duplicate rows.
    compute_duplicate_stats: bool = True
    #: Cross-check the parsed row count against the file's physical line count.
    verify_raw_line_count: bool = True


@dataclass
class PreprocessingConfig:
    """Stage C: cleaning, windowing, and train/val/test split construction.

    The raw CSVs are opened read-only; nothing under ``data/raw/`` is modified.
    Column names live in :mod:`src.preprocessing.schema`, not here.
    """

    #: Rows per chunk when streaming each day CSV for hash/exclusion passes.
    chunk_size: int = 200_000
    #: Minimum T=5 test windows a class must have to appear in per-class
    #: performance claims.  Classes below the threshold are documented in the
    #: split metadata but excluded from primary evaluation; never silently merged.
    rare_class_min_test_windows: int = 30
    #: Output sub-directory under ``processed_data_dir``.
    output_subdir: str = "preprocessed"
    #: Write cleaning statistics (exact/conflict/temporal counts) to JSON.
    save_cleaning_stats: bool = True
    #: Write per-split window counts and class distributions to JSON.
    save_split_metadata: bool = True


@dataclass
class ProfilingConfig:
    """Multi-day corpus profiling (Stage B). Measure-only: nothing is cleaned.

    Column names live in :mod:`src.preprocessing.schema`, not here.
    """

    #: Sub-directory of ``raw_data_dir`` holding the CSV corpus.
    csv_subdir: str = "csv"
    #: Sub-directory of ``raw_data_dir`` holding the Parquet copy (metadata only).
    parquet_subdir: str = "parquet"
    #: Glob patterns used to discover CSV files (searched recursively).
    csv_patterns: List[str] = field(default_factory=lambda: ["*.csv"])
    #: Rows per chunk in the data pass. 200k x 78 float64 is ~125 MB, which keeps
    #: peak memory well under a laptop's budget even for the 3.8 GiB file.
    chunk_size: int = 200_000
    #: Measure the three duplicate categories per file.
    compute_duplicates: bool = True
    #: Also measure them across the whole corpus (holds 3 uint64 hashes per row
    #: in memory: ~24 bytes x 16.2M rows = ~390 MB).
    compute_global_duplicates: bool = True
    #: Compare against the Parquet copy using file metadata only.
    compare_parquet: bool = True
    #: Output sub-directory under ``results_dir``.
    output_subdir: str = "data_profile"
    #: Cap on how many embedded-header positions / examples are stored per file.
    max_examples: int = 64


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    inspection: InspectionConfig = field(default_factory=InspectionConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    #: Path the config was loaded from (``None`` for pure defaults).
    source_path: Optional[Path] = None

    # -- convenience accessors (stable API for later phases) ---------------- #
    @property
    def seed(self) -> int:
        return self.project.seed

    @property
    def dataset_dir(self) -> Path:
        """Configured dataset (input) directory."""
        return self.paths.raw_data_dir

    @property
    def output_dir(self) -> Path:
        """Configured primary output directory."""
        return self.paths.processed_data_dir

    @property
    def window_size(self) -> int:
        """Temporal window T."""
        return self.temporal.window_size

    # -- loading ----------------------------------------------------------- #
    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str] | None = None,
        *,
        apply_env: bool = True,
    ) -> "Config":
        """Load configuration from YAML.

        Parameters
        ----------
        path:
            Config file. Defaults to ``$NIDS_CONFIG`` if set, else
            ``configs/default.yaml``.
        apply_env:
            Apply ``NIDS_*`` environment-variable overrides after parsing.
        """
        if path is None:
            path = os.environ.get(f"{ENV_PREFIX}CONFIG") or DEFAULT_CONFIG_PATH
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = (PROJECT_ROOT / config_path).resolve()

        if not config_path.is_file():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - malformed YAML
            raise ConfigError(f"Could not parse YAML config {config_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

        cfg = cls.from_dict(raw)
        cfg.source_path = config_path
        if apply_env:
            cfg._apply_env_overrides()
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        """Build a config from a plain dict (unknown keys are rejected)."""
        sections = {
            "project": ProjectConfig,
            "paths": PathsConfig,
            "debug": DebugConfig,
            "temporal": TemporalConfig,
            "split": SplitConfig,
            "inspection": InspectionConfig,
            "loader": LoaderConfig,
            "profiling": ProfilingConfig,
            "preprocessing": PreprocessingConfig,
        }
        unknown_sections = set(raw) - set(sections)
        if unknown_sections:
            raise ConfigError(
                f"Unknown config section(s): {sorted(unknown_sections)}. "
                f"Expected any of {sorted(sections)}."
            )

        kwargs: Dict[str, Any] = {}
        for name, section_cls in sections.items():
            values = raw.get(name) or {}
            if not isinstance(values, dict):
                raise ConfigError(f"Config section '{name}' must be a mapping.")
            allowed = {f.name for f in section_cls.__dataclass_fields__.values()}
            unknown_keys = set(values) - allowed
            if unknown_keys:
                raise ConfigError(
                    f"Unknown key(s) in section '{name}': {sorted(unknown_keys)}. "
                    f"Expected any of {sorted(allowed)}."
                )
            kwargs[name] = section_cls(**values)

        cfg = cls(**kwargs)
        cfg._resolve_paths()
        return cfg

    # -- internals --------------------------------------------------------- #
    def _resolve_paths(self) -> None:
        """Make every configured directory absolute, relative to the repo root."""
        for name in (
            "raw_data_dir",
            "processed_data_dir",
            "results_dir",
            "plots_dir",
            "checkpoints_dir",
            "logs_dir",
        ):
            value = Path(getattr(self.paths, name)).expanduser()
            if not value.is_absolute():
                value = PROJECT_ROOT / value
            setattr(self.paths, name, value)

    def _apply_env_overrides(self) -> None:
        """Apply ``NIDS_*`` environment overrides (useful for CI / quick runs)."""
        env = os.environ

        def _get(key: str) -> Optional[str]:
            value = env.get(f"{ENV_PREFIX}{key}")
            return value.strip() if value and value.strip() else None

        for key, attr in (
            ("RAW_DATA_DIR", "raw_data_dir"),
            ("PROCESSED_DATA_DIR", "processed_data_dir"),
            ("RESULTS_DIR", "results_dir"),
            ("PLOTS_DIR", "plots_dir"),
            ("CHECKPOINTS_DIR", "checkpoints_dir"),
            ("LOGS_DIR", "logs_dir"),
        ):
            value = _get(key)
            if value:
                setattr(self.paths, attr, Path(value).expanduser())

        dataset_file = _get("DATASET_FILE")
        if dataset_file:
            self.paths.dataset_file = dataset_file

        debug = _get("DEBUG")
        if debug is not None:
            self.debug.enabled = debug.lower() in {"1", "true", "yes", "on"}

        for key, setter in (
            ("DEBUG_ROW_LIMIT", lambda v: setattr(self.debug, "row_limit", v)),
            ("WINDOW_SIZE", lambda v: setattr(self.temporal, "window_size", v)),
            ("SEED", lambda v: setattr(self.project, "seed", v)),
        ):
            value = _get(key)
            if value is None:
                continue
            try:
                setter(int(value))
            except ValueError as exc:
                raise ConfigError(
                    f"Environment variable {ENV_PREFIX}{key} must be an integer, got {value!r}"
                ) from exc

        self._resolve_paths()

    # -- validation / IO --------------------------------------------------- #
    def validate(self) -> "Config":
        """Validate value ranges and cross-field constraints."""
        if not isinstance(self.project.seed, int) or isinstance(self.project.seed, bool):
            raise ConfigError(f"project.seed must be an int, got {self.project.seed!r}")

        if not isinstance(self.debug.enabled, bool):
            raise ConfigError(f"debug.enabled must be a bool, got {self.debug.enabled!r}")
        if not isinstance(self.debug.row_limit, int) or self.debug.row_limit <= 0:
            raise ConfigError(
                f"debug.row_limit must be a positive int, got {self.debug.row_limit!r}"
            )

        if not isinstance(self.temporal.window_size, int) or self.temporal.window_size < 1:
            raise ConfigError(
                f"temporal.window_size (T) must be an int >= 1, got {self.temporal.window_size!r}"
            )

        for name in ("train", "validation", "test"):
            ratio = getattr(self.split, name)
            if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
                raise ConfigError(f"split.{name} must be a number, got {ratio!r}")
            if not 0.0 < float(ratio) < 1.0:
                raise ConfigError(f"split.{name} must be in (0, 1), got {ratio}")
        if abs(self.split.total - 1.0) > RATIO_TOLERANCE:
            raise ConfigError(
                "split ratios must sum to 1.0, got "
                f"train={self.split.train} + validation={self.split.validation} + "
                f"test={self.split.test} = {self.split.total}"
            )

        if not isinstance(self.inspection.chunk_size, int) or self.inspection.chunk_size <= 0:
            raise ConfigError(
                f"inspection.chunk_size must be a positive int, got {self.inspection.chunk_size!r}"
            )
        if not self.inspection.file_patterns:
            raise ConfigError("inspection.file_patterns must contain at least one pattern.")
        if (
            not isinstance(self.inspection.max_label_cardinality, int)
            or self.inspection.max_label_cardinality < 2
        ):
            raise ConfigError(
                "inspection.max_label_cardinality must be an int >= 2, got "
                f"{self.inspection.max_label_cardinality!r}"
            )
        if (
            not isinstance(self.inspection.timestamp_sample_size, int)
            or self.inspection.timestamp_sample_size < 1
        ):
            raise ConfigError(
                "inspection.timestamp_sample_size must be an int >= 1, got "
                f"{self.inspection.timestamp_sample_size!r}"
            )

        if not self.loader.primary_csv or not isinstance(self.loader.primary_csv, str):
            raise ConfigError(
                f"loader.primary_csv must be a non-empty string, got {self.loader.primary_csv!r}"
            )
        if self.loader.infinity_policy not in INFINITY_POLICIES:
            raise ConfigError(
                f"loader.infinity_policy must be one of {list(INFINITY_POLICIES)}, "
                f"got {self.loader.infinity_policy!r}"
            )
        for name in ("nan_replacement", "infinity_replacement"):
            value = getattr(self.loader, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(f"loader.{name} must be a number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ConfigError(f"loader.{name} must be finite, got {value!r}")
        for name in ("compute_duplicate_stats", "verify_raw_line_count"):
            value = getattr(self.loader, name)
            if not isinstance(value, bool):
                raise ConfigError(f"loader.{name} must be a bool, got {value!r}")

        if not isinstance(self.profiling.chunk_size, int) or self.profiling.chunk_size <= 0:
            raise ConfigError(
                f"profiling.chunk_size must be a positive int, got "
                f"{self.profiling.chunk_size!r}"
            )
        if not self.profiling.csv_patterns:
            raise ConfigError("profiling.csv_patterns must contain at least one pattern.")
        for name in ("csv_subdir", "parquet_subdir", "output_subdir"):
            value = getattr(self.profiling, name)
            if not value or not isinstance(value, str):
                raise ConfigError(
                    f"profiling.{name} must be a non-empty string, got {value!r}"
                )
        for name in ("compute_duplicates", "compute_global_duplicates", "compare_parquet"):
            value = getattr(self.profiling, name)
            if not isinstance(value, bool):
                raise ConfigError(f"profiling.{name} must be a bool, got {value!r}")
        if not isinstance(self.profiling.max_examples, int) or self.profiling.max_examples < 1:
            raise ConfigError(
                f"profiling.max_examples must be an int >= 1, got "
                f"{self.profiling.max_examples!r}"
            )

        if (
            not isinstance(self.preprocessing.chunk_size, int)
            or self.preprocessing.chunk_size <= 0
        ):
            raise ConfigError(
                f"preprocessing.chunk_size must be a positive int, got "
                f"{self.preprocessing.chunk_size!r}"
            )
        if (
            not isinstance(self.preprocessing.rare_class_min_test_windows, int)
            or self.preprocessing.rare_class_min_test_windows < 1
        ):
            raise ConfigError(
                "preprocessing.rare_class_min_test_windows must be an int >= 1, got "
                f"{self.preprocessing.rare_class_min_test_windows!r}"
            )
        if not self.preprocessing.output_subdir or not isinstance(
            self.preprocessing.output_subdir, str
        ):
            raise ConfigError(
                "preprocessing.output_subdir must be a non-empty string, got "
                f"{self.preprocessing.output_subdir!r}"
            )
        for name in ("save_cleaning_stats", "save_split_metadata"):
            value = getattr(self.preprocessing, name)
            if not isinstance(value, bool):
                raise ConfigError(f"preprocessing.{name} must be a bool, got {value!r}")

        return self

    def ensure_directories(self, include_raw: bool = True) -> List[Path]:
        """Create the project's data/output directories if they do not exist.

        Returns the list of directories that are guaranteed to exist afterwards.
        The raw data directory is created empty; nothing is ever written there.
        """
        targets = self.paths.required_dirs if include_raw else self.paths.output_dirs
        for directory in targets:
            directory.mkdir(parents=True, exist_ok=True)
        return list(targets)

    def to_dict(self) -> Dict[str, Any]:
        """JSON/YAML-serialisable view of the configuration."""
        data = asdict(self)
        data.pop("source_path", None)
        for key, value in data["paths"].items():
            if isinstance(value, Path):
                data["paths"][key] = str(value)
        return data

    def summary(self) -> str:
        """Multi-line, human-readable summary (used in logs and reports)."""
        lines = [
            f"config file          : {self.source_path or '<defaults>'}",
            f"project              : {self.project.name}",
            f"seed                 : {self.project.seed}",
            f"dataset dir (input)  : {self.paths.raw_data_dir}",
            f"output dir (processed): {self.paths.processed_data_dir}",
            f"results / plots      : {self.paths.results_dir} | {self.paths.plots_dir}",
            f"checkpoints / logs   : {self.paths.checkpoints_dir} | {self.paths.logs_dir}",
            f"dataset file filter  : {self.paths.dataset_file or '<all matching files>'}",
            f"debug mode           : {self.debug.enabled} (row limit {self.debug.row_limit:,})",
            f"temporal window T    : {self.temporal.window_size}",
            "split ratios         : "
            f"train={self.split.train:.2f} val={self.split.validation:.2f} "
            f"test={self.split.test:.2f}",
            f"loader primary csv   : {self.loader.primary_csv}",
            "loader cleaning      : "
            f"nan->{self.loader.nan_replacement} inf->{self.loader.infinity_policy}",
            "profiling            : "
            f"{self.paths.raw_data_dir.name}/{self.profiling.csv_subdir} "
            f"chunk={self.profiling.chunk_size:,} "
            f"duplicates={self.profiling.compute_duplicates}"
            f"/global={self.profiling.compute_global_duplicates}",
        ]
        return "\n".join(lines)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Functional alias for :meth:`Config.load`."""
    return Config.load(path)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(Config.load().summary())
