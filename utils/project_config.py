from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"

ENV_OVERRIDES = {
    "CITRANSFORMER_DATA_DIR": ("paths", "data_dir"),
    "CITRANSFORMER_RESULTS_ROOT": ("paths", "results_root"),
    "CITRANSFORMER_CHECKPOINTS_ROOT": ("paths", "checkpoints_root"),
    "CITRANSFORMER_MPLCONFIGDIR": ("paths", "matplotlib_cache"),
    "CITRANSFORMER_DEVICE": ("runtime", "device"),
    "CITRANSFORMER_NUM_WORKERS": ("runtime", "num_workers"),
    "PYTHON_BIN": ("runtime", "python_bin"),
}


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    values: dict[str, Any]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.values
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def get_path(self, dotted_key: str, default: str | Path | None = None) -> Path:
        value = self.get(dotted_key, default)
        if value is None:
            raise KeyError(f"Missing path config value: {dotted_key}")
        return resolve_project_path(value, project_root=self.project_root)


def resolve_project_path(path_value: str | Path, project_root: str | Path = PROJECT_ROOT) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return Path(project_root) / path


def load_project_config(config_path: str | Path | None = None) -> ProjectConfig:
    values: dict[str, Any] = {}
    values = _deep_merge(values, _load_yaml_file(DEFAULT_CONFIG_PATH, required=False))

    local_path = _resolve_optional_config_path(config_path)
    if local_path is not None:
        values = _deep_merge(values, _load_yaml_file(local_path, required=True))
    elif LOCAL_CONFIG_PATH.exists():
        values = _deep_merge(values, _load_yaml_file(LOCAL_CONFIG_PATH, required=True))

    values = _apply_env_overrides(values)
    values = _derive_root_based_paths(values)
    return ProjectConfig(project_root=PROJECT_ROOT, values=values)


def _resolve_optional_config_path(config_path: str | Path | None) -> Path | None:
    path_value = config_path or os.environ.get("CITRANSFORMER_CONFIG")
    if not path_value:
        return None
    return resolve_project_path(path_value)


def _load_yaml_file(path: Path, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Project config file does not exist: {path}")
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for project configuration. Install requirements.txt.") from exc

    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Project config must be a YAML mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_env_overrides(values: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(values)
    for env_name, key_path in ENV_OVERRIDES.items():
        if env_name not in os.environ or os.environ[env_name] == "":
            continue
        value: Any = os.environ[env_name]
        if env_name == "CITRANSFORMER_NUM_WORKERS":
            value = int(value)
        _set_nested(merged, key_path, value)
    return merged


def _derive_root_based_paths(values: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(values)
    paths = merged.setdefault("paths", {})
    results_root = paths.get("results_root")
    checkpoints_root = paths.get("checkpoints_root")

    if results_root and str(results_root) != "results/d1_long_no_wind_2015_2022":
        root = str(results_root).rstrip("/")
        paths["results"] = {
            "persistence": f"{root}/persistence",
            "lstm": f"{root}/lstm",
            "itransformer": f"{root}/itransformer",
            "itransformer_tuned": f"{root}/itransformer_tuned",
            "itransformer_global_pcmci_11vars": f"{root}/itransformer_global_pcmci_11vars",
            "tuning_itransformer": f"{root}/tuning/itransformer",
            "causal_graphs_global_pcmci_11vars_train": f"{root}/causal_graphs/global_pcmci_11vars_train",
            "data_audit_raw_pv": f"{root}/data_audit/raw_pv",
        }

    if checkpoints_root and str(checkpoints_root) != "checkpoints/d1_long_no_wind_2015_2022":
        root = str(checkpoints_root).rstrip("/")
        paths["checkpoints"] = {
            "lstm": f"{root}/lstm",
            "itransformer": f"{root}/itransformer",
            "itransformer_global_pcmci_11vars": f"{root}/itransformer_global_pcmci_11vars",
            "tuning_itransformer": f"{root}/tuning/itransformer",
        }

    return merged


def _set_nested(values: dict[str, Any], key_path: tuple[str, ...], value: Any) -> None:
    current = values
    for key in key_path[:-1]:
        nested = current.setdefault(key, {})
        if not isinstance(nested, dict):
            raise ValueError(f"Cannot override non-mapping config key: {'.'.join(key_path)}")
        current = nested
    current[key_path[-1]] = value


def _format_value(config: ProjectConfig, key: str, raw: bool) -> str:
    value = config.get(key)
    if value is None:
        raise KeyError(f"Unknown config key: {key}")
    if not raw and key.startswith("paths."):
        return str(resolve_project_path(str(value), config.project_root))
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read CiTransformer project configuration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_parser = subparsers.add_parser("get", help="Print one dotted config value.")
    get_parser.add_argument("key", help="Dotted config key, for example paths.data_dir.")
    get_parser.add_argument("--raw", action="store_true", help="Print the raw value without path resolution.")

    subparsers.add_parser("dump", help="Print the merged config as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config()
    if args.command == "get":
        print(_format_value(config, args.key, raw=args.raw))
        return
    if args.command == "dump":
        print(json.dumps(config.values, ensure_ascii=False, indent=2))
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
