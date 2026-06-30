from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute errors between multimodal XXZ estimates saved by eval_new.py "
            "and exact values stored in the same .npz file."
        )
    )
    parser.add_argument("--file", type=str, required=True, help="XXZ eval_new.py output .npz file.")
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Output JSON path. Default: <input_basename>_errors.json next to the .npz.",
    )
    parser.add_argument(
        "--relative_eps",
        type=float,
        default=1e-12,
        help="Denominator floor for relative errors.",
    )
    parser.add_argument(
        "--no_per_delta",
        action="store_true",
        help="Only save aggregate metrics, not per-Delta arrays.",
    )
    return parser.parse_args()


def unwrap_npz_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == () and array.dtype == object:
        item = array.item()
        if item is None:
            return None
        return item
    if array.dtype == object:
        try:
            if array.size == 1 and array.item() is None:
                return None
        except ValueError:
            pass
    return array


def get_any(data: np.lib.npyio.NpzFile, keys: list[str]) -> Any:
    for key in keys:
        if key in data.files:
            value = unwrap_npz_value(data[key])
            if value is not None:
                return value
    return None


def as_float_1d(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == () and array.dtype == object and array.item() is None:
        return None
    try:
        array = array.astype(float)
    except (TypeError, ValueError):
        return None
    array = np.squeeze(array)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[:, 0]
        else:
            array = array.reshape(array.shape[0], -1)[0]
    return array.reshape(-1)


def scalar_or_none(value: Any) -> Any:
    value = unwrap_npz_value(value)
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        if isinstance(item, np.generic):
            return item.item()
        return item
    if array.size == 1:
        item = array.reshape(-1)[0]
        if isinstance(item, np.generic):
            return item.item()
        return item
    return array.tolist()


def finite_list(array: np.ndarray) -> list[float | None]:
    values: list[float | None] = []
    for value in np.asarray(array).reshape(-1):
        value = float(value)
        values.append(value if np.isfinite(value) else None)
    return values


def align_exact_to_deltas(
    eval_deltas: np.ndarray,
    exact_x: np.ndarray | None,
    exact_values: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    exact_values = as_float_1d(exact_values)
    if exact_values is None:
        return None, "missing_exact_values"
    eval_deltas = as_float_1d(eval_deltas)
    if eval_deltas is None:
        return None, "missing_eval_deltas"

    if exact_values.size == eval_deltas.size:
        return exact_values.astype(float), "same_grid"

    exact_x = as_float_1d(exact_x)
    if exact_x is None or exact_x.size != exact_values.size:
        return None, "cannot_align_exact_grid"

    mask = np.isfinite(exact_x) & np.isfinite(exact_values)
    if np.count_nonzero(mask) < 2:
        return None, "not_enough_finite_exact_points"

    order = np.argsort(exact_x[mask])
    x_sorted = exact_x[mask][order]
    y_sorted = exact_values[mask][order]
    return np.interp(eval_deltas, x_sorted, y_sorted).astype(float), "interpolated"


def compute_metric_errors(
    *,
    deltas: np.ndarray,
    exact_x: np.ndarray | None,
    estimate: Any,
    estimate_std: Any,
    exact: Any,
    relative_eps: float,
    include_per_delta: bool,
) -> dict[str, Any] | None:
    estimate_values = as_float_1d(estimate)
    if estimate_values is None:
        return None

    deltas = as_float_1d(deltas)
    if deltas is None:
        return None
    if estimate_values.size != deltas.size:
        return {
            "status": "skipped",
            "reason": "estimate_length_does_not_match_delta_length",
            "estimate_length": int(estimate_values.size),
            "delta_length": int(deltas.size),
        }

    exact_values, alignment = align_exact_to_deltas(deltas, exact_x, exact)
    if exact_values is None:
        return {
            "status": "skipped",
            "reason": alignment,
        }

    valid = np.isfinite(deltas) & np.isfinite(estimate_values) & np.isfinite(exact_values)
    count = int(np.count_nonzero(valid))
    if count == 0:
        return {
            "status": "skipped",
            "reason": "no_finite_overlap",
            "alignment": alignment,
        }

    signed = estimate_values[valid] - exact_values[valid]
    abs_error = np.abs(signed)
    squared = signed**2
    denom = np.maximum(np.abs(exact_values[valid]), float(relative_eps))
    relative = abs_error / denom

    result: dict[str, Any] = {
        "status": "ok",
        "alignment": alignment,
        "num_points": count,
        "mae": float(np.mean(abs_error)),
        "mse": float(np.mean(squared)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "max_abs_error": float(np.max(abs_error)),
        "median_abs_error": float(np.median(abs_error)),
        "mean_signed_error": float(np.mean(signed)),
        "mean_relative_error": float(np.mean(relative)),
        "max_relative_error": float(np.max(relative)),
    }

    std_values = as_float_1d(estimate_std)
    if std_values is not None and std_values.size == deltas.size:
        std_valid = std_values[valid]
        if np.isfinite(std_valid).any():
            result["mean_estimate_std"] = float(np.nanmean(std_valid))
            result["max_estimate_std"] = float(np.nanmax(std_valid))

    if include_per_delta:
        full_signed = np.full_like(estimate_values, np.nan, dtype=float)
        full_abs = np.full_like(estimate_values, np.nan, dtype=float)
        full_squared = np.full_like(estimate_values, np.nan, dtype=float)
        full_relative = np.full_like(estimate_values, np.nan, dtype=float)
        full_signed[valid] = estimate_values[valid] - exact_values[valid]
        full_abs[valid] = np.abs(full_signed[valid])
        full_squared[valid] = full_signed[valid] ** 2
        full_relative[valid] = full_abs[valid] / np.maximum(np.abs(exact_values[valid]), float(relative_eps))

        result["per_delta"] = {
            "Delta": finite_list(deltas),
            "estimate": finite_list(estimate_values),
            "exact": finite_list(exact_values),
            "signed_error": finite_list(full_signed),
            "absolute_error": finite_list(full_abs),
            "squared_error": finite_list(full_squared),
            "mse": finite_list(full_squared),
            "relative_error": finite_list(full_relative),
        }
        if std_values is not None and std_values.size == deltas.size:
            result["per_delta"]["estimate_std"] = finite_list(std_values)

    return result


def metric_specs() -> dict[str, list[dict[str, Any]]]:
    return {
        "single": [
            {
                "name": "energy",
                "estimate": ["energy_mean"],
                "std": ["energy_std"],
                "exact": ["exact_single_energy_per_bond", "exact_single_energy", "exact_energy_per_bond"],
            },
            {
                "name": "energy_derivative",
                "estimate": ["energy_derivative_mean"],
                "std": ["energy_derivative_std"],
                "exact": [
                    "exact_single_energy_derivative_per_bond",
                    "exact_single_energy_derivative",
                    "exact_energy_derivative_per_bond",
                    "exact_energy_derivative_analytic",
                ],
            },
            {
                "name": "magnetization_z",
                "estimate": ["magnetization_z_mean"],
                "std": ["magnetization_z_std"],
                "exact": ["exact_single_magnetization_z", "exact_magnetization_z"],
            },
            {
                "name": "magnetization_s",
                "estimate": ["magnetization_s_mean"],
                "std": ["magnetization_s_std"],
                "exact": ["exact_single_magnetization_s", "exact_magnetization_s"],
            },
            {
                "name": "corr_XX",
                "estimate": ["corr_XX_mean_single"],
                "std": ["corr_XX_std_single"],
                "exact": ["exact_single_corr_XX"],
            },
            {
                "name": "corr_YY",
                "estimate": ["corr_YY_mean_single"],
                "std": ["corr_YY_std_single"],
                "exact": ["exact_single_corr_YY"],
            },
            {
                "name": "corr_ZZ",
                "estimate": ["corr_ZZ_mean_single"],
                "std": ["corr_ZZ_std_single"],
                "exact": ["exact_single_corr_ZZ", "exact_zz_correlation"],
            },
        ],
        "c": [
            {
                "name": "energy",
                "estimate": ["energy_mean_from_c"],
                "std": ["energy_std_from_c"],
                "exact": ["exact_c_energy_per_bond", "exact_c_energy"],
            },
            {
                "name": "energy_derivative",
                "estimate": ["energy_derivative_mean_from_c"],
                "std": ["energy_derivative_std_from_c"],
                "exact": ["exact_c_energy_derivative_per_bond", "exact_c_energy_derivative"],
            },
            {
                "name": "corr_XX",
                "estimate": ["corr_XX_mean_from_c"],
                "std": ["corr_XX_std_from_c"],
                "exact": ["exact_c_corr_XX", "exact_single_corr_XX"],
            },
            {
                "name": "corr_YY",
                "estimate": ["corr_YY_mean_from_c"],
                "std": ["corr_YY_std_from_c"],
                "exact": ["exact_c_corr_YY", "exact_single_corr_YY"],
            },
            {
                "name": "corr_ZZ",
                "estimate": ["corr_ZZ_mean_from_c"],
                "std": ["corr_ZZ_std_from_c"],
                "exact": ["exact_c_corr_ZZ", "exact_single_corr_ZZ", "exact_zz_correlation"],
            },
        ],
    }


def compute_errors(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.file, allow_pickle=True) as data:
        deltas = get_any(data, ["Deltas", "hs", "hs_gpt"])
        exact_x = get_any(data, ["Delta_values_dense", "Delta_values", "delta_values", "Deltas", "hs"])
        if deltas is None:
            raise ValueError(f"{args.file} does not contain Deltas/hs/hs_gpt.")

        output: dict[str, Any] = {
            "source_file": os.path.abspath(args.file),
            "relative_eps": float(args.relative_eps),
            "metadata": {
                "N": scalar_or_none(data["N"]) if "N" in data.files else None,
                "J_xy": scalar_or_none(data["J_xy"]) if "J_xy" in data.files else None,
                "pair_basis_mode": scalar_or_none(data["pair_basis_mode"]) if "pair_basis_mode" in data.files else None,
                "delta_count": int(as_float_1d(deltas).size) if as_float_1d(deltas) is not None else None,
            },
            "modalities": {},
            "skipped": {},
        }

        for modality, specs in metric_specs().items():
            modality_errors: dict[str, Any] = {}
            modality_skipped: dict[str, Any] = {}
            for spec in specs:
                estimate = get_any(data, spec["estimate"])
                exact = get_any(data, spec["exact"])
                estimate_std = get_any(data, spec["std"])
                metric_error = compute_metric_errors(
                    deltas=deltas,
                    exact_x=exact_x,
                    estimate=estimate,
                    estimate_std=estimate_std,
                    exact=exact,
                    relative_eps=args.relative_eps,
                    include_per_delta=not args.no_per_delta,
                )

                if metric_error is None:
                    modality_skipped[spec["name"]] = {
                        "reason": "missing_estimate_or_exact",
                        "estimate_keys": spec["estimate"],
                        "exact_keys": spec["exact"],
                    }
                elif metric_error.get("status") == "skipped":
                    modality_skipped[spec["name"]] = metric_error
                else:
                    modality_errors[spec["name"]] = metric_error

            if modality_errors:
                output["modalities"][modality] = modality_errors
            if modality_skipped:
                output["skipped"][modality] = modality_skipped

    return output


def default_output_path(input_path: str) -> str:
    root, _ext = os.path.splitext(input_path)
    return f"{root}_errors.json"


def main() -> int:
    args = parse_args()
    errors = compute_errors(args)
    output_path = args.output_json or default_output_path(args.file)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    print(f"Saved XXZ multimodal error JSON: {output_path}")
    for modality, metrics in errors["modalities"].items():
        print(f"  {modality}: {', '.join(metrics.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
