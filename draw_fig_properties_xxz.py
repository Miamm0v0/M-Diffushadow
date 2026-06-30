import argparse
import os

import numpy as np

plt = None


def ensure_matplotlib():
    global plt
    if plt is not None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it with: pip install matplotlib") from exc
    plt = pyplot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot single/B and pair/C modality results from eval_new.py for the XXZ model."
    )
    parser.add_argument("--file", type=str, required=True, help="eval_new.py output .npz file.")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Directory for figures.")
    parser.add_argument(
        "--modalities",
        type=str,
        default="auto",
        choices=["auto", "single", "c", "both"],
        help="Which modality figures to draw. auto draws every modality found in the file.",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Name used in output file names.")
    parser.add_argument("--fig_width", type=float, default=8.0)
    parser.add_argument("--fig_height", type=float, default=6.0)
    parser.add_argument("--save_format", type=str, default="png", choices=["png", "pdf", "svg", "eps"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--with_error_bars", action="store_true", help="Show std error bars.")
    parser.add_argument("--show_phase_boundaries", action="store_true", help="Show Delta=-1 and Delta=1.")
    parser.add_argument("--marker_size", type=int, default=45)
    parser.add_argument("--line_width", type=float, default=2.3)
    parser.add_argument("--exact_color", type=str, default="black")
    parser.add_argument("--single_color", type=str, default="tab:blue")
    parser.add_argument("--c_color", type=str, default="tab:orange")
    return parser.parse_args()


def _unwrap_npz_value(value):
    array = np.asarray(value)
    if array.shape == () and array.dtype == object:
        item = array.item()
        if item is None:
            return None
        return np.asarray(item)
    if array.dtype == object:
        try:
            if array.size == 1 and array.item() is None:
                return None
        except ValueError:
            pass
        try:
            array = array.astype(float)
        except (TypeError, ValueError):
            pass
    return array


def _get(npz, *keys, default=None):
    for key in keys:
        if key in npz.files:
            value = _unwrap_npz_value(npz[key])
            if value is not None:
                return value
    return default


def _get_scalar(npz, key, default):
    value = _get(npz, key, default=None)
    if value is None:
        return default
    array = np.asarray(value)
    if array.size == 0:
        return default
    return array.reshape(-1)[0].item()


def _as_float_1d(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == () and array.dtype == object and array.item() is None:
        return None
    try:
        array = array.astype(float)
    except (TypeError, ValueError):
        return None
    return array.reshape(-1)


def _has_finite(value):
    array = _as_float_1d(value)
    return array is not None and array.size > 0 and np.isfinite(array).any()


def _finite_xy(x, y, yerr=None):
    y = _as_float_1d(y)
    if y is None:
        return None, None, None
    x = _as_float_1d(x)
    if x is None or x.size != y.size:
        x = np.arange(y.size, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    err = _as_float_1d(yerr)
    if err is not None and err.size == y.size:
        mask = mask & np.isfinite(err)
        err = err[mask]
    else:
        err = None
    return x[mask], y[mask], err


def _line_x_y(primary_x, fallback_x, y):
    y = _as_float_1d(y)
    if y is None:
        return None, None
    x = _as_float_1d(primary_x)
    fallback = _as_float_1d(fallback_x)
    if x is None or x.size != y.size:
        x = fallback if fallback is not None and fallback.size == y.size else np.arange(y.size, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _format_pair_basis(value):
    if value is None:
        return "unknown"
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array.reshape(-1)[0])


def load_data(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        deltas = _get(data, "Deltas", "hs")
        dense = _get(data, "Delta_values_dense", "Delta_values", default=deltas)
        result = {
            "source_file": file_path,
            "Deltas": _as_float_1d(deltas),
            "Delta_values_dense": _as_float_1d(dense),
            "N": int(_get_scalar(data, "N", 10)),
            "J_xy": float(_get_scalar(data, "J_xy", 1.0)),
            "pair_basis_mode": _format_pair_basis(_get(data, "pair_basis_mode", default=None)),
            "single": {
                "energy": _get(data, "energy_mean"),
                "energy_std": _get(data, "energy_std"),
                "derivative": _get(data, "energy_derivative_mean"),
                "derivative_std": _get(data, "energy_derivative_std"),
                "mz": _get(data, "magnetization_z_mean"),
                "mz_std": _get(data, "magnetization_z_std"),
                "ms": _get(data, "magnetization_s_mean"),
                "ms_std": _get(data, "magnetization_s_std"),
                "exact_energy": _get(
                    data,
                    "exact_single_energy_per_bond",
                    "exact_single_energy",
                    "exact_energy_per_bond",
                ),
                "exact_derivative": _get(
                    data,
                    "exact_single_energy_derivative_per_bond",
                    "exact_single_energy_derivative",
                    "exact_energy_derivative_per_bond",
                    "exact_energy_derivative_analytic",
                ),
                "exact_mz": _get(data, "exact_single_magnetization_z", "exact_magnetization_z"),
                "exact_ms": _get(data, "exact_single_magnetization_s", "exact_magnetization_s"),
                "exact_corr_xx": _get(data, "exact_single_corr_XX"),
                "exact_corr_yy": _get(data, "exact_single_corr_YY"),
                "exact_corr_zz": _get(data, "exact_single_corr_ZZ", "exact_zz_correlation"),
            },
            "c": {
                "energy": _get(data, "energy_mean_from_c"),
                "energy_std": _get(data, "energy_std_from_c"),
                "derivative": _get(data, "energy_derivative_mean_from_c"),
                "derivative_std": _get(data, "energy_derivative_std_from_c"),
                "corr_xx": _get(data, "corr_XX_mean_from_c"),
                "corr_xx_std": _get(data, "corr_XX_std_from_c"),
                "corr_yy": _get(data, "corr_YY_mean_from_c"),
                "corr_yy_std": _get(data, "corr_YY_std_from_c"),
                "corr_zz": _get(data, "corr_ZZ_mean_from_c"),
                "corr_zz_std": _get(data, "corr_ZZ_std_from_c"),
                "exact_energy": _get(data, "exact_c_energy_per_bond", "exact_c_energy"),
                "exact_derivative": _get(data, "exact_c_energy_derivative_per_bond", "exact_c_energy_derivative"),
                "exact_corr_xx": _get(data, "exact_c_corr_XX", "exact_single_corr_XX"),
                "exact_corr_yy": _get(data, "exact_c_corr_YY", "exact_single_corr_YY"),
                "exact_corr_zz": _get(data, "exact_c_corr_ZZ", "exact_single_corr_ZZ", "exact_zz_correlation"),
            },
        }

    print(f"Loaded: {os.path.basename(file_path)}")
    print(f"  N={result['N']}, J_xy={result['J_xy']}, pair_basis_mode={result['pair_basis_mode']}")
    if result["Deltas"] is not None and result["Deltas"].size:
        print(f"  Delta points: {result['Deltas'].size}, range=[{result['Deltas'].min():.3g}, {result['Deltas'].max():.3g}]")
    return result


def has_single_data(data):
    single = data["single"]
    return any(_has_finite(single[key]) for key in ("energy", "derivative", "mz", "ms"))


def has_c_data(data):
    c_data = data["c"]
    return any(_has_finite(c_data[key]) for key in ("energy", "derivative", "corr_xx", "corr_yy", "corr_zz"))


def add_phase_boundaries(ax, args):
    if not args.show_phase_boundaries:
        return
    ax.axvline(-1.0, color="crimson", linestyle="--", linewidth=1.1, alpha=0.65)
    ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.1, alpha=0.65)


def plot_quantity(ax, data, pred, pred_std, exact, title, ylabel, pred_label, color, marker, args):
    x_exact, y_exact = _line_x_y(data["Delta_values_dense"], data["Deltas"], exact)
    if x_exact is not None and x_exact.size:
        ax.plot(
            x_exact,
            y_exact,
            color=args.exact_color,
            linewidth=args.line_width,
            label="Exact",
        )

    x_pred, y_pred, yerr = _finite_xy(data["Deltas"], pred, pred_std)
    if x_pred is not None and x_pred.size:
        if args.with_error_bars and yerr is not None:
            ax.errorbar(
                x_pred,
                y_pred,
                yerr=yerr,
                fmt=marker,
                color=color,
                capsize=4,
                markersize=max(4, args.marker_size // 9),
                label=pred_label,
            )
        else:
            ax.scatter(
                x_pred,
                y_pred,
                color=color,
                s=args.marker_size,
                marker=marker,
                alpha=0.85,
                label=pred_label,
            )

    add_phase_boundaries(ax, args)
    ax.set_title(title)
    ax.set_xlabel(r"Anisotropy $\Delta$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()


def save_figure(fig, path, args):
    fig.tight_layout()
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_single_figures(data, args, base):
    single = data["single"]
    paths = []

    plots = [
        ("single_energy", single["energy"], single["energy_std"], single["exact_energy"], "Single/B: Energy per bond", r"$E/N_{\mathrm{bond}}$", "Single/B", "o"),
        ("single_energy_derivative", single["derivative"], single["derivative_std"], single["exact_derivative"], r"Single/B: $dE/d\Delta$ per bond", r"$dE/d\Delta$", "Single/B", "s"),
        ("single_magnetization_z", single["mz"] * 3, single["mz_std"], single["exact_mz"], r"Single/B: Magnetization $m_z$", r"$m_z$", "Single/B", "o"),
        ("single_staggered_magnetization", single["ms"], single["ms_std"], single["exact_ms"], r"Single/B: Staggered magnetization $m_s$", r"$m_s$", "Single/B", "s"),
    ]

    for suffix, pred, std, exact, title, ylabel, label, marker in plots:
        if not (_has_finite(pred) or _has_finite(exact)):
            continue
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        plot_quantity(ax, data, pred, std, exact, title, ylabel, label, args.single_color, marker, args)
        path = f"{base}_{suffix}.{args.save_format}"
        save_figure(fig, path, args)
        paths.append(path)

    if any(_has_finite(single[key]) for key in ("energy", "derivative", "mz", "ms")):
        fig, axes = plt.subplots(2, 2, figsize=(args.fig_width * 1.5, args.fig_height * 1.2))
        for ax, (_suffix, pred, std, exact, title, ylabel, label, marker) in zip(axes.reshape(-1), plots):
            plot_quantity(ax, data, pred, std, exact, title, ylabel, label, args.single_color, marker, args)
        path = f"{base}_single_summary.{args.save_format}"
        save_figure(fig, path, args)
        paths.append(path)

    return paths


def plot_c_correlations(data, args, path):
    c_data = data["c"]
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    configs = [
        ("XX", c_data["corr_xx"], c_data["corr_xx_std"], c_data["exact_corr_xx"], "tab:blue", "o"),
        ("YY", c_data["corr_yy"], c_data["corr_yy_std"], c_data["exact_corr_yy"], "tab:green", "s"),
        ("ZZ", c_data["corr_zz"], c_data["corr_zz_std"], c_data["exact_corr_zz"], "tab:red", "^"),
    ]

    for name, pred, std, exact, color, marker in configs:
        x_exact, y_exact = _line_x_y(data["Delta_values_dense"], data["Deltas"], exact)
        if x_exact is not None and x_exact.size:
            ax.plot(
                x_exact,
                y_exact,
                color=color,
                linewidth=args.line_width,
                linestyle="-",
                alpha=0.65,
                label=f"Exact {name}",
            )
        x_pred, y_pred, yerr = _finite_xy(data["Deltas"], pred, std)
        if x_pred is not None and x_pred.size:
            if args.with_error_bars and yerr is not None:
                ax.errorbar(
                    x_pred,
                    y_pred,
                    yerr=yerr,
                    fmt=marker,
                    color=color,
                    capsize=4,
                    markersize=max(4, args.marker_size // 9),
                    label=f"C estimate {name}",
                )
            else:
                ax.scatter(
                    x_pred,
                    y_pred,
                    color=color,
                    marker=marker,
                    s=args.marker_size,
                    alpha=0.85,
                    label=f"C estimate {name}",
                )

    add_phase_boundaries(ax, args)
    ax.set_title("Pair/C: nearest-neighbor Pauli products")
    ax.set_xlabel(r"Anisotropy $\Delta$")
    ax.set_ylabel(r"$\langle \sigma_i^\alpha \sigma_{i+1}^\alpha \rangle$")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(ncol=2, fontsize=10)
    save_figure(fig, path, args)


def plot_c_figures(data, args, base):
    c_data = data["c"]
    paths = []
    plots = [
        ("c_energy", c_data["energy"], c_data["energy_std"], c_data["exact_energy"], "Pair/C: Energy per bond", r"$E/N_{\mathrm{bond}}$", "Pair/C", "o"),
        ("c_energy_derivative", c_data["derivative"], c_data["derivative_std"], c_data["exact_derivative"], r"Pair/C: $dE/d\Delta$ per bond", r"$dE/d\Delta$", "Pair/C", "s"),
    ]

    for suffix, pred, std, exact, title, ylabel, label, marker in plots:
        if not (_has_finite(pred) or _has_finite(exact)):
            continue
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        plot_quantity(ax, data, pred, std, exact, title, ylabel, label, args.c_color, marker, args)
        path = f"{base}_{suffix}.{args.save_format}"
        save_figure(fig, path, args)
        paths.append(path)

    if any(_has_finite(c_data[key]) or _has_finite(c_data[f"exact_{key}"]) for key in ("corr_xx", "corr_yy", "corr_zz")):
        path = f"{base}_c_correlations.{args.save_format}"
        plot_c_correlations(data, args, path)
        paths.append(path)

    if any(_has_finite(c_data[key]) for key in ("energy", "derivative", "corr_xx", "corr_yy", "corr_zz")):
        fig, axes = plt.subplots(2, 2, figsize=(args.fig_width * 1.5, args.fig_height * 1.2))
        plot_quantity(
            axes[0, 0],
            data,
            c_data["energy"],
            c_data["energy_std"],
            c_data["exact_energy"],
            "Pair/C: Energy per bond",
            r"$E/N_{\mathrm{bond}}$",
            "Pair/C",
            args.c_color,
            "o",
            args,
        )
        plot_quantity(
            axes[0, 1],
            data,
            c_data["derivative"],
            c_data["derivative_std"],
            c_data["exact_derivative"],
            r"Pair/C: $dE/d\Delta$ per bond",
            r"$dE/d\Delta$",
            "Pair/C",
            args.c_color,
            "s",
            args,
        )
        plot_quantity(
            axes[1, 0],
            data,
            c_data["corr_xx"],
            c_data["corr_xx_std"],
            c_data["exact_corr_xx"],
            r"Pair/C: $\langle XX\rangle$",
            r"$\langle XX\rangle$",
            "Pair/C",
            "tab:blue",
            "o",
            args,
        )
        plot_quantity(
            axes[1, 1],
            data,
            c_data["corr_zz"],
            c_data["corr_zz_std"],
            c_data["exact_corr_zz"],
            r"Pair/C: $\langle ZZ\rangle$",
            r"$\langle ZZ\rangle$",
            "Pair/C",
            "tab:red",
            "^",
            args,
        )
        path = f"{base}_c_summary.{args.save_format}"
        save_figure(fig, path, args)
        paths.append(path)

    return paths


def plot_compare(data, args, base):
    single = data["single"]
    c_data = data["c"]
    paths = []
    compare_specs = [
        (
            "energy_compare",
            single["energy"],
            single["energy_std"],
            c_data["energy"],
            c_data["energy_std"],
            single["exact_energy"],
            "Single/B vs Pair/C: Energy per bond",
            r"$E/N_{\mathrm{bond}}$",
        ),
        (
            "energy_derivative_compare",
            single["derivative"],
            single["derivative_std"],
            c_data["derivative"],
            c_data["derivative_std"],
            single["exact_derivative"],
            r"Single/B vs Pair/C: $dE/d\Delta$ per bond",
            r"$dE/d\Delta$",
        ),
    ]

    for suffix, y_single, std_single, y_c, std_c, exact, title, ylabel in compare_specs:
        if not (_has_finite(y_single) and _has_finite(y_c)):
            continue
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        x_exact, y_exact = _line_x_y(data["Delta_values_dense"], data["Deltas"], exact)
        if x_exact is not None and x_exact.size:
            ax.plot(x_exact, y_exact, color=args.exact_color, linewidth=args.line_width, label="Exact")

        x_s, y_s, err_s = _finite_xy(data["Deltas"], y_single, std_single)
        x_c, y_c_plot, err_c = _finite_xy(data["Deltas"], y_c, std_c)
        if x_s is not None and x_s.size:
            if args.with_error_bars and err_s is not None:
                ax.errorbar(x_s, y_s, yerr=err_s, fmt="o", color=args.single_color, capsize=4, label="Single/B")
            else:
                ax.scatter(x_s, y_s, color=args.single_color, s=args.marker_size, marker="o", label="Single/B")
        if x_c is not None and x_c.size:
            if args.with_error_bars and err_c is not None:
                ax.errorbar(x_c, y_c_plot, yerr=err_c, fmt="s", color=args.c_color, capsize=4, label="Pair/C")
            else:
                ax.scatter(x_c, y_c_plot, color=args.c_color, s=args.marker_size, marker="s", label="Pair/C")

        add_phase_boundaries(ax, args)
        ax.set_title(title)
        ax.set_xlabel(r"Anisotropy $\Delta$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
        path = f"{base}_{suffix}.{args.save_format}"
        save_figure(fig, path, args)
        paths.append(path)

    return paths


def main():
    args = parse_args()
    ensure_matplotlib()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.file)
    base_name = os.path.basename(args.file).replace(".npz", "")
    model_name = args.model_name or "xxz"
    base = os.path.join(args.output_dir, f"{base_name}_{model_name}")

    single_ok = has_single_data(data)
    c_ok = has_c_data(data)
    print(f"Detected modalities: single/B={single_ok}, pair/C={c_ok}")

    draw_single = args.modalities in ("single", "both") or (args.modalities == "auto" and single_ok)
    draw_c = args.modalities in ("c", "both") or (args.modalities == "auto" and c_ok)
    draw_compare = (args.modalities in ("both", "auto")) and single_ok and c_ok

    paths = []
    if draw_single:
        paths.extend(plot_single_figures(data, args, base))
    if draw_c:
        paths.extend(plot_c_figures(data, args, base))
    if draw_compare:
        paths.extend(plot_compare(data, args, base))

    if not paths:
        print("No plottable XXZ modality data found. Check the .npz keys or use --modalities explicitly.")
        return

    print("\nGenerated figures:")
    for path in paths:
        print(f"  {os.path.basename(path)}")


if __name__ == "__main__":
    main()
