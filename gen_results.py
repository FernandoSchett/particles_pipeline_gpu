# %% [markdown]
# # Weak and Strong Scaling — Setup

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math

BASE_RESULTS_DIR = Path("results")
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CPU_CORES_PER_NODE = 128
GPUS_PER_NODE = 4
BASE_EFF_NPROCS = 2  # baseline n=2 para split/dist/sd/total

# -----------------------------
# Utils
# -----------------------------
def savefig(results_dir, name):
    plt.tight_layout()
    plt.savefig(results_dir / f"{name}.png", dpi=150)
    plt.close()

def ceil_div(a, b):
    return int(math.ceil(float(a) / float(b)))

def add_nodes_col(df_like):
    df_like = df_like.copy()
    def _calc_nodes(row):
        dev = str(row["device"]).lower()
        if dev == "cpu":
            return ceil_div(row["num_procs"], CPU_CORES_PER_NODE)
        else:
            return ceil_div(row["num_procs"], GPUS_PER_NODE)
    df_like["nodes"] = df_like.apply(_calc_nodes, axis=1)
    return df_like

def _pooled_mean_std(values, stds, weights):
    m = np.asarray(values, float)
    s = np.asarray(stds,   float)
    w = np.asarray(weights, float)
    mask = np.isfinite(m) & np.isfinite(s) & np.isfinite(w) & (w > 0)
    m, s, w = m[mask], s[mask], w[mask]
    if m.size == 0:
        return None, None
    N = np.sum(w)
    mean_w = np.sum(w * m) / N
    var_within = np.sum((w - 1.0) * (s ** 2.0))
    var_between = np.sum(w * (m - mean_w) ** 2.0)
    dof = max(N - 1.0, 1.0)
    s_w = math.sqrt(max((var_within + var_between) / dof, 0.0))
    return mean_w, s_w

def combine_powers_2lines(g, mean_col, std_col, axis_col):
    rows = []
    for dev, d in g.groupby("device"):
        for x, dd in d.groupby(axis_col):
            mw, sw = _pooled_mean_std(dd[mean_col], dd[std_col], dd["runs"])
            if mw is None:
                continue
            rows.append({"device": dev, axis_col: x, "y_mean": mw, "y_std": sw})
    out = pd.DataFrame(rows)
    return out.sort_values(axis_col)

def combine_generic_2lines(g, y_col, ystd_col, axis_col):
    g2 = g.rename(columns={y_col:"mean_col", ystd_col:"std_col"})
    g2["runs"] = g2.get("runs", pd.Series([1]*len(g2)))
    agg = combine_powers_2lines(g2, "mean_col", "std_col", axis_col)
    return agg.rename(columns={"y_mean": y_col, "y_std": ystd_col})

# >>> baseline-safe aggregator:
# Garante que no ponto do eixo que coincide com o baseline (n_ref_procs=2 ou n_ref_nodes correspondente),
# a média use SOMENTE as linhas de baseline. Assim, o valor plotado fica exatamente 1.
def combine_perf_2lines_baseline_safe(g, y_col, ystd_col, axis_col, baseline_tags, base_tag):
    enforce = (base_tag in baseline_tags)  # só força para split/dist/sd/total
    rows = []
    for dev, d in g.groupby("device"):
        for x, dd in d.groupby(axis_col):
            sel = dd
            if enforce:
                if axis_col == "num_procs":
                    if (dd["n_ref_procs"] == x).any():
                        sel = dd[dd["n_ref_procs"] == x]
                else:  # axis_col == "nodes"
                    if (dd["n_ref_nodes"] == x).any():
                        sel = dd[(dd["n_ref_nodes"] == x) & (dd["num_procs"] == dd["n_ref_procs"])]
            mw, sw = _pooled_mean_std(sel[y_col], sel[ystd_col], sel["runs"])
            if mw is None:
                continue
            rows.append({"device": dev, axis_col: x, y_col: mw, ystd_col: sw})
    out = pd.DataFrame(rows)
    return out.sort_values(axis_col)

# -----------------------------
# Load & summarize
# -----------------------------
def prepare_data(csv_file, prefix, baseline_nprocs=BASE_EFF_NPROCS):
    results_dir = BASE_RESULTS_DIR / prefix
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file, parse_dates=["datetime"])
    # columns:
    # datetime,power,total_particles,length_per_rank,num_procs,box_length,RAM_GB,gen_time,splitters_time,dist_time,total_time,device,seed,mode

    counts = df.groupby(["device","power","num_procs"]).size()
    print(f"[{prefix}] Runs per configuration:\n{counts}")

    summary = (
        df.groupby(["device","power","num_procs"], as_index=False)
          .agg(
              gen_mean=("gen_time","mean"),           gen_std=("gen_time","std"),
              split_mean=("splitters_time","mean"),   split_std=("splitters_time","std"),
              dist_mean=("dist_time","mean"),         dist_std=("dist_time","std"),
              total_mean=("total_time","mean"),       total_std=("total_time","std"),
              runs=("total_time","count")
          )
    )
    summary = add_nodes_col(summary)
    summary["sd_mean"] = summary["split_mean"] + summary["dist_mean"]
    summary["sd_std"]  = np.sqrt(np.maximum(summary["split_std"]**2 + summary["dist_std"]**2, 0.0))
    for c in ["gen_std","split_std","dist_std","sd_std","total_std"]:
        summary[c] = summary[c].fillna(0.0)

    # --- Baselines ---
    # (A) baseline n=2 por (device,power) — para split/dist/sd/total (speedup & efficiency)
    base_dev2 = (
        summary[summary["num_procs"] == baseline_nprocs]
          .rename(columns={
              "num_procs":"n_ref_procs",
              "nodes":"n_ref_nodes",
              "gen_mean":"gen_ref","gen_std":"gen_ref_s",
              "split_mean":"split_ref","split_std":"split_ref_s",
              "dist_mean":"dist_ref","dist_std":"dist_ref_s",
              "sd_mean":"sd_ref","sd_std":"sd_ref_s",
              "total_mean":"total_ref","total_std":"total_ref_s",
          })[[
              "device","power","n_ref_procs","n_ref_nodes",
              "gen_ref","gen_ref_s","split_ref","split_ref_s",
              "dist_ref","dist_ref_s","sd_ref","sd_ref_s",
              "total_ref","total_ref_s"
          ]]
    )

    # (B) baselines para generation:
    #     - speedup: baseline CPU (menor num_procs por power)
    #     - efficiency: baseline mínimo por (device,power)
    cpu_only = summary[summary["device"] == "cpu"].copy()
    cpu_base_gen = (
        cpu_only.sort_values(["power","num_procs"])
                .groupby("power", as_index=False)
                .first()
                .rename(columns={
                    "num_procs":"cpu_base_nprocs",
                    "gen_mean":"gen_cpu","gen_std":"gen_cpu_s"
                })[["power","cpu_base_nprocs","gen_cpu","gen_cpu_s"]]
    )
    base_min_gen = (
        summary.sort_values(["device","power","num_procs"])
               .groupby(["device","power"], as_index=False)
               .first()
               .rename(columns={
                   "num_procs":"n_ref_procs_min",
                   "nodes":"n_ref_nodes_min",
                   "gen_mean":"gen_ref_min","gen_std":"gen_ref_s_min"
               })[["device","power","n_ref_procs_min","n_ref_nodes_min","gen_ref_min","gen_ref_s_min"]]
    )

    with_base = (summary
        .merge(base_dev2, on=["device","power"], how="left")
        .merge(cpu_base_gen, on="power", how="left")
        .merge(base_min_gen, on=["device","power"], how="left")
    )

    # --- helpers ---
    def ratio(a, b): 
        a = np.asarray(a, float); b = np.asarray(b, float)
        out = np.full_like(a, np.nan, dtype=float)
        ok = (b > 0) & np.isfinite(b)
        out[ok] = a[ok] / b[ok]
        return out

    def ratio_std(a,sa,b,sb):
        a = np.asarray(a, float); sa = np.asarray(sa, float)
        b = np.asarray(b, float); sb = np.asarray(sb, float)
        s = ratio(a,b)
        out = np.full_like(s, np.nan, dtype=float)
        ok = (a>0)&(b>0)&np.isfinite(a)&np.isfinite(b)
        out[ok] = s[ok] * np.sqrt( (sa[ok]/a[ok])**2 + (sb[ok]/b[ok])**2 )
        return out

    # --- SPEEDUP ---
    # split/dist/sd/total: baseline n=2 (speedup = T_ref(n=2)/T_n)
    for tag, mcol, scol, rcol, rscol in [
        ("split","split_mean","split_std","split_ref","split_ref_s"),
        ("dist","dist_mean","dist_std","dist_ref","dist_ref_s"),
        ("sd","sd_mean","sd_std","sd_ref","sd_ref_s"),
        ("total","total_mean","total_std","total_ref","total_ref_s"),
    ]:
        with_base[f"speedup_{tag}"]   = ratio(with_base[rcol], with_base[mcol])
        with_base[f"speedup_{tag}_s"] = ratio_std(with_base[rcol], with_base[rscol], with_base[mcol], with_base[scol])

    # generation: baseline CPU (menor num_procs por power)
    with_base["speedup_gen"]   = ratio(with_base["gen_cpu"], with_base["gen_mean"])
    with_base["speedup_gen_s"] = ratio_std(with_base["gen_cpu"], with_base["gen_cpu_s"], with_base["gen_mean"], with_base["gen_std"])

    # --- EFFICIENCY ---
    # split/dist/sd/total: baseline n=2 (por processadores e por nós)
    for tag, mcol, scol, rcol, rscol in [
        ("split","split_mean","split_std","split_ref","split_ref_s"),
        ("dist","dist_mean","dist_std","dist_ref","dist_ref_s"),
        ("sd","sd_mean","sd_std","sd_ref","sd_ref_s"),
        ("total","total_mean","total_std","total_ref","total_ref_s"),
    ]:
        ok = (with_base["n_ref_procs"]>0) & (with_base[mcol]>0) & np.isfinite(with_base[mcol]) & np.isfinite(with_base["num_procs"])
        with_base.loc[ok, f"eff_procs_{tag}"] = (with_base.loc[ok, rcol] * with_base.loc[ok, "n_ref_procs"]) / (with_base.loc[ok, mcol] * with_base.loc[ok, "num_procs"])
        with_base.loc[ok, f"eff_procs_{tag}_s"] = with_base.loc[ok, f"eff_procs_{tag}"] * np.sqrt(
            (with_base.loc[ok, rscol] / with_base.loc[ok, rcol])**2 + (with_base.loc[ok, scol] / with_base.loc[ok, mcol])**2
        )

        ok2 = (with_base["n_ref_nodes"]>0) & (with_base[mcol]>0) & np.isfinite(with_base[mcol]) & np.isfinite(with_base["nodes"])
        with_base.loc[ok2, f"eff_nodes_{tag}"] = (with_base.loc[ok2, rcol] * with_base.loc[ok2, "n_ref_nodes"]) / (with_base.loc[ok2, mcol] * with_base.loc[ok2, "nodes"])
        with_base.loc[ok2, f"eff_nodes_{tag}_s"] = with_base.loc[ok2, f"eff_nodes_{tag}"] * np.sqrt(
            (with_base.loc[ok2, rscol] / with_base.loc[ok2, rcol])**2 + (with_base.loc[ok2, scol] / with_base.loc[ok2, mcol])**2
        )

    # generation: baseline mínimo por (device,power)
    okg = (with_base["n_ref_procs_min"]>0) & (with_base["gen_mean"]>0) & np.isfinite(with_base["gen_mean"]) & np.isfinite(with_base["num_procs"])
    with_base.loc[okg, "eff_procs_gen"] = (with_base.loc[okg, "gen_ref_min"]*with_base.loc[okg, "n_ref_procs_min"])/(with_base.loc[okg, "gen_mean"]*with_base.loc[okg, "num_procs"])
    with_base.loc[okg, "eff_procs_gen_s"] = with_base.loc[okg, "eff_procs_gen"] * np.sqrt(
        (with_base.loc[okg, "gen_ref_s_min"]/with_base.loc[okg, "gen_ref_min"])**2 + (with_base.loc[okg, "gen_std"]/with_base.loc[okg, "gen_mean"])**2
    )
    okg2 = (with_base["n_ref_nodes_min"]>0) & (with_base["gen_mean"]>0) & np.isfinite(with_base["gen_mean"]) & np.isfinite(with_base["nodes"])
    with_base.loc[okg2, "eff_nodes_gen"] = (with_base.loc[okg2, "gen_ref_min"]*with_base.loc[okg2, "n_ref_nodes_min"])/(with_base.loc[okg2, "gen_mean"]*with_base.loc[okg2, "nodes"])
    with_base.loc[okg2, "eff_nodes_gen_s"] = with_base.loc[okg2, "eff_nodes_gen"] * np.sqrt(
        (with_base.loc[okg2, "gen_ref_s_min"]/with_base.loc[okg2, "gen_ref_min"])**2 + (with_base.loc[okg2, "gen_std"]/with_base.loc[okg2, "gen_mean"])**2
    )

    return df, summary, with_base, results_dir

# -----------------------------
# Plots (2 lines: CPU & GPU)
# -----------------------------
def plot_time_vs_axis(summary, axis_col, file_tag, mean_col, std_col, results_dir, prefix):
    agg = combine_powers_2lines(summary, mean_col, std_col, axis_col)
    if agg.empty:
        return
    plt.figure(figsize=(9,5))
    for dev, g in agg.groupby("device"):
        plt.errorbar(g[axis_col], g["y_mean"], yerr=g["y_std"], marker="o", capsize=4, label=dev)
    if axis_col == "num_procs":
        plt.xscale("log"); plt.xlabel("Processors (num_procs)"); axis_name = "procs"
    else:
        plt.xlabel("Nodes"); axis_name = "nodes"
    ylabels = {
        "generation":"Generation time (s)",
        "splitters":"Find splitters time (s)",
        "distribution":"Data distribution time (s)",
        "split_plus_dist":"Find+Distribution time (s)",
        "total":"Total time (s)"
    }
    plt.ylabel(ylabels.get(file_tag, "Time (s)"))
    plt.title(f"Time × {'processors' if axis_col=='num_procs' else 'nodes'} — {file_tag}")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    savefig(results_dir, f"time_{axis_name}_{file_tag}_{prefix}")

def plot_speedup_vs_axis(with_base, axis_col, file_tag, base_tag, results_dir, prefix):
    y  = f"speedup_{base_tag}"
    ys = f"speedup_{base_tag}_s"
    baseline_tags = {"split","dist","sd","total"}
    if base_tag in baseline_tags:
        agg = combine_perf_2lines_baseline_safe(with_base, y, ys, axis_col, baseline_tags, base_tag)
    else:
        agg = combine_generic_2lines(with_base, y, ys, axis_col)
    if agg.empty:
        return
    plt.figure(figsize=(9,5))
    for dev, g in agg.groupby("device"):
        plt.errorbar(g[axis_col], g[y], yerr=g[ys], marker="o", capsize=4, label=dev)
    if axis_col == "num_procs":
        plt.xscale("log"); plt.xlabel("Processors (num_procs)"); axis_name = "procs"
    else:
        plt.xlabel("Nodes"); axis_name = "nodes"
    plt.ylabel("Speedup")
    suffix = "(baseline n=2)" if base_tag in baseline_tags else "(baseline min)"
    plt.title(f"Speedup × {'processors' if axis_col=='num_procs' else 'nodes'} — {file_tag} {suffix}")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    savefig(results_dir, f"speedup_{axis_name}_{file_tag}_{prefix}")

def plot_efficiency_vs_axis(with_base, axis_col, file_tag, base_tag, results_dir, prefix):
    if axis_col == "num_procs":
        y, ys = f"eff_procs_{base_tag}", f"eff_procs_{base_tag}_s"
        axis_name, xlabel = "procs", "Processors (num_procs)"
    else:
        y, ys = f"eff_nodes_{base_tag}", f"eff_nodes_{base_tag}_s"
        axis_name, xlabel = "nodes", "Nodes"

    baseline_tags = {"split","dist","sd","total"}
    if base_tag in baseline_tags:
        agg = combine_perf_2lines_baseline_safe(with_base, y, ys, axis_col, baseline_tags, base_tag)
    else:
        agg = combine_generic_2lines(with_base, y, ys, axis_col)

    if agg.empty:
        return
    plt.figure(figsize=(9,5))
    for dev, g in agg.groupby("device"):
        plt.errorbar(g[axis_col], g[y], yerr=g[ys], marker="o", capsize=4, label=dev)
    if axis_col == "num_procs":
        plt.xscale("log")
    suffix = "(baseline n=2)" if base_tag in baseline_tags else "(baseline min)"
    plt.xlabel(xlabel); plt.ylabel("Parallel efficiency")
    plt.title(f"Parallel efficiency × {'processors' if axis_col=='num_procs' else 'nodes'} — {file_tag} {suffix}")
    plt.axhline(1.0, ls="--", lw=1)
    plt.grid(True, which="both", ls=":")
    plt.legend()
    savefig(results_dir, f"efficiency_{axis_name}_{file_tag}_{prefix}")

# -----------------------------
# Run for weak/strong
# -----------------------------
def run_all(csv_file, prefix):
    df, summary, with_base, results_dir = prepare_data(csv_file, prefix)

    # file_tag -> (mean_col, std_col, base_tag)
    metrics = {
        "generation":      ("gen_mean",   "gen_std",   "gen"),
        "splitters":       ("split_mean", "split_std", "split"),
        "distribution":    ("dist_mean",  "dist_std",  "dist"),
        "split_plus_dist": ("sd_mean",    "sd_std",    "sd"),
        "total":           ("total_mean", "total_std", "total"),
    }

    # Time × nodes / processors
    for file_tag, (m, s, _b) in metrics.items():
        plot_time_vs_axis(summary, "nodes",     file_tag, m, s, results_dir, prefix)
        plot_time_vs_axis(summary, "num_procs", file_tag, m, s, results_dir, prefix)

    # Speedup × nodes / processors
    for file_tag, (_m, _s, base_tag) in metrics.items():
        plot_speedup_vs_axis(with_base, "nodes",     file_tag, base_tag, results_dir, prefix)
        plot_speedup_vs_axis(with_base, "num_procs", file_tag, base_tag, results_dir, prefix)

    # Parallel efficiency × nodes / processors
    for file_tag, (_m, _s, base_tag) in metrics.items():
        plot_efficiency_vs_axis(with_base, "nodes",     file_tag, base_tag, results_dir, prefix)
        plot_efficiency_vs_axis(with_base, "num_procs", file_tag, base_tag, results_dir, prefix)

# %%
run_all("results_weak.csv", "weak")
run_all("results_strong.csv", "strong")
