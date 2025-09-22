# %% [markdown]
# # Weak and Strong Scaling Analysis

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math

BASE_RESULTS_DIR = Path("results")
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %%
def analyze_scaling(csv_file, prefix, 
                    PLOT_BARS=True, 
                    PLOT_DIST_MEAN=True, 
                    PLOT_GEN_MEAN=True, 
                    PLOT_TOTAL_MEAN=True, 
                    PLOT_SPEEDUP_DIST=True, 
                    PLOT_SPEEDUP_GEN=True, 
                    PLOT_SPEEDUP_TOTAL=True, 
                    PLOT_EFF_TOTAL=True, 
                    PLOT_EFF_DIST_GEN=True,
                    PLOT_TIME_VS_NODES=True,
                    BASELINE_NPROCS=None):

    results_dir = BASE_RESULTS_DIR / prefix
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file, parse_dates=["datetime"])
    df["total_time"] = df["gen_time"] + df["dist_time"]

    counts = df.groupby(["device", "power", "num_procs"]).size()
    print(f"\n[{prefix}] Execuções por configuração:")
    print(counts)

    summary = (
        df.groupby(["device", "power", "num_procs"])
          .agg(
              gen_mean=("gen_time", "mean"),
              gen_std=("gen_time", "std"),
              dist_mean=("dist_time", "mean"),
              dist_std=("dist_time", "std"),
              total_mean=("total_time", "mean"),
              total_std=("total_time", "std"),
              runs=("total_time", "count")
          )
          .reset_index()
    )

    def calc_nodes(row):
        dev = str(row["device"]).lower()
        if dev == "cpu":
            return int(math.ceil(row["num_procs"] / 128.0))
        else:
            return int(math.ceil(row["num_procs"] / 4.0))
    summary["nodes"] = summary.apply(calc_nodes, axis=1)

    print(f"\n[{prefix}] Resumo por configuração:")
    print(summary)

    def savefig(name):
        plt.tight_layout()
        plt.savefig(results_dir / f"{name}.png", dpi=150)
        plt.show()
        plt.close()

    if PLOT_BARS:
        plt.figure(figsize=(12, 6))
        for i, (key, group) in enumerate(df.groupby(["device", "power", "num_procs"])):
            label = f"{key[0]}_p{key[1]}_np{key[2]}"
            xpos = range(i*len(group), (i+1)*len(group))
            plt.bar(xpos, group["total_time"], label=label)
        tick_positions, tick_labels = [], []
        for i, (key, group) in enumerate(df.groupby(["device", "power", "num_procs"])):
            center = i*len(group) + len(group)/2 - 0.5
            tick_positions.append(center)
            tick_labels.append(f"{key[0]}_p{key[1]}_np{key[2]}")
        plt.xticks(tick_positions, tick_labels, rotation=20, ha="right")
        plt.ylabel("Total time (s)")
        plt.xlabel("Runs")
        plt.title("Total time per run (5 runs expected per configuration)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        savefig("bars_total_time")

    if PLOT_DIST_MEAN:
        plt.figure(figsize=(8, 5))
        for dev in summary["device"].unique():
            subset = summary[summary["device"] == dev].sort_values("num_procs")
            plt.errorbar(subset["num_procs"], subset["dist_mean"], yerr=subset["dist_std"], marker="o", capsize=4, label=f"{dev} (dist)")
        plt.xlabel("Number of processors"); plt.ylabel("Distribution time (s)")
        plt.title("CPU vs GPU — Distribution"); plt.legend(); plt.grid(True)
        savefig("distribution_time")

    if PLOT_GEN_MEAN:
        plt.figure(figsize=(8, 5))
        for dev in summary["device"].unique():
            subset = summary[summary["device"] == dev].sort_values("num_procs")
            plt.errorbar(subset["num_procs"], subset["gen_mean"], yerr=subset["gen_std"], marker="o", capsize=4, label=f"{dev} (gen)")
        plt.xlabel("Number of processors"); plt.ylabel("Generation time (s)")
        plt.title("CPU vs GPU — Generation"); plt.legend(); plt.grid(True)
        savefig("generation_time")

    if PLOT_TOTAL_MEAN:
        plt.figure(figsize=(8, 5))
        for dev in summary["device"].unique():
            subset = summary[summary["device"] == dev].sort_values("num_procs")
            plt.errorbar(subset["num_procs"], subset["total_mean"], yerr=subset["total_std"], marker="o", capsize=4, label=f"{dev} (total)")
        plt.xlabel("Number of processors"); plt.ylabel("Total time (s)")
        plt.title("CPU vs GPU — Total"); plt.legend(); plt.grid(True)
        savefig("total_time")

    cpu_only = summary[summary["device"] == "cpu"].copy()
    if BASELINE_NPROCS is None:
        cpu_base = (cpu_only.sort_values(["power", "num_procs"])
                            .groupby("power", as_index=False)
                            .first())
    else:
        cpu_base = cpu_only[cpu_only["num_procs"] == BASELINE_NPROCS].copy()
        if cpu_base.empty:
            raise ValueError(f"No CPU baseline with num_procs={BASELINE_NPROCS}.")
    cpu_base = cpu_base.rename(columns={
        "num_procs": "cpu_base_nprocs",
        "gen_mean": "gen_cpu", "gen_std": "gen_cpu_std",
        "dist_mean": "dist_cpu", "dist_std": "dist_cpu_std",
        "total_mean": "total_cpu", "total_std": "total_cpu_std",
    })[["power","cpu_base_nprocs","gen_cpu","gen_cpu_std","dist_cpu","dist_cpu_std","total_cpu","total_cpu_std"]]

    with_base = summary.merge(cpu_base, on="power", how="inner")

    def ratio(a,b): return np.where((b>0)&np.isfinite(b), a/b, np.nan)
    def ratio_std(a,sa,b,sb):
        s = ratio(a,b); term = np.zeros_like(s,float)
        good = (a>0)&(b>0)&np.isfinite(a)&np.isfinite(b)
        term[good] = np.sqrt((sa[good]/a[good])**2 + (sb[good]/b[good])**2)
        out = np.full_like(s, np.nan, float); out[good] = s[good]*term[good]; return out

    if PLOT_SPEEDUP_DIST:
        with_base["speedup_dist"]   = ratio(with_base["dist_cpu"], with_base["dist_mean"])
        with_base["speedup_dist_s"] = ratio_std(with_base["dist_cpu"], with_base["dist_cpu_std"], with_base["dist_mean"], with_base["dist_std"])
        plt.figure(figsize=(8,5))
        for dev in with_base["device"].unique():
            sub = with_base[with_base["device"] == dev].sort_values("num_procs")
            plt.errorbar(sub["num_procs"], sub["speedup_dist"], yerr=sub["speedup_dist_s"], marker="o", capsize=4, label=dev)
        plt.axhline(1.0, ls="--", lw=1)
        for pw in sorted(with_base["power"].unique()):
            g = with_base[with_base["power"] == pw]
            if g.empty: continue
            nref = g["cpu_base_nprocs"].iloc[0]
            xs = sorted(g["num_procs"].unique())
            ys = [x/nref for x in xs]
            plt.plot(xs, ys, ls=":", lw=1.5, label=f"Ideal (p{pw}, N/Nref={nref})")
        plt.xlabel("Number of processors"); plt.ylabel("Speedup (distribution)")
        plt.title("Speedup vs CPU (Distribution)"); plt.legend(); plt.grid(True)
        savefig("speedup_distribution")

    if PLOT_SPEEDUP_GEN:
        with_base["speedup_gen"]   = ratio(with_base["gen_cpu"], with_base["gen_mean"])
        with_base["speedup_gen_s"] = ratio_std(with_base["gen_cpu"], with_base["gen_cpu_std"], with_base["gen_mean"], with_base["gen_std"])
        plt.figure(figsize=(8,5))
        for dev in with_base["device"].unique():
            sub = with_base[with_base["device"] == dev].sort_values("num_procs")
            plt.errorbar(sub["num_procs"], sub["speedup_gen"], yerr=sub["speedup_gen_s"], marker="o", capsize=4, label=dev)
        plt.axhline(1.0, ls="--", lw=1)
        for pw in sorted(with_base["power"].unique()):
            g = with_base[with_base["power"] == pw]
            if g.empty: continue
            nref = g["cpu_base_nprocs"].iloc[0]
            xs = sorted(g["num_procs"].unique())
            ys = [x/nref for x in xs]
            plt.plot(xs, ys, ls=":", lw=1.5, label=f"Ideal (p{pw}, N/Nref={nref})")
        plt.xlabel("Number of processors"); plt.ylabel("Speedup (generation)")
        plt.title("Speedup vs CPU (Generation)"); plt.legend(); plt.grid(True)
        savefig("speedup_generation")

    if PLOT_SPEEDUP_TOTAL:
        with_base["speedup_total"]   = ratio(with_base["total_cpu"], with_base["total_mean"])
        with_base["speedup_total_s"] = ratio_std(with_base["total_cpu"], with_base["total_cpu_std"], with_base["total_mean"], with_base["total_std"])
        plt.figure(figsize=(8,5))
        for dev in with_base["device"].unique():
            sub = with_base[with_base["device"] == dev].sort_values("num_procs")
            plt.errorbar(sub["num_procs"], sub["speedup_total"], yerr=sub["speedup_total_s"], marker="o", capsize=4, label=dev)
        plt.axhline(1.0, ls="--", lw=1)
        for pw in sorted(with_base["power"].unique()):
            g = with_base[with_base["power"] == pw]
            if g.empty: continue
            nref = g["cpu_base_nprocs"].iloc[0]
            xs = sorted(with_base["num_procs"].unique())
            ys = [x/nref for x in xs]
            plt.plot(xs, ys, ls=":", lw=1.5, label=f"Ideal (p{pw}, N/Nref={nref})")
        plt.xlabel("Number of processors"); plt.ylabel("Speedup (total)")
        plt.title("Speedup vs CPU (Total)"); plt.legend(); plt.grid(True)
        savefig("speedup_total")

    if PLOT_EFF_TOTAL or PLOT_EFF_DIST_GEN:
        ref = (summary.sort_values(["device","power","num_procs"]).groupby(["device","power"],as_index=False).first())
        ref_total = ref.rename(columns={"num_procs":"n_ref","total_mean":"t_ref","total_std":"s_ref"})[["device","power","n_ref","t_ref","s_ref"]]
        eff_total = summary.merge(ref_total,on=["device","power"],how="inner")
        eff_total["eff_strong"] = (eff_total["t_ref"]*eff_total["n_ref"])/(eff_total["total_mean"]*eff_total["num_procs"])
        good = (eff_total["t_ref"]>0)&(eff_total["total_mean"]>0)&np.isfinite(eff_total["t_ref"])&np.isfinite(eff_total["total_mean"])
        eff_total["eff_strong_s"] = np.nan
        eff_total.loc[good,"eff_strong_s"] = eff_total.loc[good,"eff_strong"]*np.sqrt(
            (eff_total.loc[good,"s_ref"]/eff_total["t_ref"])**2 + (eff_total.loc[good,"total_std"]/eff_total["total_mean"])**2
        )
        if PLOT_EFF_TOTAL:
            plt.figure(figsize=(8,5))
            for (dev,pw),g in eff_total.sort_values("num_procs").groupby(["device","power"]):
                plt.errorbar(g["num_procs"],g["eff_strong"],yerr=g["eff_strong_s"],marker="o",capsize=4,label=f"{dev} (p{pw})")
            plt.axhline(1.0,ls="--")
            plt.xlabel("Number of processors"); plt.ylabel("Parallel efficiency (strong)")
            plt.title("Parallel efficiency (Total)"); plt.legend(); plt.grid(True)
            savefig("efficiency_total")

        def plot_eff(mean_col, std_col, title, fname):
            ref_x = ref.rename(columns={"num_procs":"n_ref", mean_col:"t_ref", std_col:"s_ref"})[["device","power","n_ref","t_ref","s_ref"]]
            e2 = summary.merge(ref_x,on=["device","power"],how="inner")
            e2["eff"] = (e2["t_ref"]*e2["n_ref"])/(e2[mean_col]*e2["num_procs"])
            good = (e2["t_ref"]>0)&(e2[mean_col]>0)&np.isfinite(e2["t_ref"])&np.isfinite(e2[mean_col])
            e2["eff_s"] = np.nan
            e2.loc[good,"eff_s"] = e2.loc[good,"eff"]*np.sqrt((e2.loc[good,"s_ref"]/e2.loc[good,"t_ref"])**2+(e2.loc[good,std_col]/e2.loc[good,mean_col])**2)
            plt.figure(figsize=(8,5))
            for (dev,pw),g in e2.sort_values("num_procs").groupby(["device","power"]):
                plt.errorbar(g["num_procs"],g["eff"],yerr=g["eff_s"],marker="o",capsize=4,label=f"{dev} (p{pw})")
            plt.axhline(1.0,ls="--")
            plt.xlabel("Number of processors"); plt.ylabel("Parallel efficiency (strong)")
            plt.title(title); plt.legend(); plt.grid(True)
            savefig(fname)

        if PLOT_EFF_DIST_GEN:
            plot_eff("dist_mean","dist_std","Parallel efficiency — Distribution","efficiency_distribution")
            plot_eff("gen_mean","gen_std","Parallel efficiency — Generation","efficiency_generation")

    if PLOT_TIME_VS_NODES:
        def plot_vs_nodes(y_mean, y_std, title, fname, y_label):
            plt.figure(figsize=(8,5))
            agg_nodes = (summary.groupby(["device","power","nodes"], as_index=False)
                                .agg(y_mean=(y_mean,"mean"),
                                     y_std=(y_std,"mean")))
            for (dev, pw), g in agg_nodes.sort_values(["nodes"]).groupby(["device","power"]):
                plt.errorbar(g["nodes"], g["y_mean"], yerr=g["y_std"],
                             marker="o", capsize=4, label=f"{dev} (p{pw})")
            plt.xlabel("Number of nodes used")
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend(); plt.grid(True)
            savefig(fname)

        plot_vs_nodes("dist_mean","dist_std","Distribution time vs nodes used","time_vs_nodes_distribution","Distribution time (s)")
        plot_vs_nodes("gen_mean","gen_std","Generation time vs nodes used","time_vs_nodes_generation","Generation time (s)")
        plot_vs_nodes("total_mean","total_std","Total time vs nodes used","time_vs_nodes_total","Total time (s)")

# %%
#analyze_scaling("results_strong.csv", "strong")
analyze_scaling("results_weak.csv", "weak")
