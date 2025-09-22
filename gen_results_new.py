# %% [markdown]
# # Weak and Strong Scaling Analysis

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% 
# Função principal que gera todas as figuras a partir de um CSV
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
                    BASELINE_NPROCS=None):
    
    df = pd.read_csv(csv_file, parse_dates=["datetime"])
    df["total_time"] = df["gen_time"] + df["dist_time"]

    counts = df.groupby(["device", "power", "num_procs"]).size()
    print(f"\n[{prefix}] Execuções por configuração:")
    print(counts)

    # --- resumo ---
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
    print(f"\n[{prefix}] Resumo por configuração:")
    print(summary)

    # --- função auxiliar para salvar figuras ---
    def savefig(name):
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{prefix}_{name}.png", dpi=150)
        plt.close()

    # --- Gráfico de barras ---
    if PLOT_BARS:
        plt.figure(figsize=(12, 6))
        for i, (key, group) in enumerate(df.groupby(["device", "power", "num_procs"])):
            label = f"{key[0]}_p{key[1]}_np{key[2]}"
            xpos = range(i*len(group), (i+1)*len(group))
            plt.bar(xpos, group["total_time"], label=label)
        tick_positions = []
        tick_labels = []
        for i, (key, group) in enumerate(df.groupby(["device", "power", "num_procs"])):
            center = i*len(group) + len(group)/2 - 0.5
            tick_positions.append(center)
            tick_labels.append(f"{key[0]}_p{key[1]}_np{key[2]}")
        plt.xticks(tick_positions, tick_labels, rotation=20, ha="right")
        plt.ylabel("Total time (s)")
        plt.xlabel("Runs")
        plt.title("Total time per run")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        savefig("bars_total_time")

    # --- Gráficos médios (dist, gen, total) ---
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

    # --- baseline CPU ---
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

    with_base["speedup_dist"]   = ratio(with_base["dist_cpu"],  with_base["dist_mean"])
    with_base["speedup_dist_s"] = ratio_std(with_base["dist_cpu"], with_base["dist_cpu_std"], with_base["dist_mean"], with_base["dist_std"])
    with_base["speedup_gen"]    = ratio(with_base["gen_cpu"],   with_base["gen_mean"])
    with_base["speedup_gen_s"]  = ratio_std(with_base["gen_cpu"],  with_base["gen_cpu_std"], with_base["gen_mean"],  with_base["gen_std"])
    with_base["speedup_total"]  = ratio(with_base["total_cpu"], with_base["total_mean"])
    with_base["speedup_total_s"]= ratio_std(with_base["total_cpu"],with_base["total_cpu_std"],with_base["total_mean"],with_base["total_std"])

    def plot_speedup(df, y, yerr, title, fname):
        plt.figure(figsize=(8,5))
        for dev in df["device"].unique():
            sub = df[df["device"] == dev].sort_values("num_procs")
            plt.errorbar(sub["num_procs"], sub[y], yerr=sub[yerr], marker="o", capsize=4, label=dev)
        plt.axhline(1.0, ls="--", lw=1)
        for pw in sorted(df["power"].unique()):
            g = df[df["power"] == pw]
            if g.empty: continue
            nref = g["cpu_base_nprocs"].iloc[0]
            xs = sorted(g["num_procs"].unique())
            ys = [x/nref for x in xs]
            plt.plot(xs, ys, ls=":", lw=1.5, label=f"Ideal (p{pw}, N/Nref={nref})")
        plt.xlabel("Number of processors"); plt.ylabel("Speedup")
        plt.title(title); plt.legend(); plt.grid(True)
        savefig(fname)

    if PLOT_SPEEDUP_DIST:  plot_speedup(with_base,"speedup_dist","speedup_dist_s","Speedup vs CPU (Distribution)","speedup_distribution")
    if PLOT_SPEEDUP_GEN:   plot_speedup(with_base,"speedup_gen","speedup_gen_s","Speedup vs CPU (Generation)","speedup_generation")
    if PLOT_SPEEDUP_TOTAL: plot_speedup(with_base,"speedup_total","speedup_total_s","Speedup vs CPU (Total)","speedup_total")

    # --- Eficiência total ---
    ref = (summary.sort_values(["device","power","num_procs"]).groupby(["device","power"],as_index=False).first())
    ref = ref.rename(columns={"num_procs":"n_ref","total_mean":"t_ref","total_std":"s_ref"})[["device","power","n_ref","t_ref","s_ref"]]
    eff = summary.merge(ref,on=["device","power"],how="inner")
    eff["eff_strong"] = (eff["t_ref"]*eff["n_ref"])/(eff["total_mean"]*eff["num_procs"])
    good = (eff["t_ref"]>0)&(eff["total_mean"]>0)&np.isfinite(eff["t_ref"])&np.isfinite(eff["total_mean"])
    eff["eff_strong_s"] = np.nan
    eff.loc[good,"eff_strong_s"] = eff.loc[good,"eff_strong"]*np.sqrt((eff.loc[good,"s_ref"]/eff.loc[good,"t_ref"])**2+(eff.loc[good,"total_std"]/eff.loc[good,"total_mean"])**2)

    if PLOT_EFF_TOTAL:
        plt.figure(figsize=(8,5))
        for (dev,pw),g in eff.sort_values("num_procs").groupby(["device","power"]):
            plt.errorbar(g["num_procs"],g["eff_strong"],yerr=g["eff_strong_s"],marker="o",capsize=4,label=f"{dev} (p{pw})")
        plt.axhline(1.0,ls="--")
        plt.xlabel("Number of processors"); plt.ylabel("Parallel efficiency (strong)")
        plt.title("Parallel efficiency (Total)")
        plt.legend(); plt.grid(True)
        savefig("efficiency_total")

    # --- Eficiência dist/gen ---
    def plot_eff(summary, mean_col, std_col, title, fname):
        ref = (summary.sort_values(["device","power","num_procs"]).groupby(["device","power"],as_index=False).first())
        ref = ref.rename(columns={"num_procs":"n_ref", mean_col:"t_ref", std_col:"s_ref"})[["device","power","n_ref","t_ref","s_ref"]]
        e2 = summary.merge(ref,on=["device","power"],how="inner")
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
        plot_eff(summary,"dist_mean","dist_std","Parallel efficiency — Distribution","efficiency_distribution")
        plot_eff(summary,"gen_mean","gen_std","Parallel efficiency — Generation","efficiency_generation")


# %% 
# Executa análise para os dois CSVs
analyze_scaling("results_strong.csv", "strong")
analyze_scaling("results_weak.csv", "weak")
