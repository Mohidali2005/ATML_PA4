import argparse
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.utils import load_json,load_yaml,repo_path,save_json
from task1_compress.data import hellaswag

FAMILY = {"fp16":"FP16","p1":"P1 magnitude","p2":"P2 Wanda","p3":"P3 adaptive Wanda","q1":"Q1 round to nearest",
          "q2":"Q2 activation aware","q3":"Q3 mixed precision"}
COLOR = {"fp16":"black","p1":"steelblue","p2":"darkorange","p3":"seagreen","q1":"steelblue","q2":"darkorange",
         "q3":"seagreen","p2_24":"firebrick"}
PROJ_COLOR = {"q_proj":"steelblue","k_proj":"darkorange","v_proj":"seagreen","o_proj":"firebrick",
              "gate_proj":"mediumpurple","up_proj":"saddlebrown","down_proj":"gray"}
PAIRS = [("p3_s50","p2_s50"),("p3_s75","p2_s75"),("q3_int4","q2_b4"),("q3_int3","q2_b3")]
# matched method comparisons beyond the two adaptive ones
METHOD_PAIRS = [("p2_s25","p1_s25"),("p2_s50","p1_s50"),("p2_s75","p1_s75"),("p2_24","p2_s50"),
                ("q2_b8","q1_b8"),("q2_b4","q1_b4"),("q2_b3","q1_b3")]


def family(name):
    """Method family of a condition name"""
    return name if name == "fp16" else name[:2]


def table(res,name,kind):
    """Read one saved per window or per row table of a condition"""
    return pd.read_csv(res/"conditions"/name/f"{kind}.csv")


def boot_ppl(a,b,n_boot,seed):
    """Paired bootstrap over whole windows of the perplexity difference a minus b"""
    rng = np.random.default_rng(seed)
    assert (a["window"].values == b["window"].values).all()
    idx = rng.integers(0,len(a),(n_boot,len(a)))
    tok = a["tokens"].values[idx].sum(1)
    pa = np.exp(a["nll_sum"].values[idx].sum(1)/tok)
    pb = np.exp(b["nll_sum"].values[idx].sum(1)/tok)
    point = math.exp(a["nll_sum"].sum()/a["tokens"].sum())-math.exp(b["nll_sum"].sum()/b["tokens"].sum())
    return point,*np.percentile(pa-pb,[2.5,97.5])


def boot_rows(a,b,col,n_boot,seed):
    """Paired bootstrap over hellaswag rows of the accuracy difference a minus b in points"""
    rng = np.random.default_rng(seed)
    m = a.merge(b,on="order",suffixes=("_a","_b"))
    diff = 100*((m[f"{col}_a"] == m["gold_a"]).astype(float)-(m[f"{col}_b"] == m["gold_b"]).astype(float)).values
    idx = rng.integers(0,len(diff),(n_boot,len(diff)))
    return diff.mean(),*np.percentile(diff[idx].mean(1),[2.5,97.5]),len(diff)


def compare(res,a,b,n_boot,seed):
    """All paired differences between two conditions with their intervals"""
    rows = []
    for suite in ("wikitext","c4"):
        d,lo,hi = boot_ppl(table(res,a,f"windows_{suite}"),table(res,b,f"windows_{suite}"),n_boot,seed)
        rows.append({"a":a,"b":b,"metric":f"{suite}_ppl","diff":d,"lo":lo,"hi":hi,"n":len(table(res,a,f"windows_{suite}"))})
    for col,metric in (("pred","acc"),("pred_norm","acc_norm")):
        d,lo,hi,n = boot_rows(table(res,a,"hellaswag"),table(res,b,"hellaswag"),col,n_boot,seed)
        rows.append({"a":a,"b":b,"metric":f"hellaswag_{metric}_points","diff":d,"lo":lo,"hi":hi,"n":n})
    return rows


def subset_acc(res,name,count):
    """Normalized hellaswag accuracy on the first rows of the manifest"""
    h = table(res,name,"hellaswag").sort_values("order").head(count)
    return (h["pred_norm"] == h["gold"]).mean()


def summary(res,scores,cfg,n_boot):
    """One row per condition with its paired change from the unchanged model"""
    rows = []
    for name,s in scores.items():
        storage = s.get("packed_bytes",s.get("sparse_format_bytes",s.get("dense_fp16_bytes")))
        row = {"condition":name,"family":FAMILY[family(name)],"sparsity":s["sparsity"],"nominal_bits":s.get("nominal_bits"),
               "storage_gb":storage/1e9,"calib_nll":s["calib_nll"],"dev_ppl":s["dev"]["ppl"],
               "wikitext_ppl":s["wikitext"]["ppl"],"c4_ppl":s["c4"]["ppl"],"hellaswag_rows":s["hellaswag"]["rows"],
               "acc":100*s["hellaswag"]["acc"],"acc_norm":100*s["hellaswag"]["acc_norm"],
               "acc_norm_first512":100*subset_acc(res,name,cfg["eval"]["hellaswag_subset"]),
               "mean_rel_error":table(res,name,"calib_errors")["rel_error"].mean() if name != "fp16" else 0.0}
        if name != "fp16":
            for r in compare(res,name,"fp16",n_boot,cfg["seed"]):
                key = r["metric"].replace("_points","")
                row[f"d_{key}"],row[f"d_{key}_lo"],row[f"d_{key}_hi"] = r["diff"],r["lo"],r["hi"]
        rows.append(row)
    return pd.DataFrame(rows)


def carryover(res,scores,cfg):
    """Development nll and held out perplexity of every allocation candidate with both rankings"""
    rows = []
    for kind,prefix in (("p3","p3_s"),("q3","q3_int")):
        dev = pd.read_csv(res/"selection"/f"{kind}_candidates_dev.csv")
        audit = pd.read_csv(res/"selection"/f"{kind}_audit_test.csv")
        choice = load_json(res/"selection"/f"{kind}_selection.json")
        for target,group in dev.groupby("target"):
            label = round(100*target) if kind == "p3" else int(target)
            picked = choice[str(target)]["selected"]
            for _,r in group.iterrows():
                if r["design"] == picked:
                    s = scores[f"{prefix}{label}"]
                    wiki,c4 = s["wikitext"]["ppl"],s["c4"]["ppl"]
                else:
                    a = audit[(audit.target == target) & (audit.design == r["design"])].iloc[0]
                    wiki,c4 = a["wikitext"],a["c4"]
                rows.append({"method":kind,"target":target,"design":r["design"],"selected":r["design"] == picked,
                             "dev_nll":r["dev_nll"],"wikitext_ppl":wiki,"c4_ppl":c4})
    df = pd.DataFrame(rows)
    for col in ("dev_nll","wikitext_ppl","c4_ppl"):
        df[f"{col}_rank"] = df.groupby(["method","target"])[col].rank(method="min").astype(int)
    return df


def legend_below(fig,axes,ncol):
    """Put one shared legend under the panels"""
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=ncol,frameon=False,bbox_to_anchor=(0.5,-0.02))


def plain_log(ax,axis,ticks):
    """Label a log axis with plain numbers at the given ticks"""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    target.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_:f"{v:g}"))
    target.set_minor_formatter(matplotlib.ticker.NullFormatter())


def tradeoff_figures(df,out):
    """Quality against sparsity for pruning and against packed bytes for quantization"""
    metrics = [("wikitext_ppl","WikiText-2 perplexity",True),("c4_ppl","C4 perplexity",True),
               ("acc_norm_first512","HellaSwag acc_norm (%)",False)]
    base = df[df.condition == "fp16"].iloc[0]
    fig,axes = plt.subplots(1,3,figsize=(9.4,3.2))
    for ax,(col,label,log) in zip(axes,metrics):
        ax.axhline(base[col],color="black",ls=":",lw=1,label="FP16")
        for fam in ("p1","p2","p3"):
            d = df[(df.condition.str[:2] == fam) & (df.condition != "p2_24")].sort_values("sparsity")
            ax.plot(100*d["sparsity"],d[col],marker="o",color=COLOR[fam],label=FAMILY[fam])
        d = df[df.condition == "p2_24"]
        ax.scatter(100*d["sparsity"],d[col],marker="s",color=COLOR["p2_24"],zorder=3,label="Wanda 2:4")
        ax.set_xlabel("eligible sparsity (%)")
        ax.set_ylabel(label)
        if log:
            ax.set_yscale("log")
    axes[2].set_title("first 512 rows",fontsize=9)
    legend_below(fig,axes,5)
    fig.tight_layout(rect=(0,0.08,1,1))
    fig.savefig(out/"pruning_tradeoff.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(9.4,3.2))
    for ax,(col,label,log) in zip(axes,metrics):
        ax.axhline(base[col],color="black",ls=":",lw=1,label="FP16")
        for fam in ("q1","q2","q3"):
            d = df[df.condition.str[:2] == fam].sort_values("storage_gb")
            ax.plot(d["storage_gb"],d[col],marker="o",color=COLOR[fam],label=FAMILY[fam],
                    ls="-" if fam != "q3" else "none")
        ax.set_xlabel("packed model size (GB)")
        ax.set_ylabel(label)
    axes[2].set_title("first 512 rows",fontsize=9)
    legend_below(fig,axes,4)
    fig.tight_layout(rect=(0,0.08,1,1))
    fig.savefig(out/"quantization_tradeoff.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


def sensitivity_figures(res,out):
    """Measured sensitivities and the allocations they produced"""
    sens = res/"sensitivity"
    probes = pd.read_csv(sens/"p3_block_probes.csv")
    owl = pd.read_csv(sens/"owl_ratios.csv").sort_values("block")
    grid = probes.pivot(index="sparsity",columns="block",values="delta")
    fig,axes = plt.subplots(1,3,figsize=(9.4,3.0),gridspec_kw={"width_ratios":[1.3,1,1]})
    im = axes[0].imshow(np.log10(np.clip(grid.values,1e-4,None)),aspect="auto",origin="lower",cmap="viridis")
    axes[0].set_yticks(range(len(grid.index)),[f"{v:.2f}" for v in grid.index])
    axes[0].set_xticks(range(0,len(grid.columns),3),grid.columns[::3])
    axes[0].set_xlabel("block")
    axes[0].set_ylabel("sparsity of that block alone")
    axes[0].set_title("(a) log10 calibration nll rise",fontsize=9)
    fig.colorbar(im,ax=axes[0],fraction=0.05)
    axes[1].bar(owl["block"],100*owl["outlier_ratio"],color="steelblue")
    axes[1].set_xlabel("block")
    axes[1].set_ylabel("outlier ratio (%)")
    axes[1].set_title("(b) Wanda score outliers",fontsize=9)
    for target,color in ((50,"seagreen"),(75,"firebrick")):
        path = res/"conditions"/f"p3_s{target}"/"allocation.json"
        if path.exists():
            alloc = load_json(path)
            axes[2].plot(range(len(alloc["block_sparsity"])),100*np.array(alloc["block_sparsity"]),marker=".",
                         color=color,label=f"{target}% target, {alloc['design']}")
    axes[2].set_xlabel("block")
    axes[2].set_ylabel("block sparsity (%)")
    axes[2].set_title("(c) selected allocation",fontsize=9)
    axes[2].legend(loc="upper center",bbox_to_anchor=(0.5,-0.25),frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"p3_sensitivity.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    q = pd.read_csv(sens/"q3_projection_probes.csv")
    names = list(PROJ_COLOR)
    fig,axes = plt.subplots(2,2,figsize=(9.4,4.6))
    for ax,b in zip(axes[0],(3,2)):
        g = q[q.bits == b].pivot(index="proj",columns="block",values="delta").loc[names]
        im = ax.imshow(np.log10(np.clip(g.values,1e-5,None)),aspect="auto",cmap="viridis")
        ax.set_yticks(range(len(names)),names,fontsize=7)
        ax.set_xticks(range(0,len(g.columns),3),g.columns[::3])
        ax.set_title(f"({'a' if b == 3 else 'b'}) log10 nll rise at {b} bits, one projection alone",fontsize=9)
        fig.colorbar(im,ax=ax,fraction=0.04)
    cmap = matplotlib.colors.ListedColormap(["firebrick","darkorange","steelblue","seagreen"])
    for ax,budget,letter in zip(axes[1],(4,3),("c","d")):
        path = res/"conditions"/f"q3_int{budget}"/"allocation.json"
        if path.exists():
            alloc = load_json(path)
            blocks = sorted({int(k.split(".")[0]) for k in alloc["bits"]})
            m = np.array([[[2,3,4,8].index(alloc["bits"][f"{i}.{n}"]) for i in blocks] for n in names])
            ax.imshow(m,aspect="auto",cmap=cmap,vmin=-0.5,vmax=3.5)
            ax.set_yticks(range(len(names)),names,fontsize=7)
            ax.set_xticks(range(0,len(blocks),3),blocks[::3])
            ax.set_title(f"({letter}) bits under the INT{budget} budget, {alloc['design']}",fontsize=9)
        ax.set_xlabel("block")
    patches = [matplotlib.patches.Patch(color=c,label=f"{b} bits") for c,b in zip(cmap.colors,(2,3,4,8))]
    fig.legend(handles=patches,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.03))
    fig.tight_layout(rect=(0,0.05,1,1))
    fig.savefig(out/"q3_sensitivity.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


def agreement_figure(res,cfg,out):
    """Overlap between the weights magnitude and Wanda remove for every projection type and depth"""
    fig,axes = plt.subplots(1,3,figsize=(9.4,3.0),sharey=True)
    for ax,s in zip(axes,cfg["prune"]["sparsities"]):
        path = res/"conditions"/f"p2_s{round(100*s)}"/"mask_agreement.csv"
        if not path.exists():
            continue
        a = pd.read_csv(path)
        for n,color in PROJ_COLOR.items():
            d = a[a.proj == n].sort_values("block")
            ax.plot(d["block"],100*d["overlap"],color=color,lw=1.2,label=n)
        ax.set_title(f"{round(100*s)}% sparsity",fontsize=9)
        ax.set_xlabel("block")
    axes[0].set_ylabel("overlap with magnitude mask (%)")
    legend_below(fig,axes,7)
    fig.tight_layout(rect=(0,0.08,1,1))
    fig.savefig(out/"mask_agreement.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


def calibration_figure(df,out):
    """Calibration nll against held out quality across every condition"""
    fig,axes = plt.subplots(1,3,figsize=(9.4,3.0))
    stats = {}
    for ax,(col,label) in zip(axes,(("wikitext_ppl","WikiText-2 perplexity"),("c4_ppl","C4 perplexity"),
                                    ("acc_norm_first512","HellaSwag acc_norm (%)"))):
        for fam in ("fp16","p1","p2","p3","q1","q2","q3"):
            d = df[df.condition.map(family) == fam]
            ax.scatter(d["calib_nll"],d[col],color=COLOR[fam],marker="o" if fam[0] != "q" else "^",s=18,
                       label=FAMILY[fam])
        rho = spearmanr(df["calib_nll"],df[col]).correlation
        stats[col] = rho
        ax.set_xscale("log")
        plain_log(ax,"x",[3,5,10])
        if col != "acc_norm_first512":
            ax.set_yscale("log")
        ax.set_xlabel("calibration nll")
        ax.set_ylabel(label)
        ax.set_title(f"Spearman {rho:.2f}"+(", first 512 rows" if col == "acc_norm_first512" else ""),fontsize=9)
    legend_below(fig,axes,4)
    fig.tight_layout(rect=(0,0.14,1,1))
    fig.savefig(out/"calibration_vs_heldout.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    return stats


def qualitative(res,cfg,scores):
    """Fixed hellaswag cases chosen in manifest order by the rule in the config"""
    q = cfg["qualitative"]
    models = [m for m in q["hellaswag_models"] if m in scores]
    docs = {d["order"]:d for d in hellaswag(cfg)}
    tabs = {m:table(res,m,"hellaswag").set_index("order") for m in ["fp16"]+models}
    first = sorted(tabs["fp16"].index)[:cfg["eval"]["hellaswag_subset"]]
    flips,fixes = [],[]
    for o in first:
        base = tabs["fp16"].loc[o]
        picks = [tabs[m].loc[o,"pred_norm"] for m in models]
        if base["pred_norm"] == base["gold"] and any(p != base["pred_norm"] for p in picks):
            flips.append(o)
        if base["pred_norm"] != base["gold"] and any(p == base["gold"] for p in picks):
            fixes.append(o)
    chosen = flips[:q["hellaswag_flips"]]+fixes[:q["hellaswag_fixes"]]
    out = []
    for o in chosen:
        case = {"order":o,"query":docs[o]["query"],"endings":docs[o]["choices"],"gold":docs[o]["gold"],"models":{}}
        for m,t in tabs.items():
            r = t.loc[o]
            case["models"][m] = {"pick":int(r["pred_norm"]),"ll_per_char":[r[f"ll{j}"]/r[f"chars{j}"] for j in range(4)]}
        out.append(case)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--results")
    args = ap.parse_args()
    plt.switch_backend("Agg")
    cfg = load_yaml(args.config)
    res = repo_path(args.results or cfg["paths"]["results_dir"])
    n_boot = cfg["bootstrap"]["resamples"]
    scores = {p.name:load_json(p/"scores.json") for p in sorted((res/"conditions").iterdir()) if (p/"scores.json").exists()}
    tables = res/"tables"
    figs = res/"figures"
    tables.mkdir(parents=True,exist_ok=True)
    figs.mkdir(parents=True,exist_ok=True)
    df = summary(res,scores,cfg,n_boot)
    df.to_csv(tables/"summary.csv",index=False)
    boot = [r for a,b in PAIRS if a in scores and b in scores for r in compare(res,a,b,n_boot,cfg["seed"])]
    pd.DataFrame(boot).to_csv(tables/"bootstrap.csv",index=False)
    extra = [r for a,b in METHOD_PAIRS if a in scores and b in scores for r in compare(res,a,b,n_boot,cfg["seed"])]
    pd.DataFrame(extra).to_csv(tables/"method_pairs.csv",index=False)
    carryover(res,scores,cfg).to_csv(tables/"selection_carryover.csv",index=False)
    checks = []
    for a,b in PAIRS:
        if a in scores and b in scores:
            key = "eligible_zeros" if a[:2] == "p3" else "packed_bytes"
            checks.append({"adaptive":a,"uniform":b,"quantity":key,"adaptive_value":scores[a][key],
                           "uniform_value":scores[b][key],"within_budget":scores[a][key] <= scores[b][key],
                           "equal":scores[a][key] == scores[b][key]})
    pd.DataFrame(checks).to_csv(tables/"budget_checks.csv",index=False)
    timing = res/"resources"/"timing.csv"
    if timing.exists():
        t = pd.read_csv(timing)
        r = df[["condition","sparsity","nominal_bits","storage_gb"]].merge(t,on="condition")
        r["storage_vs_fp16"] = r["storage_gb"]/(scores["fp16"]["dense_fp16_bytes"]/1e9)
        r["compress_seconds"] = r["condition"].map(lambda n:scores[n].get("compress_seconds"))
        r["compress_peak_gb"] = r["condition"].map(lambda n:scores[n].get("compress_peak_gb"))
        r.to_csv(tables/"resources.csv",index=False)
        if "fp16_end" in set(t["condition"]):
            drift = t.set_index("condition").loc[["fp16","fp16_end"],["prefill_512_ms","prefill_2048_ms","decode_ms_per_token"]]
            drift.loc["end_over_start"] = drift.loc["fp16_end"]/drift.loc["fp16"]
            drift.to_csv(tables/"timing_drift.csv")
    tradeoff_figures(df,figs)
    if (res/"sensitivity"/"p3_block_probes.csv").exists() and (res/"sensitivity"/"q3_projection_probes.csv").exists():
        sensitivity_figures(res,figs)
    agreement_figure(res,cfg,figs)
    stats = calibration_figure(df,figs)
    save_json(tables/"calibration_correlation.json",stats)
    save_json(res/"qualitative"/"hellaswag_cases.json",qualitative(res,cfg,scores))
    print(df[["condition","sparsity","storage_gb","wikitext_ppl","c4_ppl","acc_norm"]].to_string(index=False))


if __name__ == "__main__":
    main()
