import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import load_json,load_yaml,repo_path

IMPL_COLOR = {"dense":"steelblue","tiled":"darkorange","sdpa":"seagreen","sdpa_efficient":"firebrick","sdpa_math":"mediumpurple"}
IMPL_LABEL = {"dense":"dense explicit","tiled":"tiled online softmax","sdpa":"SDPA default","sdpa_efficient":"SDPA efficient",
              "sdpa_math":"SDPA math"}
MECH_COLOR = {"dense":"steelblue","window":"darkorange","dilated":"seagreen","linear":"firebrick","local_linear":"mediumpurple"}
MECH_LABEL = {"dense":"dense","window":"window 64","dilated":"local and dilated","linear":"linear ELU+1",
              "local_linear":"window 16 and linear"}
METHOD_COLOR = {"fp16":"black","sink_recent":"steelblue","question_aware":"darkorange","int4_token":"seagreen",
                "int4_channel":"firebrick","hybrid":"mediumpurple"}
METHOD_LABEL = {"fp16":"FP16 full cache","sink_recent":"sink and recent","question_aware":"question aware",
                "int4_token":"INT4 token wise","int4_channel":"INT4 channel K","hybrid":"hybrid FP16 and INT2"}
# principal paired accuracy differences of the cache study as first minus second
KV_PAIRS = [("sink_recent","fp16"),("question_aware","fp16"),("int4_token","fp16"),("int4_channel","fp16"),("hybrid","fp16"),
            ("question_aware","sink_recent"),("int4_channel","int4_token"),("hybrid","question_aware"),
            ("hybrid","int4_channel"),("int4_channel","question_aware")]


def legend_below(fig,axes,ncol):
    """Put one shared legend under the panels"""
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=ncol,frameon=False,bbox_to_anchor=(0.5,-0.02))


def boot(diff,n_boot,seed):
    """Mean of paired differences with a percentile bootstrap interval"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0,len(diff),(n_boot,len(diff)))
    return diff.mean(),*np.percentile(diff[idx].mean(1),[2.5,97.5])


def log2_axis(ax,ticks):
    """Plain sequence length ticks on a base two axis"""
    ax.set_xscale("log",base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])


def exact_part(res,figs,tables):
    """Error latency and memory figures of the exact attention study"""
    ex = res/"exact"
    fid = pd.read_csv(ex/"fidelity.csv")
    tim = pd.read_csv(ex/"timing.csv")
    fid = fid[fid["status"] == "ok"]
    dtypes = sorted(fid["dtype"].unique(),key=["float16","float32"].index)
    ticks = sorted(tim["length"].unique())
    fig,axes = plt.subplots(2,len(dtypes),figsize=(4.6*len(dtypes),6.6),squeeze=False)
    for j,dt in enumerate(dtypes):
        for i,(col,title) in enumerate((("out_rel","output"),("dq_rel","query gradient"))):
            ax = axes[i,j]
            for impl,g in fid[fid["dtype"] == dt].groupby("impl"):
                g = g.dropna(subset=[col]).sort_values("length")
                if len(g):
                    # drawing the default dispatch dashed on top since it runs the efficient kernel and would hide behind it
                    dashed = impl == "sdpa"
                    ax.plot(g["length"],g[col],marker="x" if dashed else "o",color=IMPL_COLOR[impl],label=IMPL_LABEL[impl],
                            ls="--" if dashed else "-",zorder=3 if dashed else 2)
            ax.set_yscale("log")
            log2_axis(ax,ticks)
            ax.set_title(f"({'ab'[j] if i == 0 else 'cd'[j]}) {title} error, {dt}",fontsize=10)
            ax.set_ylabel("relative error to dense FP32")
            ax.set_xlabel("sequence length")
    legend_below(fig,[axes[0,0]],5)
    fig.tight_layout(rect=(0,0.06,1,1))
    fig.savefig(figs/"exact_errors.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    ok = tim[tim["status"] == "ok"].reindex(columns=list(tim.columns)+[c for c in ("forward_peak_gb","forward_backward_peak_gb")
                                                                       if c not in tim.columns])
    fig,axes = plt.subplots(2,4,figsize=(16,6.6))
    panels = [(dt,mode) for dt in dtypes for mode in ("forward","forward_backward")]
    for j,(dt,mode) in enumerate(panels):
        for i,(col,unit) in enumerate(((f"{mode}_ms","time (ms)"),(f"{mode}_peak_gb","peak allocated (GB)"))):
            ax = axes[i,j]
            for impl,g in ok[ok["dtype"] == dt].groupby("impl"):
                g = g.dropna(subset=[col]).sort_values("length")
                if len(g):
                    # drawing the default dispatch dashed on top since it runs the efficient kernel and would hide behind it
                    dashed = impl == "sdpa"
                    ax.plot(g["length"],g[col],marker="x" if dashed else "o",color=IMPL_COLOR[impl],label=IMPL_LABEL[impl],
                            ls="--" if dashed else "-",zorder=3 if dashed else 2)
            if ax.has_data():
                ax.set_yscale("log")
            log2_axis(ax,ticks)
            ax.set_title(f"({'abcdefgh'[i*4+j]}) {mode.replace('_',' and ')}, {dt}",fontsize=10)
            ax.set_ylabel(unit)
            ax.set_xlabel("sequence length")
    legend_below(fig,[axes[0,0]],5)
    fig.tight_layout(rect=(0,0.06,1,1))
    fig.savefig(figs/"exact_cost.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    status = tim.pivot_table(index=["impl","dtype"],columns="length",values="status",aggfunc="first")
    status.to_csv(tables/"exact_status.csv")


def recall_runs(res):
    """Every finished recall run with its summary curves logs and test tables"""
    root = res/"recall"/"runs"
    out = {"summary":[],"curve":[],"log":[],"sets":[],"bins":[]}
    for d in sorted(root.iterdir()):
        if not (d/"summary.json").exists():
            continue
        s = load_json(d/"summary.json")
        out["summary"].append(s)
        for name,file in (("curve","curve.csv"),("log","train_log.csv")):
            t = pd.read_csv(d/file)
            t["run"],t["seed"],t["mechanism"] = s["run"],s["seed"],s["mechanism"]
            out[name].append(t)
        out["sets"].append(pd.read_csv(d/"test_sets.csv"))
        out["bins"].append(pd.read_csv(d/"test_bins.csv"))
    return {k:(pd.DataFrame(v) if k == "summary" else pd.concat(v,ignore_index=True)) for k,v in out.items()}


def recall_part(res,cfg,figs,tables):
    """Tables learning curves accuracy figures heatmap and resource sweep of the recall study"""
    mq = cfg["mqar"]
    r = recall_runs(res)
    summ = r["summary"]
    sets = r["sets"]
    own = sets[sets["test_k"] == sets["train_k"]]
    acc = own.groupby(["run","mechanism","train_k","train_regime","mask","test_regime"])["accuracy"]
    table = acc.agg(["mean","std","count"]).reset_index()
    table.to_csv(tables/"recall_accuracy.csv",index=False)
    wide = table.pivot_table(index=["run","mechanism","train_k","train_regime","mask"],columns="test_regime",
                             values=["mean","std"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide.to_csv(tables/"recall_accuracy_wide.csv",index=False)
    costs = summ.groupby(["run","mechanism"]).agg(parameters=("parameters","first"),ms_per_update=("ms_per_update","mean"),
                                                 peak_train_gb=("peak_train_gb","mean"),
                                                 final_loss=("final_train_loss","mean")).reset_index()
    costs.to_csv(tables/"recall_costs.csv",index=False)
    # learning curves of every trained mechanism
    runs = [x for x in mq["runs"] if x["name"] in set(summ["run"])]
    mechs = [m for m in MECH_COLOR if m in {x["mechanism"] for x in runs}]
    fig,axes = plt.subplots(2,len(mechs),figsize=(4.2*len(mechs),6.8),squeeze=False)
    styles = ["-","--",":","-."]
    for j,m in enumerate(mechs):
        mine = [x for x in runs if x["mechanism"] == m]
        for s,x in enumerate(mine):
            lg = r["log"][r["log"]["run"] == x["name"]].groupby("update")["loss"].mean()
            axes[0,j].plot(lg.index,lg.values,color=MECH_COLOR[m],ls=styles[s % 4],label=f"K={x['k']} {x['regime']}")
            cv = r["curve"][(r["curve"]["run"] == x["name"]) & (r["curve"]["k"] == x["k"]) & (r["curve"]["regime"] == "mixed")]
            cv = cv.groupby("update")["accuracy"].agg(["mean","std"])
            axes[1,j].errorbar(cv.index,100*cv["mean"],yerr=100*cv["std"].fillna(0),color=MECH_COLOR[m],ls=styles[s % 4],
                               marker="o",ms=3,capsize=2,label=f"K={x['k']} {x['regime']}")
        axes[0,j].set_title(f"({'abcdefghij'[j]}) {MECH_LABEL[m]}, training loss",fontsize=10)
        axes[1,j].set_title(f"({'abcdefghij'[len(mechs)+j]}) {MECH_LABEL[m]}, development mixed",fontsize=10)
        axes[0,j].set_xlabel("update")
        axes[1,j].set_xlabel("update")
        axes[0,j].set_ylabel("cross entropy at queries")
        axes[1,j].set_ylabel("accuracy at trained K (%)")
        axes[1,j].set_ylim(-3,103)
        axes[1,j].legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=2,frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(figs/"recall_curves.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    # accuracy by number of pairs per regime for the models trained on mixed data
    fig,axes = plt.subplots(1,3,figsize=(13,4.2))
    for j,reg in enumerate(("near","far","mixed")):
        ax = axes[j]
        for (run,mask),g in table[(table["test_regime"] == reg)].groupby(["run","mask"]):
            x = next(y for y in runs if y["name"] == run)
            label = f"{MECH_LABEL[x['mechanism']]}, trained {x['regime']}"+(", window at test only" if mask == "test_window" else "")
            # shifting each mechanism a little along x so overlapping points stay visible
            shift = 0.3*mechs.index(x["mechanism"])-0.45
            ax.errorbar(np.log2(g["train_k"])+shift/4,100*g["mean"],yerr=100*g["std"].fillna(0),color=MECH_COLOR[x["mechanism"]],
                        marker="s" if mask == "test_window" else "o",ms=4,capsize=2,label=label,
                        ls="--" if mask == "test_window" else ("-" if x["regime"] == "mixed" else ":"))
        ax.axhline(100/mq["values"],color="gray",lw=0.8,ls=":")
        ax.set_xticks([2,3,4])
        ax.set_xticklabels(["4","8","16"])
        ax.set_title(f"({'abc'[j]}) {reg} test queries",fontsize=10)
        ax.set_xlabel("number of key value pairs K (trained and tested)")
        ax.set_ylabel("accuracy (%)")
        ax.set_ylim(-3,103)
    handles,labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels,handles))
    fig.legend(seen.values(),seen.keys(),loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.12),fontsize=8)
    fig.tight_layout()
    fig.savefig(figs/"recall_regimes.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    # distance by load heatmap pooled over the three test regimes
    bins = r["bins"].groupby(["run","mask","test_k","bin","seed"])[["correct","queries"]].sum().reset_index()
    bins["accuracy"] = bins["correct"]/bins["queries"]
    heat = bins.groupby(["run","mask","test_k","bin"])["accuracy"].mean().reset_index()
    heat.to_csv(tables/"recall_distance_load.csv",index=False)
    order = [(x["name"],"trained") for x in runs]+[(x["name"],"test_window") for x in runs if x["name"] in mq["test_window_runs"]]
    order = [o for o in order if ((heat["run"] == o[0]) & (heat["mask"] == o[1])).any()]
    edges = mq["bins"]
    labels = [f"[{a},{b})" for a,b in zip(edges[:-1],edges[1:])]
    cols = [(k,b) for k in (4,8,16) for b in labels]
    grid = np.full((len(order),len(cols)),np.nan)
    for i,(run,mask) in enumerate(order):
        for j,(k,b) in enumerate(cols):
            v = heat[(heat["run"] == run) & (heat["mask"] == mask) & (heat["test_k"] == k) & (heat["bin"] == b)]["accuracy"]
            if len(v):
                grid[i,j] = 100*v.iloc[0]
    fig,ax = plt.subplots(figsize=(12,0.38*len(order)+2.2))
    im = ax.imshow(grid,cmap="viridis",vmin=0,vmax=100,aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i,j]):
                ax.text(j,i,f"{grid[i,j]:.0f}",ha="center",va="center",fontsize=7,color="white" if grid[i,j] < 60 else "black")
    names = []
    for run,mask in order:
        x = next(y for y in runs if y["name"] == run)
        names.append(f"{MECH_LABEL[x['mechanism']]}, trained K={x['k']} {x['regime']}"+(", window at test only" if mask == "test_window" else ""))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names,fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([b for _,b in cols],fontsize=8,rotation=45)
    for k_i,k in enumerate((4,8,16)):
        ax.text(4*k_i+1.5,-0.9,f"test K={k}",ha="center",fontsize=9)
        if k_i:
            ax.axvline(4*k_i-0.5,color="white",lw=2)
    ax.set_xlabel("source to query distance bin (positions)")
    fig.colorbar(im,ax=ax,label="accuracy (%), mean of seeds",fraction=0.025)
    fig.tight_layout()
    fig.savefig(figs/"recall_distance_load.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    # forward only resource sweep
    sw = pd.read_csv(res/"recall"/"sweep.csv")
    sw = sw[sw["status"] == "ok"]
    fig,axes = plt.subplots(1,2,figsize=(9.5,3.8))
    for m,g in sw.groupby("mechanism"):
        g = g.sort_values("length")
        axes[0].plot(g["length"],g["forward_ms"],marker="o",color=MECH_COLOR[m],label=MECH_LABEL[m])
        if "peak_gb" in g and g["peak_gb"].notna().any():
            axes[1].plot(g["length"],g["peak_gb"],marker="o",color=MECH_COLOR[m],label=MECH_LABEL[m])
    for ax,(title,unit) in zip(axes,(("(a) forward time","time (ms)"),("(b) forward peak memory","peak allocated (GB)"))):
        if ax.has_data():
            ax.set_yscale("log")
        log2_axis(ax,sorted(sw["length"].unique()))
        ax.set_title(title,fontsize=10)
        ax.set_ylabel(unit)
        ax.set_xlabel("sequence length")
    legend_below(fig,[axes[0]],4)
    fig.tight_layout(rect=(0,0.08,1,1))
    fig.savefig(figs/"recall_sweep.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


def kv_part(res,cfg,figs,tables):
    """Matched budget table paired intervals breakdowns budget checks and figures of the cache study"""
    kv = cfg["kv"]
    d = pd.read_csv(res/"kv"/"per_prompt.csv",dtype={"code":str,"answer":str})
    methods = [m for m in kv["methods"] if m in set(d["method"])]
    ref = d[d["method"] == "fp16"].set_index("id")["correct"]
    d["ref_correct"] = d["id"].map(ref)
    rows = []
    for m in methods:
        g = d[d["method"] == m]
        rows.append({"method":m,"prompts":len(g),"accuracy":100*g["correct"].mean(),
                     "accuracy_when_fp16_correct":100*g[g["ref_correct"]]["correct"].mean(),
                     "answer_nll":g["nll"].mean(),"kl_post_prefill":g["kl_post"].mean(),"kl_answer":g["kl_answer"].mean(),
                     "evidence_retained":g["evidence_retained"].mean(),"code_retained":g["code_retained"].mean(),
                     "fraction_of_fp16_bytes":g["fraction_of_fp16"].mean(),"stored_mb":g["stored_bytes"].mean()/1e6,
                     "k_rel_error":g["k_rel_error"].mean(),"v_rel_error":g["v_rel_error"].mean(),
                     "encode_ms":g["encode_ms"].mean(),"score_ms":g["score_ms"].mean(),"suffix_ms":g["suffix_ms"].mean(),
                     "decode_ms_per_token":g["decode_ms_per_token"].mean(),"peak_gb":g["peak_gb"].mean(),
                     "truncated":int(g["truncated"].sum()),"all_finite":bool(g["finite"].all())})
    pd.DataFrame(rows).to_csv(tables/"kv_summary.csv",index=False)
    out = []
    for a,b in KV_PAIRS:
        if a not in methods or b not in methods:
            continue
        m = d[d["method"] == a].merge(d[d["method"] == b],on="id",suffixes=("_a","_b"))
        for col,scale,name in (("correct",100,"accuracy_points"),("nll",1,"answer_nll")):
            diff = scale*(m[f"{col}_a"].astype(float)-m[f"{col}_b"].astype(float)).values
            est,lo,hi = boot(diff,kv["bootstrap"],cfg["seed"])
            out.append({"a":a,"b":b,"metric":name,"diff":est,"lo":lo,"hi":hi,"n":len(diff)})
    pd.DataFrame(out).to_csv(tables/"kv_bootstrap.csv",index=False)
    parts = []
    for factor in ("target_tokens","position","distractors"):
        g = d.groupby([factor,"method"]).agg(accuracy=("correct","mean"),nll=("nll","mean"),kl_post=("kl_post","mean"),
                                             evidence_retained=("evidence_retained","mean"),prompts=("id","count")).reset_index()
        g = g.rename(columns={factor:"level"})
        g.insert(0,"factor",factor)
        parts.append(g)
    pd.concat(parts).to_csv(tables/"kv_breakdown.csv",index=False)
    cells = d.groupby(["target_tokens","position","distractors","method"])["correct"].mean().unstack("method")
    cells.to_csv(tables/"kv_cells.csv")
    comp = d[d["method"] != "fp16"].copy()
    comp["over"] = comp["stored_bytes"]-comp["budget_bytes"]
    comp["formula_match"] = comp["stored_bytes"] == comp["formula_bytes"]
    checks = comp.groupby("method").agg(prompts=("id","count"),max_bytes_over_budget=("over","max"),
                                        min_fraction=("fraction_of_fp16","min"),max_fraction=("fraction_of_fp16","max"),
                                        packed_equals_formula=("formula_match","all")).reset_index()
    checks["any_over_budget"] = checks["max_bytes_over_budget"] > 0
    checks["packed_equals_formula"] = checks["packed_equals_formula"].astype(object)
    checks.loc[~checks["method"].str.startswith("int4"),"packed_equals_formula"] = None
    checks.to_csv(tables/"kv_budget_checks.csv",index=False)
    prompts = d[d["method"] == "fp16"][["id","target_tokens","position","distractors","prompt_tokens","context_tokens",
                                        "suffix_tokens","evidence_tokens","budget_bytes","fp16_bytes","boundary_exact"]]
    prompts.to_csv(tables/"kv_prompts.csv",index=False)
    # accuracy by factor
    fig,axes = plt.subplots(1,3,figsize=(14,4.0))
    for j,factor in enumerate(("target_tokens","position","distractors")):
        g = d.groupby([factor,"method"])["correct"].mean().unstack("method")[methods]
        levels = list(g.index) if factor != "position" else [x for x in ("start","middle","end") if x in g.index]
        g = g.loc[levels]
        w = 0.8/len(methods)
        for i,m in enumerate(methods):
            xs = np.arange(len(levels))+i*w-0.4+w/2
            axes[j].bar(xs,100*g[m].values,w,color=METHOD_COLOR[m],label=METHOD_LABEL[m])
            # marking bars of zero height so a method at zero accuracy stays visible
            for xv,yv in zip(xs,g[m].values):
                if yv == 0:
                    axes[j].text(xv,1,"0",ha="center",va="bottom",fontsize=7,color=METHOD_COLOR[m])
        axes[j].set_xticks(range(len(levels)))
        axes[j].set_xticklabels([str(x) for x in levels])
        axes[j].set_ylim(0,105)
        axes[j].set_ylabel("exact match (%)")
        axes[j].set_xlabel({"target_tokens":"nominal context tokens","position":"evidence position",
                            "distractors":"incorrect distractor records"}[factor])
        axes[j].set_title(f"({'abc'[j]}) by {factor.replace('_',' ')}, n per bar {len(d)//len(methods)//len(levels)}",fontsize=10)
    legend_below(fig,[axes[0]],6)
    fig.tight_layout(rect=(0,0.08,1,1))
    fig.savefig(figs/"kv_breakdown.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    # likelihood and divergence per prompt
    fig,axes = plt.subplots(1,2,figsize=(11,3.8))
    for ax,(col,title) in zip(axes,(("nll","(a) gold code NLL per token"),("kl_post","(b) post prefill KL from FP16"))):
        data = [d[d["method"] == m][col].values for m in methods]
        bp = ax.boxplot(data,patch_artist=True,showfliers=True,flierprops={"markersize":2})
        for patch,m in zip(bp["boxes"],methods):
            patch.set_facecolor(METHOD_COLOR[m])
            patch.set_alpha(0.6)
        ax.set_xticks(range(1,len(methods)+1))
        ax.set_xticklabels([METHOD_LABEL[m].replace(" ","\n",1) for m in methods],fontsize=8)
        ax.set_title(title,fontsize=10)
        if col == "kl_post":
            ax.set_yscale("symlog",linthresh=1e-3)
            ax.set_ylim(bottom=0)
    axes[0].set_ylabel("nats")
    axes[1].set_ylabel("nats")
    fig.tight_layout()
    fig.savefig(figs/"kv_fidelity.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    qual = load_json(res/"kv"/"qualitative.json")
    if qual:
        fig,axes = plt.subplots(2,len(qual),figsize=(5*len(qual),6.4),squeeze=False)
        for j,q in enumerate(qual):
            ax = axes[0,j]
            for m,share in q["kept_share"].items():
                ax.plot(100*np.array(share),color=METHOD_COLOR[m],lw=0.8,label=METHOD_LABEL[m]+(" FP16 set" if m == "hybrid" else ""))
            ev = q["evidence_tokens"]
            ax.axvspan(ev[0],ev[-1],color="gold",alpha=0.5,label="gold evidence")
            ax.set_title(f"({'abc'[j]}) {q['id']}",fontsize=9)
            ax.set_xlabel("context position")
            ax.set_ylabel("layer and head units keeping it (%)")
            ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.18),ncol=2,frameon=False,fontsize=7)
            ax = axes[1,j]
            mask = np.array([[int(c) for c in row] for row in q["masks"]["question_aware"]])
            lo,hi = q["window"]
            ax.imshow(mask,aspect="auto",cmap="Greys",extent=(lo,hi,mask.shape[0],0),vmin=0,vmax=1)
            ax.axvspan(ev[0],ev[-1]+1,color="gold",alpha=0.35)
            ax.set_title(f"({'def'[j]}) question aware selection near the evidence",fontsize=9)
            ax.set_xlabel("context position")
            ax.set_ylabel("layer and key value head")
        fig.tight_layout()
        fig.savefig(figs/"kv_selection.png",dpi=200,bbox_inches="tight")
        plt.close(fig)
    if (res/"kv"/"key_stats.csv").exists():
        ks = pd.read_csv(res/"kv"/"key_stats.csv")
        layer = ks.groupby("layer")[["k_channel_max_over_median","v_channel_max_over_median","k_abs_max","v_abs_max",
                                     "k_error_token","k_error_channel","v_error_token"]].mean()
        layer.to_csv(tables/"kv_key_stats_by_layer.csv")
        fig,axes = plt.subplots(1,2,figsize=(11,3.8))
        axes[0].plot(layer.index,layer["k_channel_max_over_median"],marker="o",color="firebrick",label="keys")
        axes[0].plot(layer.index,layer["v_channel_max_over_median"],marker="o",color="seagreen",label="values")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("largest over median channel magnitude")
        axes[0].set_title("(a) channel outliers per layer",fontsize=10)
        for col,color,label in (("k_error_token","seagreen","keys, token wise groups"),
                                ("k_error_channel","firebrick","keys, channel wise groups"),
                                ("v_error_token","gray","values, token wise groups")):
            axes[1].plot(layer.index,layer[col],marker="o",color=color,label=label)
        axes[1].set_yscale("log")
        axes[1].set_ylabel("relative INT4 reconstruction error")
        axes[1].set_title("(b) INT4 error per layer",fontsize=10)
        for ax in axes:
            ax.set_xlabel("layer")
            ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.18),ncol=3,frameon=False,fontsize=8)
        fig.tight_layout()
        fig.savefig(figs/"kv_key_stats.png",dpi=200,bbox_inches="tight")
        plt.close(fig)
    bench = pd.read_csv(res/"kv"/"bench.csv")
    bench.to_csv(tables/"kv_bench.csv",index=False)
    shapes = bench[["context","batch"]].drop_duplicates().values.tolist()
    fig,axes = plt.subplots(1,3,figsize=(14,3.8))
    layouts = ["int4_token","int4_channel","int2_channel"]
    colors = {"fp16":"black","int4_token":"seagreen","int4_channel":"firebrick","int2_channel":"mediumpurple"}
    x = np.arange(len(shapes))
    for j,(col,title) in enumerate((("encode_ms","(a) quantize and pack"),("unpack_ms","(b) unpack and dequantize"),
                                     ("decode_step_ms","(c) one decode step"))):
        names = layouts if col != "decode_step_ms" else ["fp16"]+layouts
        w = 0.8/len(names)
        for i,n in enumerate(names):
            vals = [bench[(bench["layout"] == n) & (bench["context"] == c) & (bench["batch"] == b)][col].mean() for c,b in shapes]
            axes[j].bar(x+i*w-0.4+w/2,vals,w,color=colors[n],label=n.replace("_"," ")+(" with unpacking" if n != "fp16" and col == "decode_step_ms" else ""))
        axes[j].set_xticks(x)
        axes[j].set_xticklabels([f"L={c}\nbatch {b}" for c,b in shapes],fontsize=8)
        axes[j].set_ylabel("time (ms)")
        axes[j].set_title(title,fontsize=10)
        axes[j].legend(loc="upper center",bbox_to_anchor=(0.5,-0.22),ncol=2,frameon=False,fontsize=7)
    fig.tight_layout()
    fig.savefig(figs/"kv_bench.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--results",default=None)
    ap.add_argument("--parts",nargs="*",default=["exact","recall","kv"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(args.results or cfg["paths"]["results_dir"])
    figs,tables = res/"figures",res/"tables"
    figs.mkdir(parents=True,exist_ok=True)
    tables.mkdir(parents=True,exist_ok=True)
    if "exact" in args.parts and (res/"exact"/"timing.csv").exists():
        exact_part(res,figs,tables)
    if "recall" in args.parts and (res/"recall"/"runs").exists():
        recall_part(res,cfg,figs,tables)
    if "kv" in args.parts and (res/"kv"/"per_prompt.csv").exists():
        kv_part(res,cfg,figs,tables)
    print("task2 tables and figures written")


if __name__ == "__main__":
    main()
