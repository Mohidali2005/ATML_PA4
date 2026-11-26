import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import load_json,load_yaml,repo_path,save_json
from task3_speculative.alignment import SHARED
from task3_speculative.data import SPLITS,read_gz
from task3_speculative.models import DRAFTS,STUDENTS

COLOR = {"original":"dimgray","off":"steelblue","on":"darkorange"}
LABEL = {"original":"original draft","off":"off policy","on":"on policy"}
BANK_LABEL = {"teacher":"target","original_draft":"original draft","exact_spec_original_gamma4":"exact speculative",
              "off_6304":"off policy 6304","off_6305":"off policy 6305","on_6304":"on policy 6304","on_6305":"on policy 6305"}
BINS = [(1,16),(17,32),(33,64),(65,192)]
METRICS = ["forward_kl","reverse_kl","tv","greedy_agree","expected_accept"]
PARTS = [("draft_ms","draft proposal","steelblue"),("verify_ms","target verification","darkorange"),
         ("accept_ms","acceptance and residual","seagreen"),("rollback_ms","cache rollback","firebrick"),
         ("host_ms","host overhead","silver")]


def method_of(name):
    """Method part of a draft name"""
    return name.split("_")[0]


def boot_indices(n,n_boot,seed):
    """Resampled prompt positions shared by every statistic of one comparison"""
    return np.random.default_rng(seed).integers(0,n,(n_boot,n))


def ratio_ci(num,den,n_boot,seed):
    """Ratio of sums over prompts with a percentile interval from resampling prompts"""
    num,den = np.asarray(num,float),np.asarray(den,float)
    idx = boot_indices(len(num),n_boot,seed)
    r = num[idx].sum(1)/den[idx].sum(1)
    return num.sum()/den.sum(),*np.nanpercentile(r,[2.5,97.5])


def ratio_diff_ci(a,b,n_boot,seed):
    """Difference of two ratios of sums over the same resampled prompts"""
    (na,da),(nb,db) = [[np.asarray(x,float) for x in p] for p in (a,b)]
    idx = boot_indices(len(na),n_boot,seed)
    d = na[idx].sum(1)/da[idx].sum(1)-nb[idx].sum(1)/db[idx].sum(1)
    return na.sum()/da.sum()-nb.sum()/db.sum(),*np.nanpercentile(d,[2.5,97.5])


def legend_below(fig,ax,ncol,y=-0.02):
    """Put one shared legend under the panels"""
    h,l = ax.get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=ncol,frameon=False,bbox_to_anchor=(0.5,y))


def save(fig,path):
    """Write a figure and close it"""
    fig.savefig(path,dpi=200,bbox_inches="tight")
    plt.close(fig)


def checks_part(res,figs,tables):
    """Goodness of fit table joint probability figure and worst case of the cache checks"""
    tests = load_json(res/"checks"/"toy_tests.json")
    pd.DataFrame(tests).drop(columns=["eos_routes"]).to_csv(tables/"sampler_tests.csv",index=False)
    j = pd.read_csv(res/"checks"/"toy_joint_length3.csv",dtype={"sequence":str})
    n = j["speculative_gamma2"].sum()
    fig,ax = plt.subplots(figsize=(9.4,3.4))
    x = np.arange(len(j))
    ax.bar(x,j["target_probability"],0.8,color="lightsteelblue",label="exact target probability")
    ax.plot(x,j["speculative_gamma2"]/n,"o",ms=4,color="seagreen",label="exact sampler, γ = 2")
    ax.plot(x,j["wrong_residual_gamma2"]/n,"x",ms=5,color="firebrick",label="residual replaced by p, γ = 2")
    ax.set_xticks(x)
    ax.set_xticklabels(j["sequence"],rotation=90,fontsize=8)
    ax.set_xlabel("sequence of length three")
    ax.set_ylabel("joint probability")
    legend_below(fig,ax,3,-0.12)
    save(fig,figs/"toy_joint.png")
    real = load_json(res/"checks"/"real_checks.json")
    rows = real["prompts"]
    out = {"prompts":len(rows),"padding":real["padding"],"batched_sampling_ok":real["batched_sampling_ok"]}
    for m in ("target","draft"):
        for k in ("split_block","split_single"):
            out[f"{m}_{k}_max_tv"] = max(r[f"{m}_{k}"]["max_total_variation"] for r in rows)
            out[f"{m}_{k}_same_argmax"] = min(r[f"{m}_{k}"]["same_argmax"] for r in rows)
        out[f"{m}_rollback_max_abs_prob"] = max(r[f"{m}_rollback_max_abs_prob"] for r in rows)
        out[f"{m}_no_rollback_max_abs_prob"] = max(r[f"{m}_no_rollback_max_abs_prob"] for r in rows)
    for g in (1,2,4):
        out[f"gamma{g}_max_tv_after_run"] = max(max(r[f"gamma{g}_{m}_after_run"]["max_total_variation"] for m in ("target","draft"))
                                                for r in rows)
        out[f"gamma{g}_cache_lengths_ok"] = all(r[f"gamma{g}_cache_lengths_ok"] for r in rows)
        out[f"gamma{g}_token_count_ok"] = all(r[f"gamma{g}_token_count_ok"] for r in rows)
    out["cap_tokens_ok"] = all(r[f"cap{c}_tokens"] == c for r in rows for c in (1,2,3))
    out["self_draft_acceptance"] = float(np.mean([r["self_draft_accepted_over_examined"] for r in rows]))
    out["forced_eos_stopped"] = all(r["forced_eos"]["stopped"] for r in rows)
    save_json(tables/"real_checks_summary.json",out)


def training_part(cfg,res,figs,tables):
    """Training curves budgets and end to end distillation cost"""
    pool = [load_json(res/"distill"/f"pool_shard{s}.json") for s in (0,1)]
    pool_s = {c:sum(p[c]["seconds"] for p in pool) for c in ("math","short")}
    pool_t = {c:sum(p[c]["tokens"] for p in pool) for c in ("math","short")}
    merge = {m["name"]:m for m in load_json(res/"distill"/"merge.json")}
    frozen = {r["id"]:r for s in (0,1) for r in read_gz(res/"rollouts"/f"pool_shard{s}.jsonl.gz")}
    rows,logs = [],{}
    for name in STUDENTS:
        s = load_json(res/"distill"/name/"summary.json")
        logs[name] = pd.DataFrame(load_json(res/"distill"/name/"log.json"))
        pres = read_gz(res/"distill"/name/"presentations.jsonl.gz")
        used = {r["id"] for r in pres}
        row = {"name":name,"method":s["method"],"seed":s["seed"],"updates":s["updates"],"skipped_updates":s["skipped_updates"],
               "scored_math":s["scored_tokens"]["math"],"scored_short":s["scored_tokens"]["short"],
               "adapter_parameters":s["adapter_parameters"],"backbone_parameters":s["backbone_parameters"],
               "kl_first10":logs[name]["kl"].head(10).mean(),"kl_last10":logs[name]["kl"].tail(10).mean(),
               "kl_math_last20":logs[name]["kl_math"].tail(20).mean(),"kl_short_last20":logs[name]["kl_short"].tail(20).mean(),
               "lora_init_abs_sum":s["lora_init"]["a_abs_sum"],"peak_gb":s["peak_gb"],"wall_seconds":s["wall_seconds"]}
        for c in ("math","short"):
            for k,v in s["responses"][c].items():
                row[f"{c}_{k}"] = v
        for k,v in s["seconds"].items():
            row[f"{k}_seconds"] = v
        # the full collection is a one time cost shared by both seeds and the used share counts each response once
        off = s["method"] == "off"
        row["teacher_pool_full_seconds"] = sum(pool_s.values()) if off else 0.0
        row["teacher_pool_used_seconds"] = sum(frozen[i]["ms"] for i in used)/1000 if off else 0.0
        row["teacher_pool_tokens"] = sum(pool_t.values()) if off else 0
        row["merge_seconds"] = merge[name]["merge_seconds"]
        row["total_seconds_full_pool"] = row["wall_seconds"]+row["teacher_pool_full_seconds"]+row["merge_seconds"]
        row["total_seconds_used_pool"] = row["wall_seconds"]+row["teacher_pool_used_seconds"]+row["merge_seconds"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables/"training.csv",index=False)
    pd.DataFrame(merge.values()).to_csv(tables/"merge.csv",index=False)
    fig,axes = plt.subplots(1,2,figsize=(9.4,3.2))
    for ax,(col,title) in zip(axes,(("kl_math","(a) mathematics"),("kl_short","(b) short form"))):
        for name,log in logs.items():
            m,seed = name.split("_")
            x = log["scored_math"]+log["scored_short"]
            ax.plot(x/1000,log[col].rolling(8,min_periods=1).mean(),color=COLOR[m],ls="-" if seed == "6304" else "--",lw=1.4,
                    label=f"{LABEL[m]}, seed {seed}")
        ax.set_xlabel("teacher scored tokens (thousands)")
        ax.set_ylabel("forward KL (nats per token)")
        ax.set_title(title,fontsize=10)
    fig.tight_layout(rect=(0,0.08,1,1))
    legend_below(fig,axes[0],4,-0.02)
    save(fig,figs/"training_curves.png")


def add_bins(df):
    """Attach the response position bin"""
    df = df.copy()
    df["bin"] = pd.cut(df["position"],[0,16,32,64,192],labels=[f"{a}-{b}" for a,b in BINS]).astype(str)
    return df


def alignment_part(cfg,res,figs,tables):
    """Divergence tables and figures on the shared banks and on the students own rollouts"""
    n_boot,seed = cfg["analysis"]["bootstrap"],cfg["seed"]
    df = add_bins(pd.read_csv(res/"alignment"/"prefix_metrics.csv.gz"))
    scopes = [("1-64",df[df["position"] <= 64]),("65-192",df[df["position"] > 64])]
    rows = []
    for scope,part in scopes:
        for split in ("all",)+SPLITS:
            sub = part if split == "all" else part[part["split"] == split]
            for (bank,draft),g in sub.groupby(["bank","draft"]):
                per = g.groupby("prompt_id")
                row = {"positions":scope,"split":split,"bank":bank,"draft":draft,"prefixes":len(g),"prompts":per.ngroups}
                for m in METRICS:
                    s = per[m].agg(["sum","count"])
                    row[m],row[f"{m}_lo"],row[f"{m}_hi"] = ratio_ci(s["sum"],s["count"],n_boot,seed)
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(tables/"alignment.csv",index=False)
    bins = df.groupby(["bank","split","bin","draft"])[METRICS].mean().reset_index()
    bins["prefixes"] = df.groupby(["bank","split","bin","draft"]).size().values
    bins.to_csv(tables/"alignment_bins.csv",index=False)
    # paired differences on identical prefixes first minus second
    pairs = [(f"{m}_{s}","original") for m in ("off","on") for s in cfg["distill"]["seeds"]]
    pairs += [(f"on_{s}",f"off_{s}") for s in cfg["distill"]["seeds"]]
    pairs += [(f"{m}_6305",f"{m}_6304") for m in ("off","on")]
    out = []
    for split in ("all",)+SPLITS:
        sub = df[df["position"] <= 64]
        sub = sub if split == "all" else sub[sub["split"] == split]
        for bank,g in sub.groupby("bank"):
            for m in METRICS:
                wide = g.pivot_table(index=["prompt_id","position"],columns="draft",values=m)
                for a,b in pairs:
                    s = (wide[a]-wide[b]).groupby("prompt_id").agg(["sum","count"])
                    d,lo,hi = ratio_ci(s["sum"],s["count"],n_boot,seed)
                    out.append({"positions":"1-64","split":split,"bank":bank,"metric":m,"first":a,"second":b,"diff":d,"lo":lo,"hi":hi,
                                "prompts":len(s),"prefixes":int(s["count"].sum())})
    pd.DataFrame(out).to_csv(tables/"alignment_pairs.csv",index=False)
    # alignment figure with one bar group per bank and one panel per split and metric
    banks = list(SHARED)+list(STUDENTS)
    fig,axes = plt.subplots(2,3,figsize=(11,5.6),sharex=True)
    t64 = table[table["positions"] == "1-64"]
    for j,split in enumerate(SPLITS):
        for i,(m,name) in enumerate((("forward_kl","forward KL (nats)"),("expected_accept","expected acceptance"))):
            ax = axes[i,j]
            for k,meth in enumerate(("original","off","on")):
                names = [d for d in DRAFTS if method_of(d) == meth]
                vals = np.array([[t64[(t64["split"] == split)&(t64["bank"] == b)&(t64["draft"] == d)][m].iloc[0] for b in banks]
                                 for d in names])
                x = np.arange(len(banks))+0.27*(k-1)
                # drawing acceptance as points so a zoomed axis does not cut any bar
                if i == 0:
                    ax.bar(x,vals.mean(0),0.25,color=COLOR[meth],label=LABEL[meth])
                else:
                    ax.plot(x,vals.mean(0),"o",color=COLOR[meth],ms=6,label=LABEL[meth])
                if len(names) > 1:
                    for v in vals:
                        ax.plot(x,v,".",color="black",ms=3)
            ax.set_title(f"({'abcdef'[3*i+j]}) {split}",fontsize=10)
            ax.set_xticks(range(len(banks)))
            ax.set_xticklabels([BANK_LABEL[b] for b in banks],rotation=40,ha="right",fontsize=8)
            if i == 1:
                ax.grid(axis="y",color="lightgray",lw=0.5)
                ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(name)
    fig.tight_layout(rect=(0,0.04,1,1))
    legend_below(fig,axes[0,0],3,-0.02)
    save(fig,figs/"alignment_banks.png")
    fig,axes = plt.subplots(1,3,figsize=(9.4,3.0),sharey=True)
    labels = [f"{a}-{b}" for a,b in BINS]
    for ax,bank in zip(axes,SHARED):
        for d in DRAFTS:
            g = bins[(bins["bank"] == bank)&(bins["split"] == "math")&(bins["draft"] == d)].set_index("bin").reindex(labels)
            ax.plot(labels,g["expected_accept"],marker="o",ms=4,color=COLOR[method_of(d)],ls="--" if d.endswith("6305") else "-",
                    label=d.replace("_"," ").replace("off","off policy").replace("on ","on policy "))
        ax.set_title(f"{BANK_LABEL[bank]} prefixes, mathematics",fontsize=9)
        ax.set_xlabel("response position")
    axes[0].set_ylabel("expected acceptance 1 - TV")
    fig.tight_layout(rect=(0,0.1,1,1))
    legend_below(fig,axes[0],5,-0.02)
    save(fig,figs/"alignment_positions.png")
    return table


def consistency_part(cfg,res,tables):
    """Target against speculative consistency of rollouts prefix metrics and emitted tokens"""
    n_boot,seed = cfg["analysis"]["bootstrap"],cfg["seed"]
    seq = pd.read_csv(res/"alignment"/"rollout_surprisal.csv")
    rows = []

    def z_row(label,g):
        """Summed surprisal excess against its conditional standard deviation"""
        e,v,n = g["excess"].sum(),g["variance"].sum(),g["tokens"].sum()
        return {"sample":label,"sequences":len(g),"tokens":int(n),"mean_tokens":g["tokens"].mean(),
                "mean_surprisal":g["surprisal"].sum()/n,"mean_entropy":g["entropy"].sum()/n,"excess_per_token":e/n,
                "z":e/np.sqrt(v),"z_prompt_clustered":e/np.sqrt((g.groupby("prompt_id")["excess"].sum()**2).sum())}
    for bank,g in seq.groupby("source",sort=False):
        rows.append({**z_row(f"bank {bank}",g),"eos_rate":(g["termination"] == "eos").mean()})
    bench = pd.read_csv(res/"alignment"/"benchmark_surprisal.csv")
    one = bench[(bench["who"] == "target")&(bench["draft"] == "original")&(bench["gamma"] == 2)]
    rows.append(z_row("benchmark target only",one))
    for (d,g),part in bench[bench["who"] == "spec"].groupby(["draft","gamma"],sort=False):
        rows.append(z_row(f"benchmark speculative {d} gamma {g}",part))
    rows.append(z_row("benchmark speculative pooled",bench[bench["who"] == "spec"]))
    pd.DataFrame(rows).to_csv(tables/"consistency.csv",index=False)
    df = pd.read_csv(res/"alignment"/"prefix_metrics.csv.gz")
    df = df[df["position"] <= 64]
    out = []
    for split in ("all",)+SPLITS:
        sub = df if split == "all" else df[df["split"] == split]
        for d in DRAFTS:
            for m in METRICS:
                a = sub[(sub["bank"] == "teacher")&(sub["draft"] == d)].groupby("prompt_id")[m].agg(["sum","count"])
                b = sub[(sub["bank"] == SHARED[2])&(sub["draft"] == d)].groupby("prompt_id")[m].agg(["sum","count"])
                idx = a.index.union(b.index)
                a,b = a.reindex(idx,fill_value=0),b.reindex(idx,fill_value=0)
                diff,lo,hi = ratio_diff_ci((b["sum"],b["count"]),(a["sum"],a["count"]),n_boot,seed)
                out.append({"split":split,"draft":d,"metric":m,"teacher":a["sum"].sum()/a["count"].sum(),
                            "speculative":b["sum"].sum()/b["count"].sum(),"speculative_minus_teacher":diff,"lo":lo,"hi":hi})
    lens = seq[seq["source"].isin([SHARED[0],SHARED[2]])].pivot(index="prompt_id",columns="source",values="tokens")
    d = (lens[SHARED[2]]-lens[SHARED[0]]).to_numpy(float)
    b = d[boot_indices(len(d),n_boot,seed)].mean(1)
    out.append({"split":"all","draft":"none","metric":"response_length","teacher":lens[SHARED[0]].mean(),
                "speculative":lens[SHARED[2]].mean(),"speculative_minus_teacher":d.mean(),
                "lo":np.percentile(b,2.5),"hi":np.percentile(b,97.5)})
    pd.DataFrame(out).to_csv(tables/"teacher_vs_speculative.csv",index=False)


def break_even(gamma,c,v):
    """Smallest acceptance at which the projected speedup exceeds one"""
    grid = np.linspace(0,1,100001)
    ok = 1+sum(grid**j for j in range(1,gamma+1)) > gamma*c+v
    return float(grid[ok][0]) if ok.any() else None


def cost_table(iso):
    """Cost ratios of the draft step and of the verification pass for every context length"""
    step = iso[iso["pass"] == "step"].set_index(["model","context","tokens"])["ms"]
    rows = []
    for (model,context) in sorted({(m,c) for m,c,_ in step.index}):
        if model == "target":
            continue
        for g in (2,4):
            rows.append({"draft":model,"context":context,"gamma":g,"draft_step_ms":step[model,context,1],
                         "target_step_ms":step["target",context,1],"verify_ms":step["target",context,g+1],
                         "c":step[model,context,1]/step["target",context,1],
                         "v":step["target",context,g+1]/step["target",context,1]})
    return pd.DataFrame(rows)


def spec_summary(g,n_boot,seed):
    """Acceptance work and throughput of one draft and proposal length"""
    s = g.groupby("prompt_id").sum(numeric_only=True)
    out = {"runs":len(g),"prompts":len(s),"output_tokens":int(g["spec_tokens"].sum()),"mean_output_tokens":g["spec_tokens"].mean(),
           "eos_rate":g["spec_eos"].mean(),"truncated_rate":g["spec_truncated"].mean(),
           "target_only_mean_tokens":g["target_tokens"].mean(),"target_only_eos_rate":g["target_only_eos"].mean(),
           "accepted_prefix_per_verification":g["spec_accepted"].sum()/g["spec_rounds"].sum(),
           "draft_passes_per_token":g["spec_draft_passes"].sum()/g["spec_tokens"].sum(),
           "draft_tokens_fed_per_token":g["spec_draft_fed"].sum()/g["spec_tokens"].sum(),
           "spec_peak_gb":g["spec_peak_gb"].max(),"target_only_peak_gb":g["target_only_peak_gb"].max(),
           "target_prefill_ms":g["target_only_prefill_ms"].mean(),
           "spec_prefill_ms":(g["spec_target_prefill_ms"]+g["spec_draft_prefill_ms"]).mean(),
           "target_only_ms_per_token":g["target_only_decode_ms"].sum()/g["target_tokens"].sum(),
           "spec_ms_per_token":g["spec_decode_ms"].sum()/g["spec_tokens"].sum()}
    s["spec_total_ms"] = s["spec_decode_ms"]+s["spec_target_prefill_ms"]+s["spec_draft_prefill_ms"]
    s["target_total_ms"] = s["target_only_decode_ms"]+s["target_only_prefill_ms"]
    out["latency_spec_ms"],out["latency_target_ms"] = s["spec_total_ms"].sum()/len(g),s["target_total_ms"].sum()/len(g)
    for key,(a,b) in {"accepted_over_examined":("spec_accepted","spec_examined"),
                      "accepted_over_proposed":("spec_accepted","spec_proposed"),
                      "target_passes_per_token":("spec_target_passes","spec_tokens")}.items():
        out[key],out[f"{key}_lo"],out[f"{key}_hi"] = ratio_ci(s[a],s[b],n_boot,seed)
    # throughput ratio as target only time per token divided by speculative time per token over the same prompts
    idx = boot_indices(len(s),n_boot,seed)
    for key,ta,sa in (("decode_speedup","target_only_decode_ms","spec_decode_ms"),("end_to_end_speedup","target_total_ms","spec_total_ms")):
        t,tt,sp,st = (s[c].to_numpy(float) for c in (ta,"target_tokens",sa,"spec_tokens"))
        r = (t[idx].sum(1)/tt[idx].sum(1))/(sp[idx].sum(1)/st[idx].sum(1))
        out[key] = (t.sum()/tt.sum())/(sp.sum()/st.sum())
        out[f"{key}_lo"],out[f"{key}_hi"] = np.percentile(r,[2.5,97.5])
    per = (g["target_only_decode_ms"]/g["target_tokens"])/(g["spec_decode_ms"]/g["spec_tokens"])
    out["per_prompt_ratio_median"] = per.median()
    out["per_prompt_ratio_above_one"] = (per > 1).mean()
    out["spec_tokens_per_second"] = 1000/out["spec_ms_per_token"]
    out["target_tokens_per_second"] = 1000/out["target_only_ms_per_token"]
    return out


def benchmark_part(cfg,res,figs,tables,align):
    """Acceptance throughput round timing projected speedup and workload tables"""
    n_boot,seed = cfg["analysis"]["bootstrap"],cfg["seed"]
    runs = pd.read_csv(res/"benchmark"/"runs.csv")
    rows = []
    for (kind,d,g),part in runs.groupby(["kind","draft","gamma"],sort=False):
        for split in ("all",)+SPLITS:
            sub = part if split == "all" else part[part["split"] == split]
            if len(sub):
                rows.append({"kind":kind,"draft":d,"gamma":g,"split":split,**spec_summary(sub,n_boot,seed)})
    spec = pd.DataFrame(rows)
    spec.to_csv(tables/"speculative.csv",index=False)
    reps = [{"kind":k,"draft":d,"gamma":g,"rep":r,**spec_summary(p,200,seed)} for (k,d,g,r),p in
            runs.groupby(["kind","draft","gamma","rep"],sort=False)]
    pd.DataFrame(reps)[["kind","draft","gamma","rep","accepted_over_examined","decode_speedup","end_to_end_speedup",
                        "target_only_ms_per_token","spec_ms_per_token"]].to_csv(tables/"speculative_by_rep.csv",index=False)
    # bins of the realized output length of the speculative run
    runs["length_bin"] = ""
    for a,b in cfg["analysis"]["length_bins"]:
        runs.loc[(runs["spec_tokens"] >= a)&(runs["spec_tokens"] <= b),"length_bin"] = f"{a}-{b}" if a < b else str(a)
    eos = runs[runs["kind"] == "eos"]
    rows = [{"draft":d,"gamma":g,"length_bin":l,**spec_summary(p,n_boot,seed)} for (d,g,l),p in
            eos.groupby(["draft","gamma","length_bin"],sort=False)]
    pd.DataFrame(rows).to_csv(tables/"speculative_by_length.csv",index=False)
    # paired comparisons of the distilled drafts with the original draft over the same prompts
    pairs = [(s,"original") for s in STUDENTS]+[(f"on_{s}",f"off_{s}") for s in cfg["distill"]["seeds"]]
    out = []
    for g in cfg["benchmark"]["gammas"]:
        for split in ("all",)+SPLITS:
            sub = eos[eos["gamma"] == g]
            sub = sub if split == "all" else sub[sub["split"] == split]
            sums = {d:p.groupby("prompt_id").sum(numeric_only=True) for d,p in sub.groupby("draft")}
            for a,b in pairs:
                x,y = sums[a],sums[b]
                for key,(num,den) in {"accepted_over_examined":("spec_accepted","spec_examined"),
                                      "target_passes_per_token":("spec_target_passes","spec_tokens"),
                                      "spec_ms_per_token":("spec_decode_ms","spec_tokens")}.items():
                    diff,lo,hi = ratio_diff_ci((x[num],x[den]),(y[num],y[den]),n_boot,seed)
                    out.append({"gamma":g,"split":split,"first":a,"second":b,"metric":key,"diff":diff,"lo":lo,"hi":hi})
                idx = boot_indices(len(x),n_boot,seed)

                def speed(s,i=None):
                    """Decode throughput ratio against the adjacent target only runs"""
                    v = [s[c].to_numpy(float) for c in ("target_only_decode_ms","target_tokens","spec_decode_ms","spec_tokens")]
                    v = [q.sum() for q in v] if i is None else [q[i].sum(1) for q in v]
                    return (v[0]/v[1])/(v[2]/v[3])
                d = speed(x,idx)-speed(y,idx)
                out.append({"gamma":g,"split":split,"first":a,"second":b,"metric":"decode_speedup","diff":speed(x)-speed(y),
                            "lo":np.percentile(d,2.5),"hi":np.percentile(d,97.5)})
    pd.DataFrame(out).to_csv(tables/"speculative_pairs.csv",index=False)
    # four part time of every round from the synchronized repetition
    t = pd.read_csv(res/"benchmark"/"rounds_timed.csv")
    rows = []
    for (d,g,split),p in pd.concat([t.assign(scope="all"),t.assign(scope=t["split"])]).groupby(["draft","gamma","scope"],sort=False):
        n = p["rounds"].sum()
        rows.append({"draft":d,"gamma":g,"split":split,"rounds":int(n),**{f"{k}_per_round":p[k].sum()/n for k,_,_ in PARTS},
                     "round_ms":p["decode_ms"].sum()/n,"draft_ms_per_pass":p["draft_ms"].sum()/max(1,p["draft_passes"].sum()),
                     "same_tokens":p["same_tokens"].mean(),"timer_overhead":p["decode_ms"].sum()/p["untimed_decode_ms"].sum()-1,
                     "target_prefill_ms":p["target_prefill_ms"].mean(),"draft_prefill_ms":p["draft_prefill_ms"].mean()})
    rounds = pd.DataFrame(rows)
    rounds.to_csv(tables/"rounds.csv",index=False)
    # projected speedup from isolated costs and measured acceptance
    iso = pd.read_csv(res/"benchmark"/"isolated.csv")
    cost = cost_table(iso)
    cost.to_csv(tables/"isolated_costs.csv",index=False)
    near = min(cost["context"].unique(),key=lambda c:abs(c-128))
    # every merged draft has the same architecture so the merged student that was timed stands for all of them
    ratio = cost[cost["context"] == near].set_index(["draft","gamma"])
    rows = []
    for kind in ("eos","fixed"):
        for split in ("all","math") if kind == "eos" else ("all",):
            for d in DRAFTS:
                for g in cfg["benchmark"]["gammas"]:
                    r = ratio.loc[("original" if d == "original" else cfg["benchmark"]["unmerged"],g)]
                    s = spec[(spec["kind"] == kind)&(spec["draft"] == d)&(spec["gamma"] == g)&(spec["split"] == split)].iloc[0]
                    a = s["accepted_over_examined"]
                    tm = rounds[(rounds["draft"] == d)&(rounds["gamma"] == g)&(rounds["split"] == split)].iloc[0]
                    rows.append({"kind":kind,"split":split,"draft":d,"gamma":g,"alpha":a,"c":r["c"],"v":r["v"],
                                 "projected_speedup":(1+sum(a**j for j in range(1,g+1)))/(g*r["c"]+r["v"]),
                                 "measured_decode_speedup":s["decode_speedup"],"measured_lo":s["decode_speedup_lo"],
                                 "measured_hi":s["decode_speedup_hi"],"measured_end_to_end_speedup":s["end_to_end_speedup"],
                                 "projected_tokens_per_round":1+sum(a**j for j in range(1,g+1)),
                                 "measured_tokens_per_round":1/s["target_passes_per_token"],
                                 "break_even_alpha":break_even(g,r["c"],r["v"]),"max_projected_speedup":(1+g)/(g*r["c"]+r["v"]),
                                 "round_ms":tm["round_ms"],"model_round_ms":(g*r["c"]+r["v"])*r["target_step_ms"]})
    model = pd.DataFrame(rows)
    model.to_csv(tables/"cost_model.csv",index=False)
    # expected acceptance on the common banks against the measured share along each trajectory
    rows = []
    for d in DRAFTS:
        for split in ("all",)+SPLITS:
            a = align[(align["positions"] == "1-64")&(align["split"] == split)&(align["draft"] == d)].set_index("bank")
            row = {"draft":d,"split":split,**{f"expected_{b}":a.loc[b,"expected_accept"] for b in SHARED},
                   **{f"{m}_teacher_bank":a.loc["teacher",m] for m in ("forward_kl","reverse_kl","tv")}}
            for g in cfg["benchmark"]["gammas"]:
                s = spec[(spec["kind"] == "eos")&(spec["draft"] == d)&(spec["gamma"] == g)&(spec["split"] == split)].iloc[0]
                row[f"measured_gamma{g}"] = s["accepted_over_examined"]
                row[f"target_passes_per_token_gamma{g}"] = s["target_passes_per_token"]
            rows.append(row)
    expected = pd.DataFrame(rows)
    expected.to_csv(tables/"expected_vs_measured.csv",index=False)
    speed_figure(cfg,figs,spec,rounds,model,expected)
    probe_part(res,figs,tables)
    return spec


def speed_figure(cfg,figs,spec,rounds,model,expected):
    """Acceptance throughput round time and projected speedup in one figure"""
    gammas = cfg["benchmark"]["gammas"]
    eos = spec[(spec["kind"] == "eos")&(spec["split"] == "all")].set_index(["draft","gamma"])
    fig,axes = plt.subplots(1,4,figsize=(14,3.6))
    x = np.arange(len(DRAFTS))
    short = [d.replace("_","\n").replace("original","orig") for d in DRAFTS]
    for k,g in enumerate(gammas):
        for ax,col in ((axes[0],"accepted_over_examined"),(axes[1],"decode_speedup")):
            v = np.array([eos.loc[(d,g),col] for d in DRAFTS])
            lo = np.array([eos.loc[(d,g),f"{col}_lo"] for d in DRAFTS])
            hi = np.array([eos.loc[(d,g),f"{col}_hi"] for d in DRAFTS])
            ax.errorbar(x+0.12*(2*k-1),v,[v-lo,hi-v],fmt="os"[k],ms=5,capsize=2,color=("steelblue","firebrick")[k],label=f"γ = {g}")
    axes[0].set_ylabel("accepted over examined")
    axes[1].set_ylabel("decode speedup over target only")
    axes[1].axhline(1,color="black",lw=0.8,ls=":")
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(short,fontsize=8)
        ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=2,frameon=False,fontsize=8)
    axes[0].set_title("(a) measured acceptance",fontsize=10)
    axes[1].set_title("(b) measured throughput",fontsize=10)
    ax = axes[2]
    r = rounds[rounds["split"] == "all"].set_index(["draft","gamma"])
    pos,ticks = [],[]
    for i,g in enumerate(gammas):
        for k,d in enumerate(DRAFTS):
            p = i*(len(DRAFTS)+1)+k
            pos.append(p)
            ticks.append(d.replace("original","orig").replace("_"," "))
            bottom = 0
            for key,label,color in PARTS:
                val = r.loc[(d,g),f"{key}_per_round"]
                ax.bar(p,val,0.8,bottom=bottom,color=color,label=label if not i and not k else None)
                bottom += val
    ax.set_xticks(pos)
    ax.set_xticklabels(ticks,rotation=90,fontsize=7)
    ax.set_ylabel("time per round (ms)")
    ax.set_title("(c) round time, γ = 2 then γ = 4",fontsize=10)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.32),ncol=2,frameon=False,fontsize=7)
    ax = axes[3]
    m = model[(model["kind"] == "eos")&(model["split"] == "all")]
    for k,g in enumerate(gammas):
        for d in DRAFTS:
            q = m[(m["gamma"] == g)&(m["draft"] == d)].iloc[0]
            # drawing trained drafts hollow so the original draft stays visible underneath
            face = COLOR["original"] if d == "original" else "none"
            ax.plot(q["projected_speedup"],q["measured_decode_speedup"],"os"[k],color=COLOR[method_of(d)],ms=7,
                    markerfacecolor=face,markeredgewidth=1.4,
                    label=f"{LABEL[method_of(d)]}, γ = {g}" if d in ("original","off_6304","on_6304") else None)
    lim = [0.95*min(m["projected_speedup"].min(),m["measured_decode_speedup"].min()),
           1.05*max(m["projected_speedup"].max(),m["measured_decode_speedup"].max())]
    ax.plot(lim,lim,":",color="black",lw=0.8)
    ax.set_xlabel("projected speedup")
    ax.set_ylabel("measured decode speedup")
    ax.set_title("(d) cost model",fontsize=10)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.27),ncol=2,frameon=False,fontsize=7)
    fig.tight_layout(w_pad=3)
    save(fig,figs/"speculative_speed.png")
    fig,axes = plt.subplots(1,2,figsize=(8.4,3.3))
    e = expected[expected["split"] != "all"]
    mark = {"math":"o","familiar":"s","shifted":"^"}
    for ax,g in zip(axes,gammas):
        for d in DRAFTS:
            for s in SPLITS:
                q = e[(e["draft"] == d)&(e["split"] == s)].iloc[0]
                ax.plot(q["expected_teacher"],q[f"measured_gamma{g}"],mark[s],color=COLOR[method_of(d)],ms=6,
                        label=f"{LABEL[method_of(d)]}, {s}" if d in ("original","off_6304","on_6304") else None)
        ax.plot([0,1],[0,1],":",color="black",lw=0.8)
        ax.set_xlim(e["expected_teacher"].min()-0.05,1)
        ax.set_ylim(e[f"measured_gamma{g}"].min()-0.05,1)
        ax.set_xlabel("expected acceptance on target prefixes")
        ax.set_ylabel("measured accepted over examined")
        ax.set_title(f"γ = {g}",fontsize=10)
    fig.tight_layout(rect=(0,0.14,1,1))
    legend_below(fig,axes[0],3,-0.04)
    save(fig,figs/"expected_vs_measured.png")


def probe_part(res,figs,tables):
    """Cost ratio of draft and target when several sequences share a pass and the unmerged adapter overhead"""
    p = pd.read_csv(res/"benchmark"/"batch_probe.csv").set_index(["model","batch","tokens"])["ms"]
    rows = []
    for b in sorted({k[1] for k in p.index}):
        rows.append({"batch":b,"target_step_ms":p["target",b,1],"draft_step_ms":p["original",b,1],"c":p["original",b,1]/p["target",b,1],
                     "v2":p["target",b,3]/p["target",b,1],"v4":p["target",b,5]/p["target",b,1]})
        for g,key in ((2,"v2"),(4,"v4")):
            rows[-1][f"max_speedup_gamma{g}"] = (1+g)/(g*rows[-1]["c"]+rows[-1][key])
            rows[-1][f"break_even_gamma{g}"] = break_even(g,rows[-1]["c"],rows[-1][key])
    d = pd.DataFrame(rows)
    d.to_csv(tables/"batch_probe.csv",index=False)
    fig,ax = plt.subplots(figsize=(4.2,3.0))
    ax.plot(d["batch"],d["c"],marker="o",color="steelblue",label="draft step over target step, c")
    ax.plot(d["batch"],d["v2"],marker="s",color="darkorange",label="verification over target step, γ = 2")
    ax.plot(d["batch"],d["v4"],marker="^",color="firebrick",label="verification over target step, γ = 4")
    ax.set_xscale("log",base=2)
    ax.set_xticks(d["batch"])
    ax.set_xticklabels(d["batch"])
    ax.set_xlabel("sequences per pass")
    ax.set_ylabel("cost ratio")
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),frameon=False,fontsize=8)
    save(fig,figs/"batch_probe.png")
    u = pd.read_csv(res/"benchmark"/"unmerged_runs.csv")
    rows = [{"draft":k[0],"gamma":k[1],"ms_per_token":g["decode_ms"].sum()/g["tokens"].sum(),
             "accepted_over_examined":g["accepted"].sum()/g["examined"].sum(),"draft_prefill_ms":g["draft_prefill_ms"].mean(),
             "runs":len(g)} for k,g in u.groupby(["draft","gamma"])]
    iso = pd.read_csv(res/"benchmark"/"isolated.csv")
    iso = iso[(iso["pass"] == "step")&(iso["tokens"] == 1)&(iso["model"] != "target")]
    for r in rows:
        name = "unmerged" if r["draft"] == "unmerged" else [m for m in iso["model"].unique() if m not in ("original","unmerged")][0]
        r["isolated_step_ms"] = iso[iso["model"] == name]["ms"].mean()
    pd.DataFrame(rows).to_csv(tables/"unmerged.csv",index=False)


def quality_part(cfg,res,tables):
    """Standalone accuracy extraction format and truncation of the target and of every draft"""
    n_boot,seed = cfg["analysis"]["bootstrap"],cfg["seed"]
    q = pd.read_csv(res/"quality"/"responses.csv")
    rows = []
    for (m,split),g in q.groupby(["model","split"],sort=False):
        s = g.groupby("prompt_id")["correct"].agg(["sum","count"])
        acc,lo,hi = ratio_ci(s["sum"],s["count"],n_boot,seed)
        rows.append({"model":m,"split":split,"responses":len(g),"accuracy":acc,"lo":lo,"hi":hi,
                     "extracted":g["extracted"].astype(float).mean(),
                     "format_compliant":g["compliant"].astype(float).mean(),"contains_answer":g["contains_answer"].astype(float).mean(),
                     "truncated":g["truncated"].mean(),"mean_tokens":g["tokens"].mean(),
                     "correct_and_finished":(g["correct"].astype(bool)&~g["truncated"].astype(bool)).mean()})
    pd.DataFrame(rows).to_csv(tables/"quality.csv",index=False)
    out = []
    models = ["original"]+list(STUDENTS)
    # pairing short form answers on content as well since exact match mostly reflects the reply format
    for split in SPLITS:
        for metric in (("correct",) if split == "math" else ("correct","contains_answer")):
            wide = q[q["split"] == split].assign(v=lambda t:t[metric].astype(float)).groupby(["prompt_id","model"])["v"].mean().unstack()
            for a,b in ([(s,"original") for s in STUDENTS]+[(m,"target") for m in models]+
                        [(f"on_{s}",f"off_{s}") for s in cfg["distill"]["seeds"]]):
                d = (wide[a]-wide[b]).to_numpy(float)
                bo = d[boot_indices(len(d),n_boot,seed)].mean(1)
                out.append({"split":split,"metric":metric,"first":a,"second":b,"diff":d.mean(),
                            "lo":np.percentile(bo,2.5),"hi":np.percentile(bo,97.5)})
    pd.DataFrame(out).to_csv(tables/"quality_pairs.csv",index=False)


def deployment_part(cfg,res,tables,spec):
    """Requests needed before the distillation cost is recovered from measured time per request"""
    train = pd.read_csv(tables/"training.csv").set_index("name")
    rows = []
    for d in STUDENTS:
        for g in cfg["benchmark"]["gammas"]:
            a = spec[(spec["kind"] == "eos")&(spec["split"] == "all")&(spec["gamma"] == g)].set_index("draft")
            cost = train.loc[d,"total_seconds_full_pool"]
            cost_used = train.loc[d,"total_seconds_used_pool"]
            vs_target = (a.loc[d,"latency_target_ms"]-a.loc[d,"latency_spec_ms"])/1000
            vs_original = (a.loc["original","latency_spec_ms"]-a.loc[d,"latency_spec_ms"])/1000
            rows.append({"draft":d,"gamma":g,"distillation_seconds_full_pool":cost,"distillation_seconds_used_pool":cost_used,
                         "saving_vs_target_only_s_per_request":vs_target,
                         "saving_vs_original_draft_s_per_request":vs_original,
                         "requests_vs_target_only":cost/vs_target if vs_target > 0 else "never",
                         "requests_vs_original_draft":cost/vs_original if vs_original > 0 else "never"})
    pd.DataFrame(rows).to_csv(tables/"deployment.csv",index=False)


def qualitative_part(cfg,res):
    """Matched standalone responses chosen by rule with their scores"""
    texts = {(t["model"],t["rep"],t["prompt_id"]):t["text"] for t in read_gz(res/"quality"/"texts.jsonl.gz")}
    q = pd.read_csv(res/"quality"/"responses.csv")
    q = q[q["rep"] == 0]
    seed = cfg["distill"]["seeds"][0]
    names = ["target","original",f"off_{seed}",f"on_{seed}"]
    wide = q[q["model"].isin(names)].pivot(index="prompt_id",columns="model",values="correct").astype(bool)
    split = q.drop_duplicates("prompt_id").set_index("prompt_id")["split"]
    out = []
    for s in SPLITS:
        ids = [p for p in wide.index if split[p] == s]
        picks = {"first_prompt":ids[0]}
        for rule,cond in (("original_right_student_wrong",wide["original"]&~(wide[names[2]]&wide[names[3]])),
                          ("original_wrong_student_right",~wide["original"]&(wide[names[2]]|wide[names[3]]))):
            hits = [p for p in ids if cond[p]]
            if hits:
                picks[rule] = hits[0]
        for rule,pid in picks.items():
            out.append({"split":s,"rule":rule,"prompt_id":pid,"gold":str(q[q["prompt_id"] == pid]["gold"].iloc[0]),
                        "responses":{n:{"text":texts[(n,0,pid)],"correct":bool(wide.loc[pid,n])} for n in names}})
    save_json(res/"qualitative"/"matched_responses.json",out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--results",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(args.results or cfg["paths"]["results_dir"])
    figs,tables = res/"figures",res/"tables"
    figs.mkdir(parents=True,exist_ok=True)
    tables.mkdir(parents=True,exist_ok=True)
    checks_part(res,figs,tables)
    training_part(cfg,res,figs,tables)
    align = alignment_part(cfg,res,figs,tables)
    consistency_part(cfg,res,tables)
    spec = benchmark_part(cfg,res,figs,tables,align)
    quality_part(cfg,res,tables)
    deployment_part(cfg,res,tables,spec)
    qualitative_part(cfg,res)
    print("task 3 tables and figures written")


if __name__ == "__main__":
    main()
