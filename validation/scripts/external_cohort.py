#!/usr/bin/env python3
"""
Two checks the lab could not run on itself.

The lab's own evaluation stage says it plainly: the model was "validated on a held-out part of the
same dataset; a second, independent dataset is the stronger test and is not available here". It is
available — it just lives in a different GEO study. So:

  1. EXTERNAL REPLICATION. Take the signature and the model built on the discovery cohort
     (GSE193777, household contacts in southern India) and apply them to an entirely separate study
     (GSE193978, 76 patients across Indonesia, South Africa and Romania) where active disease at
     diagnosis can be compared with the same patients after six months of treatment. Different
     patients, different continents, different study design.

  2. IS IT TUBERCULOSIS, OR IS IT SEX? In the discovery cohort the first principal component was
     best explained by the patient's sex, not by disease, and the strongest feature the lab's model
     relies on is RPS4X, which sits on the X chromosome. So: how many of the model's features are
     sex-linked, how well does the very same pipeline predict sex instead of tuberculosis, and does
     the tuberculosis model survive having the sex chromosomes taken away from it?

Usage:
  python external_cohort.py <discovery_counts.gz> <discovery_meta.tsv> \
                            <validation_counts.gz> <validation_meta.tsv> \
                            <de_results.tsv> <gene_annotation.tsv> <outdir>
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PADJ, LFC, N_FEATURES, C = 0.05, 1.0, 200, 0.01
SEED = 42

# Y-linked genes and XIST, the genes that separate the sexes in blood. Kept explicit rather than
# pulled from an annotation build, so the reader can see exactly what counts as "sex-linked" here.
Y_GENES = {"RPS4Y1", "RPS4Y2", "DDX3Y", "EIF1AY", "KDM5D", "UTY", "USP9Y", "NLGN4Y", "TXLNGY",
           "ZFY", "TMSB4Y", "PRKY", "TTTY14", "TTTY15", "LINC00278"}
XIST = {"XIST", "TSIX"}
SEX_GENES = Y_GENES | XIST


def load_counts(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    return df


def log_cpm(counts):
    lib = counts.sum(axis=0)
    return np.log2(counts.div(lib, axis=1) * 1e6 + 1)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def auc_from_groups(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg])


def main():
    (dc, dm, vc, vm, de_path, ann_path, out) = sys.argv[1:8]
    out = pathlib.Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    report = {}

    print("reading the two cohorts", flush=True)
    disc, val = load_counts(dc), load_counts(vc)
    dmeta = pd.read_csv(dm, sep="\t", index_col=0)
    vmeta = pd.read_csv(vm, sep="\t", index_col=0)
    de = pd.read_csv(de_path, sep="\t")
    ann = pd.read_csv(ann_path, sep="\t", dtype=str)
    sym = dict(zip(ann["gene_id"].astype(str), ann["symbol"].astype(str)))

    shared = disc.index.intersection(val.index)
    print(f"discovery {disc.shape}, validation {val.shape}, genes in common {len(shared)}", flush=True)
    report["genes_in_common"] = int(len(shared))
    report["discovery_samples"] = int(disc.shape[1])
    report["validation_samples"] = int(val.shape[1])

    # tidy the validation cohort's timepoint column: it contains both "Month_6" and "Month 6"
    tp_raw = vmeta["timepoint"].astype(str)
    tp = tp_raw.str.replace(" ", "_", regex=False)
    report["timepoint_values_raw"] = {k: int(v) for k, v in tp_raw.value_counts().items()}
    report["timepoint_values_tidied"] = {k: int(v) for k, v in tp.value_counts().items()}
    vmeta = vmeta.assign(timepoint_tidy=tp)

    # ---------------------------------------------------------------- 1. the signature, externally
    sig = de.dropna(subset=["padj"])
    sig = sig[(sig["padj"] < PADJ) & (sig["log2_fold_change"].abs() > LFC)]
    up = [str(g) for g in sig.loc[sig["log2_fold_change"] > 0, "gene_id"] if str(g) in shared]
    dn = [str(g) for g in sig.loc[sig["log2_fold_change"] < 0, "gene_id"] if str(g) in shared]
    report["signature"] = {"up_genes": len(up), "down_genes": len(dn),
                           "from": "Active TB vs household contacts, discovery cohort"}
    print(f"signature: {len(up)} up, {len(dn)} down", flush=True)

    vlog = log_cpm(val.loc[shared])
    z = vlog.sub(vlog.mean(axis=1), axis=0).div(vlog.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
    score = z.loc[up].mean() - z.loc[dn].mean()

    keep = tp.isin(["Diagnosis", "Month_6"])
    sc = pd.DataFrame({"sample": score.index, "score": score.values}).set_index("sample")
    sc = sc.join(vmeta[["timepoint_tidy", "disease_category", "field_site", "patient_id"]])
    sc.to_csv(out / "tables" / "validation_signature_scores.tsv", sep="\t")
    diag = sc.loc[sc["timepoint_tidy"] == "Diagnosis", "score"].dropna()
    m6 = sc.loc[sc["timepoint_tidy"] == "Month_6", "score"].dropna()
    u = stats.mannwhitneyu(diag, m6, alternative="two-sided")
    report["external_signature"] = {
        "n_diagnosis": int(len(diag)), "n_month6": int(len(m6)),
        "median_score_diagnosis": round(float(diag.median()), 4),
        "median_score_month6": round(float(m6.median()), 4),
        "mann_whitney_p": float(u.pvalue),
        "auc_diagnosis_vs_month6": round(float(auc_from_groups(diag, m6)), 4),
        "cliffs_delta": round(float(cliffs_delta(diag, m6)), 4),
    }
    print("external signature:", json.dumps(report["external_signature"]), flush=True)

    # paired within patient, which the design allows
    paired = sc[sc["timepoint_tidy"].isin(["Diagnosis", "Month_6"])].reset_index()
    wide = paired.pivot_table(index="patient_id", columns="timepoint_tidy", values="score")
    wide = wide.dropna()
    if len(wide) >= 5:
        w = stats.wilcoxon(wide["Diagnosis"], wide["Month_6"])
        report["external_signature_paired"] = {
            "n_patients_with_both": int(len(wide)),
            "median_fall": round(float((wide["Diagnosis"] - wide["Month_6"]).median()), 4),
            "wilcoxon_p": float(w.pvalue),
            "patients_whose_score_fell": int((wide["Diagnosis"] > wide["Month_6"]).sum()),
        }
        wide.to_csv(out / "tables" / "validation_paired_scores.tsv", sep="\t")
        print("paired:", json.dumps(report["external_signature_paired"]), flush=True)

    # ---------------------------------------------------------------- 2. the model, externally
    dlog = log_cpm(disc.loc[shared])
    cult = dmeta["mtb_culture"].astype(str)
    train_mask = cult.isin(["Positive", "Negative"])
    Xtr = dlog.loc[:, dlog.columns.intersection(cult[train_mask].index)].T
    ytr = (cult.loc[Xtr.index] == "Positive").astype(int).to_numpy()

    def pipe():
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=5000))])

    def top_features(X, y, n=N_FEATURES, exclude=()):
        cols = [c for c in X.columns if c not in exclude]
        F, _ = f_classif(X[cols].to_numpy(), y)
        F = np.nan_to_num(F)
        return [cols[i] for i in np.argsort(F)[::-1][:n]]

    feats = top_features(Xtr, ytr)
    model = pipe().fit(Xtr[feats], ytr)
    cv = cross_val_score(pipe(), Xtr[feats], ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                         scoring="roc_auc")
    report["discovery_model"] = {"n_train": int(len(ytr)), "n_positive": int(ytr.sum()),
                                 "cv_auc_mean": round(float(cv.mean()), 4), "cv_auc_sd": round(float(cv.std()), 4)}

    Xva = log_cpm(val.loc[shared]).T
    sub = Xva.loc[Xva.index.intersection(vmeta.index[keep])]
    yva = (vmeta.loc[sub.index, "timepoint_tidy"] == "Diagnosis").astype(int).to_numpy()
    prob = model.predict_proba(sub[feats])[:, 1]
    report["external_model"] = {
        "n_validation": int(len(yva)), "n_diagnosis": int(yva.sum()),
        "auc": round(float(roc_auc_score(yva, prob)), 4),
        "what_is_being_predicted": "trained on culture-positive vs culture-negative in the discovery "
                                   "cohort, applied to active disease at diagnosis vs the same patients "
                                   "after six months of treatment in an independent cohort",
    }
    pd.DataFrame({"sample": sub.index, "probability": prob,
                  "timepoint": vmeta.loc[sub.index, "timepoint_tidy"].to_numpy(),
                  "field_site": vmeta.loc[sub.index, "field_site"].to_numpy()}
                 ).to_csv(out / "tables" / "validation_model_predictions.tsv", sep="\t", index=False)
    print("external model:", json.dumps(report["external_model"]), flush=True)

    # per field site, because the validation cohort spans three countries
    rows = []
    for site, idx in vmeta.loc[sub.index].groupby("field_site").groups.items():
        m = sub.index.isin(idx)
        if len(set(yva[m])) == 2:
            rows.append({"field_site": site, "n": int(m.sum()),
                         "auc": round(float(roc_auc_score(yva[m], prob[m])), 4)})
    pd.DataFrame(rows).to_csv(out / "tables" / "validation_model_by_site.tsv", sep="\t", index=False)
    report["external_model_by_site"] = rows
    print("by site:", rows, flush=True)

    # ---------------------------------------------------------------- 3. is it sex?
    sex_ids = {g for g in shared if sym.get(g, "") in SEX_GENES}
    in_model = [g for g in feats if g in sex_ids]
    report["sex_check"] = {
        "sex_linked_genes_detected": len(sex_ids),
        "sex_linked_genes_among_the_models_200_features": len(in_model),
        "their_symbols": sorted(sym.get(g, g) for g in in_model),
    }

    gender = dmeta["gender"].astype(str)
    have_sex = gender.isin(["Male", "Female"])
    Xs = dlog.loc[:, dlog.columns.intersection(gender[have_sex].index)].T
    ys = (gender.loc[Xs.index] == "Male").astype(int).to_numpy()
    fs = top_features(Xs, ys)
    cvs = cross_val_score(pipe(), Xs[fs], ys, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          scoring="roc_auc")
    report["sex_check"]["predicting_sex_instead"] = {
        "n_with_sex_recorded": int(len(ys)), "n_missing_sex": int((~have_sex).sum()),
        "cv_auc_mean": round(float(cvs.mean()), 4), "cv_auc_sd": round(float(cvs.std()), 4),
    }

    # the tuberculosis model with the sex chromosomes taken away
    feats_nosex = top_features(Xtr, ytr, exclude=sex_ids)
    cvn = cross_val_score(pipe(), Xtr[feats_nosex], ytr,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")
    model_ns = pipe().fit(Xtr[feats_nosex], ytr)
    prob_ns = model_ns.predict_proba(sub[feats_nosex])[:, 1]
    report["sex_check"]["tb_model_without_sex_linked_genes"] = {
        "cv_auc_mean": round(float(cvn.mean()), 4), "cv_auc_sd": round(float(cvn.std()), 4),
        "external_auc": round(float(roc_auc_score(yva, prob_ns)), 4),
    }
    print("sex check:", json.dumps(report["sex_check"], indent=1), flush=True)

    (out / "tables" / "validation_summary.json").write_text(json.dumps(report, indent=1))
    print("\ndone ->", out, flush=True)


if __name__ == "__main__":
    main()
