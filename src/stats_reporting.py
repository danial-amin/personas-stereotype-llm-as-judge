"""Statistical helpers and APA formatting for RQ2/RQ3 outputs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.multitest import multipletests

LLM_LMM_SPEC = (
    "Linear mixed model with LLM fixed effects and a persona random intercept: "
    "outcome ~ predictors + C(model_key), groups = persona_id, random intercept only. "
    "The same five LLMs evaluated all twelve personas, so LLM and persona are crossed; "
    "LLM-specific means are absorbed by fixed LLM indicators and persona-specific "
    "clustering is modeled by the random intercept."
)


def apa_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def apa_b(b: float) -> str:
    return f"$b = {b:.3f}$"


def apa_pct(count: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{count / denom:.1%} ({count}/{denom})"


def latex_pct(count: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100 * count / denom:.1f}\\% ({count}/{denom})"


def cramers_v(chi2: float, n: int, k: int, r: int) -> float:
    if n == 0:
        return float("nan")
    return math.sqrt(chi2 / (n * min(k - 1, r - 1)))


def fit_llm_lmm(df: pd.DataFrame, formula: str) -> tuple[Any, str]:
    """
    Defensible crossed-structure alternative: LLM fixed effects + persona random intercept.
    """
    data = df.copy()
    model = smf.mixedlm(formula, data=data, groups=data["persona_id"], re_formula="1")
    result = model.fit(method="lbfgs", maxiter=200, disp=False)
    return result, LLM_LMM_SPEC


def fit_crossed_lmm(df: pd.DataFrame, formula: str, **kwargs: Any) -> tuple[Any, str]:
    """Backward-compatible alias for LLM mixed models."""
    return fit_llm_lmm(df, _ensure_model_fixed_effects(formula))


def _ensure_model_fixed_effects(formula: str) -> str:
    if "C(model_key)" in formula:
        return formula
    parts = formula.split("~", 1)
    rhs = parts[1].strip()
    return f"{parts[0].strip()} ~ {rhs} + C(model_key)"


def lmm_result_line(result: Any, param: str, label: str | None = None) -> dict[str, Any]:
    b = float(result.params[param])
    se = float(result.bse[param])
    z = float(result.tvalues[param])
    p = float(result.pvalues[param])
    ci_low, ci_high = result.conf_int().loc[param]
    return {
        "label": label or param,
        "b": round(b, 3),
        "se": round(se, 3),
        "z": round(z, 2),
        "p": p,
        "ci_low": round(float(ci_low), 3),
        "ci_high": round(float(ci_high), 3),
        "apa": (
            f"{apa_b(b)}, $SE = {se:.3f}$, $z = {z:.2f}$, {apa_p(p)}, "
            f"95\\% CI $[{ci_low:.3f}, {ci_high:.3f}]$"
        ),
    }


def within_llm_consistency(group: pd.Series) -> float:
    clean = group.dropna().astype(str).str.strip()
    clean = clean[clean.isin(["Yes", "No"])]
    if clean.empty:
        return float("nan")
    yes_count = int((clean == "Yes").sum())
    no_count = int((clean == "No").sum())
    mode_count = max(yes_count, no_count)
    return mode_count / len(clean)


def modal_binary(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(round(clean.mean()) >= 0.5)


def parse_dim_flag(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes"}


def cited_dims_from_row(row: pd.Series) -> set[str]:
    dims: set[str] = set()
    if parse_dim_flag(row.get("stereotype_age")):
        dims.add("Age")
    if parse_dim_flag(row.get("stereotype_gender")):
        dims.add("Gender")
    if parse_dim_flag(row.get("stereotype_occupation")):
        dims.add("Occupation")
    if parse_dim_flag(row.get("stereotype_other")):
        dims.add("Other")
    return dims


def jaccard(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def likert_descriptives(series: pd.Series) -> dict[str, Any]:
    values = series.dropna()
    if values.empty:
        return {"mean": None, "sd": None, "n": 0, "ci_low": None, "ci_high": None}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    n = int(len(values))
    se = sd / math.sqrt(n) if n else float("nan")
    t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
    return {
        "mean": round(mean, 2),
        "sd": round(sd, 2),
        "n": n,
        "ci_low": round(mean - t_crit * se, 2) if n > 1 else mean,
        "ci_high": round(mean + t_crit * se, 2) if n > 1 else mean,
    }


def parse_human_dims(value: Any) -> set[str]:
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def human_dim_binary(value: Any, dimension: str) -> int:
    return int(dimension in parse_human_dims(value))


def mirror_dim_binary(row: pd.Series, col: str) -> int:
    return int(parse_dim_flag(row.get(col)))


def bootstrap_mean_ci(diffs: np.ndarray, *, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return float("nan"), float("nan")
    boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def wilcoxon_rank_biserial(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = ranks[nonzero > 0].sum()
    return float((2 * w_plus) / (n * (n + 1)) - 1)


def paired_likert_test(human: pd.Series, mirror: pd.Series) -> dict[str, Any]:
    paired = pd.DataFrame({"human": human, "mirror": mirror})
    n_matched = len(paired)
    valid = paired.dropna()
    n_valid = len(valid)
    if n_valid < 2:
        return {
            "test": "n/a",
            "reason": "insufficient valid pairs",
            "n_matched": n_matched,
            "n_valid_both": n_valid,
        }

    diffs = (valid["mirror"] - valid["human"]).to_numpy(dtype=float)
    nonzero_mask = diffs != 0
    n_nonzero = int(nonzero_mask.sum())
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1)) if n_valid > 1 else 0.0
    ci_low, ci_high = bootstrap_mean_ci(diffs)
    dz = mean_diff / sd_diff if sd_diff else float("nan")

    if n_nonzero == 0:
        w_stat, p = 0.0, 1.0
        r_rb = 0.0
    else:
        w_stat, p = stats.wilcoxon(valid.loc[nonzero_mask, "human"], valid.loc[nonzero_mask, "mirror"])
        r_rb = wilcoxon_rank_biserial(diffs)

    return {
        "test": "Wilcoxon signed-rank test",
        "reason": (
            "Ordinal Likert outcomes were compared with a nonparametric paired test on pairs "
            "with non-missing human and GPT-5.4 responses. The mean paired-difference CI is a "
            "bootstrap CI for the mean paired difference among valid pairs."
        ),
        "n_matched": n_matched,
        "n_valid_both": n_valid,
        "n_wilcoxon_nonzero": n_nonzero,
        "human_mean": round(float(valid["human"].mean()), 2),
        "human_sd": round(float(valid["human"].std(ddof=1)), 2),
        "mirror_mean": round(float(valid["mirror"].mean()), 2),
        "mirror_sd": round(float(valid["mirror"].std(ddof=1)), 2),
        "mean_diff": round(mean_diff, 2),
        "sd_diff": round(sd_diff, 2),
        "mean_diff_ci_low": round(ci_low, 2),
        "mean_diff_ci_high": round(ci_high, 2),
        "statistic": round(float(w_stat), 2),
        "p": float(p),
        "r_rb": round(r_rb, 3),
        "effect_size_dz": round(dz, 2),
        "apa": (
            f"$N_{{matched}} = {n_matched}$; $n_{{valid}} = {n_valid}$; "
            f"$M_{{human}} = {valid['human'].mean():.2f}$, $SD = {valid['human'].std(ddof=1):.2f}$; "
            f"$M_{{GPT\\text{{-}}5.4}} = {valid['mirror'].mean():.2f}$, "
            f"$SD = {valid['mirror'].std(ddof=1):.2f}$; "
            f"mean paired difference $M_{{diff}} = {mean_diff:.2f}$, "
            f"95\\% bootstrap CI for mean paired difference $[{ci_low:.2f}, {ci_high:.2f}]$; "
            f"Wilcoxon $W = {float(w_stat):.2f}$, $N_{{nonzero}} = {n_nonzero}$, {apa_p(float(p))}, "
            f"$r_{{rb}} = {r_rb:.3f}$; standardized mean difference $d_z = {dz:.2f}$"
        ),
    }


def mcnemar_attribution_batch(
    paired: pd.DataFrame,
    *,
    condition: str,
    human_cols: dict[str, str],
    mirror_cols: dict[str, str],
    correction: str = "holm",
) -> dict[str, Any]:
    sub = paired[paired["condition"] == condition].copy()
    rows: list[dict[str, Any]] = []
    raw_ps: list[float] = []
    for dim, human_col in human_cols.items():
        mirror_col = mirror_cols[dim]
        human_bin = sub[human_col].astype(int)
        mirror_bin = sub[mirror_col].astype(int)
        result = mcnemar_binary(human_bin, mirror_bin)
        result["dimension"] = dim
        rows.append(result)
        raw_ps.append(float(result["p"]))

    reject, p_adj, _, _ = multipletests(raw_ps, method=correction)
    for row, p_corr, sig in zip(rows, p_adj, reject):
        row["p_corrected"] = float(p_corr)
        row["significant_corrected"] = bool(sig)
        row["apa"] = (
            f"{row['dimension'].capitalize()}: exact agreement = {row['exact_agreement']:.3f}; "
            f"$\\kappa = {row['cohens_kappa']:.3f}$; McNemar exact {apa_p(row['p'])}, "
            f"Holm-corrected {apa_p(row['p_corrected'])}; "
            f"discordant pairs = {row['discordant_human_yes_mirror_no']} "
            f"(human selected, GPT-5.4 did not) and {row['discordant_human_no_mirror_yes']} "
            f"(GPT-5.4 selected, human did not)"
        )

    return {
        "condition": condition,
        "n_pairs": int(len(sub)),
        "correction": correction,
        "dimensions": rows,
        "primary_dimension_note": (
            "Mutually exclusive primary-dimension comparisons cannot be reproduced because "
            "selection order was not preserved in the GPT-5.4 mirror output."
        ),
    }


def mcnemar_binary(human: pd.Series, mirror: pd.Series) -> dict[str, Any]:
    paired = pd.DataFrame({"human": human.astype(int), "mirror": mirror.astype(int)}).dropna()
    table = pd.crosstab(paired["human"], paired["mirror"])
    # Ensure 2x2
    for val in [0, 1]:
        if val not in table.index:
            table.loc[val] = 0
        if val not in table.columns:
            table[val] = 0
    table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    result = mcnemar(table.values, exact=True)
    agree = float((paired["human"] == paired["mirror"]).mean())
    kappa = cohens_kappa(paired["human"], paired["mirror"])
    b01 = int(table.loc[0, 1])
    b10 = int(table.loc[1, 0])
    return {
        "n_pairs": len(paired),
        "table": table.to_dict(),
        "exact_agreement": round(agree, 3),
        "cohens_kappa": kappa,
        "discordant_human_yes_mirror_no": b10,
        "discordant_human_no_mirror_yes": b01,
        "human_yes_rate": round(float(paired["human"].mean()), 3),
        "mirror_yes_rate": round(float(paired["mirror"].mean()), 3),
        "delta_pp": round(100 * (paired["mirror"].mean() - paired["human"].mean()), 1),
        "p": float(result.pvalue),
        "apa": (
            f"Exact agreement = {agree:.3f}; $\\kappa = {kappa:.3f}$; "
            f"McNemar exact {apa_p(float(result.pvalue))}; "
            f"discordant pairs = {b10} (human Yes, GPT-5.4 No) and {b01} (human No, GPT-5.4 Yes)"
        ),
    }


def cohens_kappa(y1: pd.Series, y2: pd.Series) -> float | None:
    a = y1.astype(int)
    b = y2.astype(int)
    if len(a) == 0:
        return None
    po = float((a == b).mean())
    p_a = float(a.mean())
    p_b = float(b.mean())
    pe = p_a * p_b + (1 - p_a) * (1 - p_b)
    if pe == 1.0:
        return 1.0 if po == 1.0 else None
    return round((po - pe) / (1 - pe), 3)


def fleiss_kappa_modal(df: pd.DataFrame, rating_col: str, condition: str) -> dict[str, Any]:
    subset = df[df["condition"] == condition].copy()
    modal = (
        subset.groupby(["persona_id", "model_key"])[rating_col]
        .mean()
        .reset_index()
        .assign(modal=lambda d: (d[rating_col] >= 0.5).astype(int))
    )
    pivot = modal.pivot(index="persona_id", columns="model_key", values="modal")
    table = np.zeros((len(pivot), 2))
    for i, (_, row) in enumerate(pivot.iterrows()):
        vals = row.dropna().astype(int)
        table[i, 0] = int((vals == 0).sum())
        table[i, 1] = int((vals == 1).sum())
    if table.sum() == 0:
        return {"kappa": None, "note": "No ratings available"}
    kappa = float(fleiss_kappa(table))
    observed = float(
        (
            modal.groupby("persona_id")["modal"]
            .apply(lambda s: (s == s.mode().iloc[0]).mean())
            .mean()
        )
    )
    return {
        "kappa": round(kappa, 3),
        "observed_agreement": round(observed, 3),
        "n_personas": pivot.shape[0],
        "n_raters": pivot.shape[1],
        "note": "Fleiss' kappa on modal binary ratings across LLMs.",
    }
