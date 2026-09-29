"""
Loading, cleaning and the scientific analysis.

The dataset gives four measurements per star: effective temperature, luminosity,
radius and absolute magnitude. Those are not independent -- a star radiates as a
sphere, so L = 4*pi*R^2*sigma*T^4 ties them together with no free parameters.
That makes it possible to *audit* the catalogue rather than merely describe it,
which is what most of this module does.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

# IAU 2015 nominal solar values.
T_EFF_SUN = 5772.0          # solar effective temperature [K]
M_BOL_SUN = 4.74            # solar absolute bolometric magnitude [mag]

TYPE_LABELS = {
    0: "Brown Dwarf", 1: "Red Dwarf", 2: "White Dwarf",
    3: "Main Sequence", 4: "Supergiant", 5: "Hypergiant",
}
SPECTRAL_ORDER = ["O", "B", "A", "F", "G", "K", "M"]

#: Canonical Morgan-Keenan effective-temperature boundaries [K].
SPECTRAL_RANGES = {
    "O": (33_000, np.inf), "B": (10_000, 33_000), "A": (7_300, 10_000),
    "F": (6_000, 7_300), "G": (5_300, 6_000), "K": (3_900, 5_300),
    "M": (0, 3_900),
}

#: The raw `Color` column holds 19 spellings of 8 colours, differing by case,
#: hyphenation and trailing spaces. Keys here are the output of `_colour_key`.
COLOUR_SYNONYMS = {
    "blue": "Blue", "blue white": "Blue-White", "white": "White",
    "whitish": "White", "yellow white": "Yellow-White",
    "white yellow": "Yellow-White", "yellowish white": "Yellow-White",
    "yellowish": "Yellowish", "pale yellow orange": "Orange",
    "orange": "Orange", "orange red": "Orange-Red", "red": "Red",
}
COLOUR_ORDER = ["Blue", "Blue-White", "White", "Yellow-White",
                "Yellowish", "Orange", "Orange-Red", "Red"]

#: Header spellings that circulate for this dataset -> canonical names.
_ALIASES = {
    "temperature": "Temperature", "temperature k": "Temperature",
    "luminosity": "L", "luminosity l lo": "L", "radius": "R",
    "radius r ro": "R", "absolute magnitude": "A_M", "a m": "A_M",
    "absolute magnitude mv": "A_M", "star color": "Color", "colour": "Color",
    "spectral class": "Spectral_Class", "star type": "Type",
}
#: A textual recoding of the target ships with some copies of this file. It must
#: never reach the model, so it is dropped at load time.
_LEAKY = {"star category", "star_category", "category"}

AXIS_LABELS = {
    "Temperature": "Effective temperature  $T_{\\mathrm{eff}}$  [K]",
    "L": "Luminosity  $L/L_{\\odot}$",
    "R": "Radius  $R/R_{\\odot}$",
    "A_M": "Absolute magnitude  $M_{v}$  [mag]",
    "logL": "$\\log_{10}(L/L_{\\odot})$",
}


def _key(text: object) -> str:
    """Lower-case and collapse punctuation, so spellings compare equal."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def load_and_clean(path) -> tuple[pd.DataFrame, dict]:
    """Read the catalogue, normalise it, and add the derived physical columns.

    Returns the cleaned frame plus a small dict of cleaning facts for the report.
    Rows that look physically odd are **flagged, not deleted**: with 240 stars,
    dropping a fifth of them would bias every later result, and the anomalies
    turn out to be the most interesting thing in the dataset.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No dataset at '{path}'. Expected a CSV with the columns "
            "Temperature, L, R, A_M, Color, Spectral_Class and Type "
            "(any of the usual spellings).")

    df = pd.read_csv(path, skipinitialspace=True)
    notes = {"n_raw": len(df), "columns_raw": list(df.columns)}

    dropped = [c for c in df.columns if _key(c) in _LEAKY]
    df = df.drop(columns=dropped)
    notes["dropped"] = dropped

    df = df.rename(columns={c: _ALIASES.get(_key(c), str(c).strip()) for c in df.columns})
    required = ["Temperature", "L", "R", "A_M", "Color", "Spectral_Class", "Type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(missing)}. "
                         f"Found: {', '.join(map(str, df.columns))}")
    df = df[required].copy()

    notes["n_duplicates"] = int(df.duplicated().sum())
    notes["n_missing"] = int(df.isna().sum().sum())
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    # Normalise the free-text colour column: key first, then look the key up, so
    # one table entry covers every spelling of a colour rather than one each.
    notes["colours_raw"] = int(df["Color"].nunique())
    keys = df["Color"].map(_key)
    df["Color"] = keys.map(COLOUR_SYNONYMS).fillna(keys.str.title())
    notes["colours_clean"] = int(df["Color"].nunique())

    df["Spectral_Class"] = df["Spectral_Class"].astype(str).str.strip().str.upper().str[0]
    df["Type"] = df["Type"].astype(int)
    df["Type_Label"] = df["Type"].map(TYPE_LABELS)

    # Ordered categoricals, so every table and legend sorts physically (low mass
    # to high, hot to cool) instead of alphabetically.
    df["Type_Label"] = pd.Categorical(
        df["Type_Label"], categories=[TYPE_LABELS[t] for t in sorted(TYPE_LABELS)], ordered=True)
    df["Spectral_Class"] = pd.Categorical(
        df["Spectral_Class"], categories=SPECTRAL_ORDER, ordered=True)
    df["Color"] = pd.Categorical(
        df["Color"], categories=COLOUR_ORDER + sorted(set(df["Color"]) - set(COLOUR_ORDER)),
        ordered=True)

    # Derived physics, computed once.
    df["logT"] = np.log10(df["Temperature"])
    df["logL"] = np.log10(df["L"])
    df["logR"] = np.log10(df["R"])
    # Stefan-Boltzmann in solar units: L/Lsun = (R/Rsun)^2 * (T/Tsun)^4
    df["L_sb"] = df["R"] ** 2 * (df["Temperature"] / T_EFF_SUN) ** 4
    df["sb_residual"] = np.log10(df["L"] / df["L_sb"])       # in dex
    df["M_bol"] = M_BOL_SUN - 2.5 * df["logL"]

    notes["n_clean"] = len(df)
    return df, notes


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Median properties of each stellar population.

    The median rather than the mean: with 40 stars per class and ranges spanning
    orders of magnitude, the mean is dragged around by the largest member.
    """
    out = df.groupby("Type_Label", observed=True).agg(
        n=("Temperature", "size"),
        T_median=("Temperature", "median"),
        L_median=("L", "median"),
        R_median=("R", "median"),
        Mv_median=("A_M", "median"),
    )
    measured = [c for c in out.columns if c != "n"]
    out[measured] = out[measured].map(lambda v: float(f"{v:.4g}"))
    return out


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Temperature-luminosity correlation, globally and within each population.

    Pearson is computed on the logarithms, where a straight line is a power law.
    The global and per-population answers differ dramatically -- see the report.
    """
    rows = []

    def add(label, sub):
        pear = stats.pearsonr(sub["logT"], sub["logL"])
        spear = stats.spearmanr(sub["logT"], sub["logL"])
        rows.append({
            "population": label, "n": len(sub),
            "pearson_r": round(float(pear.statistic), 3),
            "spearman_rho": round(float(spear.statistic), 3),
            "p_value": f"{pear.pvalue:.2g}",
        })

    add("ALL STARS", df)
    for label, group in df.groupby("Type_Label", observed=True):
        add(str(label), group)
    return pd.DataFrame(rows).set_index("population")


def population_test(df: pd.DataFrame, column: str = "logL") -> dict:
    """Are the six populations genuinely distinct in `column`?

    Kruskal-Wallis rather than ANOVA: the groups have wildly different variances,
    which violates ANOVA's assumptions, and the rank-based test assumes almost
    nothing. Eta-squared is reported alongside because with 240 stars almost any
    difference is "significant" -- the size is the interesting part.
    """
    groups = [g[column].to_numpy() for _, g in df.groupby("Type_Label", observed=True)]
    kruskal = stats.kruskal(*groups)

    values = np.concatenate(groups)
    grand_mean = values.mean()
    ss_between = sum(g.size * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = float(((values - grand_mean) ** 2).sum())

    return {
        "column": column,
        "H": float(kruskal.statistic),
        "p_value": float(kruskal.pvalue),
        "eta_squared": ss_between / ss_total,
    }


def _sb_plane(xdata, log_c, alpha, beta):
    """log L = log C + alpha*log R + beta*log(T/Tsun). Theory: 0, 2, 4."""
    log_r, log_t_rel = xdata
    return log_c + alpha * log_r + beta * log_t_rel


def fit_stefan_boltzmann(df: pd.DataFrame) -> dict:
    """Fit the luminosity-radius-temperature scaling and compare it with theory.

    The Stefan-Boltzmann law fixes the exponents at exactly 2 and 4. Fitting them
    as free parameters and seeing what comes back is a direct test of whether the
    catalogue could describe real objects.
    """
    log_r = df["logR"].to_numpy()
    log_t_rel = df["logT"].to_numpy() - np.log10(T_EFF_SUN)
    log_l = df["logL"].to_numpy()

    popt, pcov = optimize.curve_fit(
        _sb_plane, np.vstack([log_r, log_t_rel]), log_l, p0=[0.0, 2.0, 4.0])
    perr = np.sqrt(np.diag(pcov))          # standard errors of the parameters

    z_alpha = (popt[1] - 2.0) / perr[1]
    z_beta = (popt[2] - 4.0) / perr[2]
    return {
        "n": len(df),
        "log_c": popt[0], "log_c_err": perr[0],
        "alpha": popt[1], "alpha_err": perr[1],
        "beta": popt[2], "beta_err": perr[2],
        "z_alpha": z_alpha, "p_alpha": float(2 * stats.norm.sf(abs(z_alpha))),
        "z_beta": z_beta, "p_beta": float(2 * stats.norm.sf(abs(z_beta))),
    }


def sb_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Per-population agreement with the *unfitted* Stefan-Boltzmann law.

    A Wilcoxon signed-rank test asks whether each population's residuals are
    centred on zero, i.e. whether that population is physically self-consistent.
    """
    rows = []
    for label, group in df.groupby("Type_Label", observed=True):
        residual = group["sb_residual"].to_numpy()
        p = float(stats.wilcoxon(residual).pvalue)
        rows.append({
            "population": str(label),
            "median_ratio": round(float(10 ** np.median(residual)), 3),
            "pct_within_2x": round(100 * float((np.abs(residual) < np.log10(2)).mean()), 1),
            "wilcoxon_p": f"{p:.2g}",
            "consistent": "yes" if p > 0.05 else "no",
        })
    return pd.DataFrame(rows).set_index("population")


def fit_power_law(x, y, n_boot: int = 2000, seed: int = 42) -> dict:
    """Least-squares power-law fit y = A*x^alpha, with a bootstrap interval.

    The fit is done on the logarithms, so relative rather than absolute error is
    minimised -- appropriate when the data span orders of magnitude. The interval
    comes from resampling the stars with replacement and refitting, which needs
    no assumption about how the residuals are distributed.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    log_x, log_y = np.log10(x), np.log10(y)

    def line(t, intercept, slope):
        return intercept + slope * t

    popt, pcov = optimize.curve_fit(line, log_x, log_y, p0=[0.0, 1.0])
    predicted = line(log_x, *popt)
    r_squared = 1 - ((log_y - predicted) ** 2).sum() / ((log_y - log_y.mean()) ** 2).sum()

    rng = np.random.default_rng(seed)
    x_curve = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    boot = np.empty((n_boot, x_curve.size))
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(x.size, size=x.size, replace=True)    # resample WITH replacement
        p, _ = optimize.curve_fit(line, log_x[idx], log_y[idx], p0=popt)
        slopes[i] = p[1]
        boot[i] = 10 ** line(np.log10(x_curve), *p)

    return {
        "n": int(x.size),
        "alpha": float(popt[1]), "alpha_err": float(np.sqrt(np.diag(pcov))[1]),
        "ci": (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))),
        "r_squared": float(r_squared),
        "x_curve": x_curve,
        "y_curve": 10 ** line(np.log10(x_curve), *popt),
        "y_lo": np.percentile(boot, 2.5, axis=0),
        "y_hi": np.percentile(boot, 97.5, axis=0),
    }
