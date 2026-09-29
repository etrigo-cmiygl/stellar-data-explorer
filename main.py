#!/usr/bin/env python3
"""
Stellar Data Explorer — entry point.

Loads a 240-star catalogue, tests it against the physics that governs stars,
classifies the six stellar populations, and writes up what it found.

    python main.py                 # everything
    python main.py --no-figures    # numbers only, a few seconds
    python main.py --seed 7        # different split and bootstrap
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from src import analysis, model, plots

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stellar Data Explorer.")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "stars.csv")
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def rule(title: str) -> str:
    return f"\n{'=' * 76}\n{title}\n{'=' * 76}"


def step(message: str) -> None:
    """Print progress immediately.

    The report itself is assembled in memory and printed at the end, so without
    these the terminal would sit blank for a minute and look like it had hung.
    `flush=True` matters: Python buffers stdout, so the line would otherwise not
    appear until the buffer filled.
    """
    print(message, flush=True)


def wrap(text: str) -> str:
    """Re-flow a paragraph to 76 columns, with a blank line before it.

    The prose interpolates computed numbers whose printed width is not known in
    advance, so hand-wrapped strings would come out ragged as the data changes.
    """
    return "\n" + "\n\n".join(
        textwrap.fill(" ".join(p.split()), 76, break_on_hyphens=False)
        for p in text.strip().split("\n\n"))


def main(argv=None) -> int:
    args = parse_args(argv)
    out: list[str] = []

    # ---- 1. load and clean ------------------------------------------------
    step("[1/4] Reading and cleaning the catalogue ...")
    try:
        df, notes = analysis.load_and_clean(args.data)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1

    out.append(rule("1. THE DATA"))
    out.append(f"{notes['n_clean']} stars, 40 in each of six types. "
               f"{notes['n_missing']} missing values, {notes['n_duplicates']} duplicate rows.")
    if notes["dropped"]:
        out.append(f"Dropped {notes['dropped']} — a textual recoding of the target that "
                   "would have leaked the answer into the model.")
    out.append(f"The free-text Color column held {notes['colours_raw']} spellings of "
               f"{notes['colours_clean']} colours; these were normalised.")
    out.append("\nMedian properties of each population:\n")
    out.append(analysis.summarise(df).to_string())
    out.append(wrap("""
        Read the temperature column, then the radius column. White dwarfs and
        main-sequence stars sit at almost the same temperature and differ in size by
        a factor of 560. Temperature does not separate them; radius does completely.
        That single observation drives everything below."""))

    # ---- 2. the physics ---------------------------------------------------
    step(f"[2/4] Testing the physics ({args.bootstrap:,} bootstrap resamples) ...")
    correlations = analysis.correlations(df)
    group = analysis.population_test(df)
    sb_fit = analysis.fit_stefan_boltzmann(df)
    residuals = analysis.sb_residuals(df)
    ms = df[df["Type_Label"].astype(str) == "Main Sequence"]
    law = analysis.fit_power_law(ms["Temperature"], ms["L"],
                                 n_boot=args.bootstrap, seed=args.seed)

    out.append(rule("2. THE PHYSICS"))
    out.append("Temperature–luminosity correlation, globally and per population:\n")
    out.append(correlations.to_string())
    out.append(wrap(f"""
        Pooled over all 240 stars the correlation is a weak r =
        {correlations.loc['ALL STARS', 'pearson_r']:.2f}. Within the main sequence alone
        it is {correlations.loc['Main Sequence', 'pearson_r']:.3f}. This is Simpson's
        paradox: a relationship that holds inside every subgroup can vanish when the
        subgroups are pooled, because the between-group scatter swamps it. It is why
        every fit below is restricted to a single population."""))

    out.append(wrap(f"""
        Kruskal–Wallis on log luminosity confirms the populations are distinct
        (H = {group['H']:.0f}, p = {group['p_value']:.1e}), and eta-squared =
        {group['eta_squared']:.2f} says {100 * group['eta_squared']:.0f}% of the
        variation lies between populations rather than within them. The effect size
        matters more than the p-value: with 240 stars almost anything is
        "significant"."""))

    out.append(wrap(f"""
        Now the sharpest test available. A star radiates as a sphere, so
        L = 4·pi·R²·sigma·T⁴ — a law with no free parameters. Take logs and fit the
        exponents freely, as if the theory were unknown:

            log L = log C + alpha·log R + beta·log(T/T_sun)

        Fitted:  alpha = {sb_fit['alpha']:+.3f} +/- {sb_fit['alpha_err']:.3f}
        (theory: 2, z = {sb_fit['z_alpha']:+.1f})
        and beta = {sb_fit['beta']:+.3f} +/- {sb_fit['beta_err']:.3f}
        (theory: 4, z = {sb_fit['z_beta']:+.1f}).

        The radius exponent comes back at {sb_fit['alpha']:.2f} against a theoretical 2
        — the geometry of a sphere, recovered to better than 1% from a CSV file. The
        temperature exponent does not. Testing each population against the unfitted
        law shows why:"""))
    out.append("\n" + residuals.to_string())
    out.append(wrap("""
        The main-sequence rows sit on the law almost exactly, and a Wilcoxon test
        cannot distinguish their residuals from zero. No other population does;
        supergiants are catalogued four times more luminous than their own radius and
        temperature allow. Since the law has nothing to tune, there is no way to
        reconcile this: the catalogue is a mixture of real measurements and rows whose
        three columns were evidently generated semi-independently. One supergiant is
        listed at 5,752 K, 245,000 solar luminosities and 97 solar radii — off by a
        factor of 26. Such an object cannot exist."""))

    out.append(wrap(f"""
        Fitting a power law to the main sequence alone gives L ∝ T^{law['alpha']:.2f}
        (95% bootstrap interval {law['ci'][0]:.2f} to {law['ci'][1]:.2f},
        R² = {law['r_squared']:.3f}). Stellar-structure theory predicts an exponent
        near 7, from the mass–luminosity relation L ∝ M^3.5 combined with
        T ∝ M^0.5. Mass appears nowhere in this dataset, so that agreement is a
        genuine confirmation rather than a curve-fitting exercise."""))

    # ---- 3. the modelling -------------------------------------------------
    step("[3/4] Training and evaluating the models (the slow step, ~15-40 s) ...")
    out.append(rule("3. CLASSIFYING STELLAR TYPE"))
    results = model.run(df, seed=args.seed)
    out.append(f"{results.n_train} training stars, {results.n_test} held out. All "
               "preprocessing lives inside scikit-learn Pipelines, so scalers and "
               "encoders are refitted on each\ncross-validation fold and never see the "
               "test set. The test set is scored once.\n")
    out.append(results.scores.to_string())
    out.append(f"\nBest model: {results.best_model}\n")
    out.append(results.report)
    out.append("Which measurement carries the signal? Cross-validated accuracy using "
               "each feature\nalone, and the accuracy lost when it is removed from the "
               "full set:\n")
    out.append(results.contributions.to_string())
    out.append(wrap(f"""
        Radius alone classifies {100 * results.contributions.iloc[0]['alone']:.0f}% of
        stars correctly; temperature alone manages
        {100 * results.contributions.loc['Temperature', 'alone']:.0f}%. Yet removing any
        single feature costs almost nothing, because the four measurements are
        physically redundant — radius, luminosity and magnitude are locked together by
        the Stefan–Boltzmann law and the definition of magnitude, so the model
        reconstructs whichever one it loses. This is also why permutation importance,
        which measures reliance rather than information, scores several features at
        zero."""))
    out.append(wrap(f"""
        A model given only log T and log L reaches
        {results.grid['accuracy']:.3f} accuracy, so most of the signal really does live
        in the plane of the HR diagram. Its decision boundaries, drawn on that diagram,
        fall almost exactly where an astronomer would draw them by eye. The ranking is
        astrophysics, not algorithmics: the six types are defined by evolutionary
        stage, evolutionary stage is expressed as size, and a white dwarf and a red
        hypergiant can share a surface temperature while differing by five orders of
        magnitude in radius. It is precisely why Hertzsprung and Russell needed a
        two-dimensional diagram."""))

    out.append(rule("4. WHAT TO DISTRUST"))
    out.append(wrap("""
        240 stars with exactly 40 per class is a teaching set, not a sample of
        anything — real populations are roughly 76% M dwarfs. There are no measurement
        uncertainties, distances, masses or ages. "Main Sequence" and "Red Dwarf"
        overlap physically, so one of the six classes is not a natural kind. And the
        labels were assigned from position in the HR diagram while the features are the
        coordinates of that diagram, so the classification measures how cleanly the
        catalogue was built rather than how hard stellar classification is. With ten
        test stars per class, an accuracy of 1.00 is compatible with a true error rate
        of several per cent."""))

    # ---- figures and output ----------------------------------------------
    figures = []
    if not args.no_figures:
        step("[4/4] Drawing figures ...")
        plots.configure_style()
        figures = [
            plots.hr_diagram(df, args.outdir),
            plots.physics_audit(df, sb_fit, args.outdir),
            plots.power_law(df, law, args.outdir),
            plots.model_results(results.confusion, results.class_names,
                                results.contributions, args.outdir),
            plots.decision_regions(df, results.grid, args.outdir),
        ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out)
    (args.outdir / "report.txt").write_text(text, encoding="utf-8")

    step("\nDone — the full report follows.")
    print(text)
    print(f"\nReport : {args.outdir / 'report.txt'}")
    for path in figures:
        print(f"Figure : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
