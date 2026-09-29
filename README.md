# Stellar Data Explorer

Most projects built on the Kaggle star dataset train a classifier and report the accuracy. This one asks a better question first: **do the numbers in the file describe objects that could actually exist?**

They mostly don't — and that turns out to be more interesting than the 100% classification accuracy.

![Hertzsprung–Russell diagram](outputs/1_hr_diagram.png)

## Run it

```bash
pip install -r requirements.txt
python main.py
```

One minute. Writes `outputs/report.txt` and five figures. The dataset ships with the project.

`--no-figures` skips plotting (a few seconds), `--seed N` changes the random split, `--data PATH` points at a different catalogue.

## What it found

**The catalogue fails a physics check.** A star radiates as a sphere, so *L* = 4π*R*²σ*T*⁴ — a law with no free parameters. Fitting the exponents freely, as if the theory were unknown, returns **α = 2.02 ± 0.04** against a theoretical 2: the geometry of a sphere, recovered to better than 1% from a CSV file. But testing each population against the unfitted law splits them apart:

| | Brown D. | Red D. | White D. | **Main Seq.** | Supergiant | Hypergiant |
|---|---|---|---|---|---|---|
| median *L*<sub>obs</sub>/*L*<sub>predicted</sub> | 0.54 | 0.30 | 0.21 | **0.99** | 4.04 | 0.48 |
| consistent with theory? | no | no | no | **yes** | no | no |

Main-sequence rows sit on the law almost exactly. Supergiants are catalogued four times more luminous than their own radius and temperature allow. One is listed at 5,752 K, 245,000 *L*<sub>☉</sub> and 97 *R*<sub>☉</sub> — off by a factor of 26. Such an object cannot exist, so parts of the dataset were evidently generated rather than measured.

**Simpson's paradox, live.** Pooled over all 240 stars the temperature–luminosity correlation is a weak *r* = 0.43. Within the main sequence alone it is **0.985**. Pooling six populations that obey different physics destroys the structure that makes the HR diagram useful.

**A theory prediction confirmed.** Fitted to the main sequence, *L* ∝ *T*<sup>6.62 ± 0.19</sup> (95% bootstrap interval 6.31–6.91, *R*² = 0.971). Stellar-structure theory predicts ≈ 7, from *L* ∝ *M*<sup>3.5</sup> combined with *T* ∝ *M*<sup>0.5</sup>. Mass appears nowhere in the dataset.

**Radius is the signal, not temperature.** All three classifiers hit 100% on the held-out set, which says more about the dataset than the models. The useful result is *which* measurement carries the information:

| | Radius | Abs. magnitude | Luminosity | Colour | Spectral class | Temperature |
|---|---|---|---|---|---|---|
| accuracy using it alone | **0.92** | 0.84 | 0.58 | 0.47 | 0.46 | **0.38** |

That ranking is astrophysics, not algorithmics. The six types are defined by evolutionary stage, evolutionary stage is expressed as *size*, and a white dwarf and a red hypergiant can share a surface temperature while differing by five orders of magnitude in radius. It is exactly why Hertzsprung and Russell needed a two-dimensional diagram.

## How it's built

```text
stellar-explorer/
├── data/stars.csv
├── src/
│   ├── analysis.py     # load, clean, derived physics, SciPy statistics
│   ├── plots.py        # the five figures
│   └── model.py        # the scikit-learn workflow
├── main.py             # runs the three stages, writes the report
├── requirements.txt
└── outputs/            # report.txt + five PNGs
```

**Cleaning** resolves the several header spellings this dataset circulates under, and drops the `Star category` column some copies ship — a textual recoding of the target that would leak the answer straight into the model. The free-text `Color` column holds 19 spellings of 8 colours; these are normalised to a key first, then looked up, so one table entry covers every spelling. Rows that look physically odd are **flagged, not deleted**: with 240 stars, dropping a fifth of them would bias every later result, and the anomalies are the most interesting thing here.

**Statistics** are Pearson and Spearman correlations (computed on logarithms, where a straight line is a power law), Kruskal–Wallis with eta-squared for the population comparison, `curve_fit` for the Stefan–Boltzmann plane and the power law, a Wilcoxon test per population, and a 2,000-sample bootstrap for the confidence band.

**Modelling** puts all preprocessing inside scikit-learn `Pipeline`s, so scalers and encoders are refitted on every cross-validation fold and never see the test set. The test set is scored once, at the end. Linear and tree models get different preprocessing — trees split on thresholds, so scaling and log transforms change nothing for them, while categoricals are ordinal-encoded in physical order (O→M, blue→red) so a single split can separate "hotter than F" from "cooler".

## Libraries

**NumPy** for the log transforms, the Stefan–Boltzmann calculation and the bootstrap. **pandas** for loading, cleaning, ordered categoricals and grouped statistics. **SciPy** — `stats` for the correlations and hypothesis tests, `optimize.curve_fit` for both fits. **scikit-learn** for the pipelines, three classifiers, cross-validation and metrics. **Matplotlib** and **seaborn** for the figures.

## Limitations

240 stars with exactly 40 per class is a teaching set, not a sample — real populations are roughly 76% M dwarfs. No measurement uncertainties, distances, masses or ages. "Main Sequence" and "Red Dwarf" overlap physically, so one of the six classes is not a natural kind. And the labels were assigned from position in the HR diagram while the features are the coordinates of that diagram, so the classification measures how cleanly the catalogue was built rather than how hard stellar classification is. With ten test stars per class, an accuracy of 1.00 is compatible with a true error rate of several per cent.

The obvious next step is a real catalogue — Gaia DR3 has parallaxes and photometry for over a billion stars, with genuine uncertainties and genuine class imbalance.

---

Dataset: *Star Type Classification / NASA*, as distributed on Kaggle. Physical constants follow IAU 2015 nominal solar values.
