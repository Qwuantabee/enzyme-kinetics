# enzkin — enzyme kinetics and inhibition analysis

Fitting Michaelis-Menten and inhibition models to initial-rate data, with
honest uncertainty estimates and mechanism assignment by AICc rather than by
eye.

Milestone 1 (this commit) is the modelling and fitting core, validated against
simulated data where the true parameters are known. Real published datasets
come next.

## Why not just use a Lineweaver-Burk plot

That was the original question behind this, and the answer turned out to be
more interesting than the textbook version.

The standard argument is that inverting `v` inflates the error on the low-[S]
points and wrecks the regression. Running it over 500 simulated datasets
(8 substrate concentrations, 5% noise, true Km = 25, Vmax = 100):

**Proportional error** (noise scales with the rate):

| method | Km bias | Km SD | Vmax bias | Vmax SD |
|---|---|---|---|---|
| nonlinear fit | +0.8% | 10.3% | +0.3% | 4.3% |
| Lineweaver-Burk | +0.1% | 11.1% | -0.0% | 6.9% |
| Hanes-Woolf | +0.8% | 11.0% | +0.2% | 5.0% |
| Eadie-Hofstee | -1.1% | **6.2%** | -0.5% | 3.5% |

**Constant absolute error** (noise independent of the rate):

| method | Km bias | Km SD | Vmax bias | Vmax SD |
|---|---|---|---|---|
| nonlinear fit | +1.0% | 18.2% | +0.3% | 5.5% |
| Lineweaver-Burk | -318% | **6086%** | -177% | 3302% |
| Hanes-Woolf | +17.6% | 228% | +2.3% | 29.9% |
| Eadie-Hofstee | -23.8% | 25.2% | -12.3% | 10.3% |

So the textbook warning is correct, but only under constant absolute error.
Under proportional error the transforms are fine, and Eadie-Hofstee was
actually the most precise of the four — the reciprocal happens to stabilise
proportional variance.

Which error model applies is a property of the assay, not of the algebra. A
continuous spectrophotometric assay reading a small absorbance change is
closer to constant absolute error, which is the regime where Lineweaver-Burk
falls apart. A discontinuous assay built on a dilution series or on counting
is usually closer to proportional.

The nonlinear fit is the only method that is never the worst choice, which is
why it is the default here. `weights="relative"` switches to proportional-error
weighting when that is what the assay has.

![Km recovery under absolute error](figures/km_recovery_absolute.png)

## Confidence intervals

The first version of `ci95` used ±1.96·SE. Over 500 simulated datasets those
intervals contained the true Km **86.8%** of the time while claiming 95%.

The cause is sample size: a typical run is 8 points and 2 parameters, so 6
degrees of freedom, where the right critical value is t = 2.45, not 1.96.
Switching to the t distribution raised coverage to **91.0%** (proportional
error) and **93.2%** (absolute error).

The remaining gap is real and is not a bug. Km and Vmax are bounded below at
zero and their sampling distributions are right-skewed, so a symmetric interval
cannot be exactly right. When the standard error is a large fraction of the
estimate, use `bootstrap_ci` instead, which resamples residuals and makes no
symmetry assumption.

## How far up the substrate range you need to go

Median relative standard error over 200 simulated runs, varying the highest
substrate concentration:

| max [S] | Vmax SE | Km SE | % of Vmax reached |
|---|---|---|---|
| 1 × Km | 7.7% | 12.3% | 50% |
| 2 × Km | 4.6% | 9.2% | 67% |
| 5 × Km | 2.9% | 7.8% | 83% |
| 10 × Km | 2.4% | 7.8% | 91% |
| 20 × Km | 2.2% | 8.1% | 95% |

Most of the benefit is gone by 5 × Km. Km is barely improved past that point,
and Vmax gains about half a percent for four times the substrate. Useful when
substrate is expensive or poorly soluble and you are deciding where to stop.

## Mechanism identification

All inhibitor concentrations are fitted together in one global fit rather than
fitting each [I] separately and watching how apparent Km and Vmax move. Models
are ranked by AICc (small-sample corrected, because these datasets are small
enough that plain AIC under-penalises the 4-parameter mixed model).

Fraction of 100 simulated experiments where the true mechanism was recovered,
on a 7 × 4 substrate-by-inhibitor grid:

| true mechanism | 2% noise | 5% noise | 10% noise |
|---|---|---|---|
| competitive | 85% | 85% | 85% |
| uncompetitive | 99% | 99% | 94% |
| noncompetitive | 100% | 100% | 100% |
| mixed | 100% | 100% | 84% |

Competitive sits at 85% regardless of noise, which says the failure is
structural rather than statistical: a mixed inhibitor with Kis much larger than
Ki is nearly indistinguishable from a pure competitive one, and no amount of
precision on this design separates them. Pushing to higher [S] is what breaks
the tie, since that is where the ES-binding term shows up.

Mixed is counted as correct when the truth is noncompetitive, because
noncompetitive is the Ki = Kis special case of mixed and nests inside it.

![inhibition fit](figures/inhibition_fit.png)

## Worked example

```
model: michaelis_menten   n=8   s=2.767   R2=0.9929   AICc=20.38
    vmax = 98.0 +/- 2.91  (3.0%)
      km = 25.5 +/- 2.48  (9.7%)
  bootstrap 95% CI vmax: [93.2, 102.6]
  bootstrap 95% CI km:   [21.6, 29.9]
```
True values were Vmax = 100, Km = 25. Lineweaver-Burk on the same data gave
Vmax = 95.0, Km = 23.5.

Mixed inhibition example, true Ki = 10 and Kis = 40:

| model | AICc | ΔAICc |
|---|---|---|
| mixed | 2.05 | 0.00 |
| noncompetitive | 55.12 | 53.08 |
| competitive | 71.66 | 69.61 |
| uncompetitive | 94.47 | 92.42 |

Recovered Ki = 10.0 ± 0.7 and Kis = 45.1 ± 3.1.

A ΔAICc below about 2 is not evidence for one mechanism over another.
`compare_models` returns the full ranking rather than just the winner so that
this stays visible.

## Usage

```python
from enzkin import fit_michaelis_menten, compare_models

fit = fit_michaelis_menten(S, v)
print(fit.summary())
print(fit.ci95("km"))

results = compare_models(S, I, v)      # ranked best first by AICc
print(results[0].model, results[0].params)
```

## Layout

```
src/enzkin/
    models.py      rate equations (MM, four inhibition models, Hill,
                   substrate inhibition) and kcat/efficiency
    fitting.py     nonlinear least squares, standard errors, AICc, bootstrap
    progress.py    integrated rate equation, Lambert W forward model
    linearize.py   the three classic transforms, for comparison
    cheng_prusoff.py  IC50 <-> Ki, 4PL dose-response fitting, assumption checks
    pairing.py     joins measured IC50 to measured Ki, tests the ratio
                   Cheng-Prusoff predicts against the ratio observed
    structure.py   PDB parsing, binding-site and contact geometry
    qsar.py        RDKit 2D descriptors, LOOCV, permutation null -- the
                   structure-affinity test on congeneric series
    datasets.py    the published traces (Frere 2023) and the ChEMBL tables
    simulate.py    synthetic data with known parameters
    plotting.py    figures
scripts/
    validate_fits.py       the simulation study behind every number above
    prepare_frere2023.py   extract the experimental traces from the archive
    fit_published.py       initial rates vs progress curves on real data
    fetch_chembl_dhfr.py   pull the IC50/Ki tables with assay descriptions
    ic50_to_ki.py          Cheng-Prusoff applied to the real IC50 series
    validate_cheng_prusoff.py  the conversion tested against measured Ki
    fetch_chembl_smiles.py canonical SMILES for the Ki compounds (cached)
    qsar_congeneric.py     2D descriptors vs Ki, with a permutation null
tests/             149 tests: limiting cases on the rate laws, parameter
                   recovery, CI coverage, mechanism identification,
                   progress-curve recovery, Cheng-Prusoff limits and
                   round trips, IC50/Ki pairing on the real ChEMBL tables,
                   PDB geometry on real structures, descriptors against
                   textbook constants, Spearman/LOOCV against SciPy, and
                   the permutation null on data with and without signal
```

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/validate_fits.py
```

The descriptor analysis is the one part that needs RDKit, which is an
optional extra (`pip install -e ".[qsar]"`) because it is a large binary
wheel and every other module here runs without it. The two tests that need it
skip cleanly when it is absent, and `tests/test_qsar.py` is otherwise
independent of it: the statistics are tested against SciPy, which is already
a dependency.

## Still to do

Updated after the published-data pass, the IC50/Ki validation, and the
congeneric-series descriptor test. None of the original items is still open;
three of them are done only as far as the available data honestly allows, and
what is left in those is written down rather than quietly dropped.

- ~~Fit published datasets rather than only simulated ones~~ — **done.**
  Eighteen experimental traces of *Enterobacter cloacae* P99 class C
  β-lactamase hydrolysing nitrocefin, from Frère, Verlaine & Matagne (2023),
  *Sci Rep* 13:14994, doi:10.1038/s41598-023-41805-y, archive CC-BY on Zenodo
  (doi:10.5281/zenodo.10059024). Five substrate concentrations from 11.1 to
  64 µM, 0.6 nM enzyme, 462 nm. See below.

- ~~Progress-curve fitting, to use the whole time course instead of initial
  rates~~ — **done.** The integrated rate equation is fitted to all 2376
  points. On these traces it puts Km within 18% of the published value with a
  Km standard error about 2× smaller than the initial-rate fit (5.03 → 2.21 µM,
  trace-level cluster bootstrap). See below.

- ~~IC50 to Ki conversion via Cheng-Prusoff, with the assumptions made
  explicit~~ — **done**, and the caveat that stood here is now measured
  rather than asserted. The relations, the assumption checks and a 4PL IC50
  fitter are in `cheng_prusoff.py`.

- ~~Validating the conversion against measured Ki~~ — **done, and the answer
  is that it is partially supported, not validated.** This item used to say
  the check was impossible because ChEMBL records carry no assay [S] or [E].
  That is true and still true, but it was too pessimistic about what *can* be
  tested. The two files in this repository overlap: of 482 IC50 records and
  266 Ki records, **52 compounds appear in both**, and for **35 of them the
  IC50 and the Ki come from the same publication** (36 molecule-document
  pairs). `scripts/validate_cheng_prusoff.py` joins them and tests the ratio
  the theory predicts, because Cheng-Prusoff for a competitive inhibitor says
  IC50/Ki = 1 + [S]/Km — a property of the assay, not of the compound.

  Measured, from `results/cheng_prusoff_ratio_stats.csv`:

  | set | n | median IC50/Ki | IQR | log10 SD |
  |---|---|---|---|---|
  | same publication (strict) | 36 | 7.47 | 6.61–20.58 | 0.407 |
  | same molecule only (loose) | 52 | 6.89 | 3.99–20.25 | 0.534 |
  | strict, enzyme assays only | 29 | 14.74 | 6.78–21.08 | 0.410 |

  The median ratio of 7.5 implies the typical assay ran at [S] ≈ 6× Km — but
  that pooled number averages papers whose own medians are 3.0, 4.2, 6.8 and
  20.9, so it is a summary and not a conversion factor. What the data does
  support is the *structure*: within each paper the ratio is reproducible to
  **1.9-fold** (pooled within-paper log10 SD 0.283) while the papers differ
  from each other by more — **53% of the log-ratio variance is between
  papers**, which is what a per-assay substrate factor predicts. The largest
  paper (CHEMBL3632554, 14 compounds spanning 1150 to 1.55×10⁶ nM of potency)
  has a ratio-vs-potency slope of **−0.025**: the ratio does not track
  potency, so it is not a per-compound correction.

  What it does not support is the *values*. No record carries [S], so no pair
  can confirm the correction it implies; any ratio above 1 is consistent with
  some substrate concentration. Two pairs sit below the competitive model's
  floor of ratio 1 (CHEMBL82985 at 0.71, CHEMBL1162149 at 0.90) and are
  flagged rather than dropped. Seven further pairs pair a purified-enzyme
  IC50 with a whole-cell "antibacterial" Ki — a different quantity — and are
  labelled `non_enzyme` and reported separately, not deleted. Validation of
  the absolute conversion still needs a paper carrying [S] alongside IC50 and
  Ki; `pairing.assay_residuals` is the function that would run that check.

- ~~Link Ki values to binding-site structure from the PDB~~ — **done**, and
  deliberately limited. Measured Ki values are joined to crystal structures
  via the PDB's chemical-component ChEMBL cross-reference, and the pocket is
  described from coordinates.

- ~~Structure-affinity relationship~~ — **done, and the answer is both yes and
  no, depending on which series.** The claim that stood here (that one protein
  with two ligands cannot support a relationship, and that it needs a
  congeneric series or mutagenesis data) was right about the requirement and
  wrong about the data: the congeneric series was already in the Ki table.
  Three papers in `data/chembl_dhfr/ecoli_dhfr_ki.csv` are single-assay
  congeneric sets, and they give two different answers. See below.

- **Open: what the two working series do not establish.** The signal is real
  against a permutation null and it survives restriction to a single Murcko
  scaffold (66 of 73 and 52 of 72 compounds, LOO R² 0.312 and 0.653, both
  p_perm = 0.001), so it is not chemotype spread — see
  `results_single_scaffold.txt`. But it is largely captured by molecular
  weight alone. What would settle the mechanism is a free-energy calculation
  against the actual 4DFR/1DDR poses rather than 2D descriptors, or a series
  whose congeners differ in a substitution the descriptors can resolve rather
  than in size. Neither is in the
  data here, and inventing a pose would be worse than the descriptor result.
  The other gap is coverage at the tight end: the 1985 series spans
  0.021–16 nM and so has 12 compounds in the 1–100 nM band, but the two large
  series are dominated by weak binders (33 of 73 and 13 of 72 fall in
  1–100 nM, the rest weaker), so their R² is carried by compounds no
  medicinal chemist would pursue.

### Something the published data forced

Two errors that simulation could never have produced, both caught by reading
the real records rather than trusting them:

**The instrument's number format.** Later time values in these files are
written `1.008.333`, meaning 1.008333 min — a dot used as a thousands
separator on top of a decimal point. `float()` raises `ValueError` on that
string rather than misreading it, which is the louder and safer failure; the
subtler trap is the value one column over, `0.008333`, which parses cleanly
while the trace's later timestamps are what actually carry the thousands dot.
Whatever the handling, the resulting curve still fits a Michaelis-Menten
model beautifully — it just returns a Vmax wrong by orders of magnitude.
Nothing in the fit diagnostics would have flagged it, which is why the parser
strips the spurious dot explicitly and `tests/test_datasets.py` pins the
conversion.

**Mutants hiding in a Ki table.** Methotrexate appears five times in the
ChEMBL Ki records for *E. coli* DHFR with values of 0.001, 0.021, 330, 2400
and 170000 nM. That looks like a five-order-of-magnitude uncertainty on one
compound. It is not: three of the records are Phe31, Tyr31 and Val31 point
mutants, and the only thing that says so is the assay description. The loader
now filters on it, and the structure comparison uses wild-type records only.

## Fitting the published dataset

`scripts/fit_published.py`, `scripts/prepare_frere2023.py`. The archive is
136 MB so it lives outside git; the 18 extracted traces (84 KB) are committed
because the tests read them.

Two analyses of the same 18 traces:

| method | Vmax (µM/min) | Km (µM) | kcat (s⁻¹) | Km rel. SE |
|---|---|---|---|---|
| initial rates, 15% window | 34.45 ± 2.32 | 33.98 ± 5.03 | 957 | 14.8% |
| progress curves, all points | 32.71 ± 0.10 | 29.57 ± 2.21 | 909 | **7.5%** |
| published literature | — | 25 | 780 | — |

The progress-curve SE is a **trace-level cluster bootstrap** (400 resamples,
stratified by [S]₀), not the i.i.d. value the fit returns: 95% CI for Km is
[25.74, 33.44] µM. The i.i.d. number, 0.19 µM, is not an uncertainty at all.
It treats 2376 points from 18 traces as 2376 independent observations, and
the residuals are strongly autocorrelated within a trace — lag-1 r runs 0.94
to 1.00 across all 18, median 1.00. Resampling whole traces is the level at
which the data actually replicate, and it gives a Km SD about **12× larger**
than the i.i.d. figure. `progress.py` now reports both, prints the lag-1
diagnostics per trace, and `cluster_se()` is in `scripts/fit_published.py`.

Km lands within 18% of the published 25 µM and kcat within 17% of 780 s⁻¹.
Residual σ is 0.14 µM against a quantisation floor of 0.007 µM. That σ is
20× the floor, which is a statement about the *fit*: the single-substrate
model with no product inhibition is not being forced onto the traces over
36–65% conversion. It is **not** evidence about what the 0.14 µM consists of —
with lag-1 autocorrelation at 0.94–1.00 the residual is a smooth per-trace
curve, not instrument noise, so "σ genuinely describes these traces" is the
only defensible reading of it.

The real gain over the initial-rate fit is the **Km standard error, about 2×
(5.03 → 2.21 µM), not 25×**. The shape of each curve constrains Km separately
from Vmax and 18 traces beat 5 averaged rates, but the 25× figure came from
the i.i.d. SE and is wrong. Note also that the two Vmax values differ by 5%:
the initial-rate window *could* include substrate depletion, but this dataset
does not show it — the gap is 0.75 of one initial-rate SE, and sweeping the
window from 5% to 40% moves Vmax from 35.24 to 34.99 with no trend. The 5%
gap is not evidence of a depletion bias.

`kcat` is 1.16× the published value, so `activation_fraction()` reports the
preparation as 116% active rather than rescaling it away. A real preparation
is not the literature preparation.

## IC50 to Ki

`scripts/ic50_to_ki.py` on 482 measured IC50 values for *E. coli* DHFR from
ChEMBL (target CHEMBL1809, 354 compounds, 42 papers).

The module refuses the mixed case outright: one IC50 cannot yield two Ki
values. It also flags the three ways the conversion silently fails:

| [S]/Km | correction | what goes wrong |
|---|---|---|
| 0.1 | 1.1× | Km error barely matters |
| 2 | 3.0× | Km error carried 3× |
| 10 | 11× | a 30% Km error moves Ki 25% |
| 50 | 51× | Km must be measured, not borrowed |

And the tight-binding bound, which is the one that reorders inhibitor series:
at an assumed 10 nM enzyme, **39% of this real compound series has an IC50
within 10× of [E]** and cannot be converted by Cheng-Prusoff at all.

### Is the conversion right? Measured against real Ki values

`scripts/validate_cheng_prusoff.py` answers the question the paragraph above
used to defer, and the answer is **partially supported**. 52 compounds appear
in both the IC50 and the Ki tables; for 35 of them (36 molecule-document
pairs) the two measurements come from the *same publication*, which holds the
lab and the protocol constant.

| set | n | median IC50/Ki | IQR | log10 SD |
|---|---|---|---|---|
| same publication (strict) | 36 | 7.47 | 6.61–20.58 | 0.407 |
| same molecule only (loose) | 52 | 6.89 | 3.99–20.25 | 0.534 |
| strict, enzyme assays only | 29 | 14.74 | 6.78–21.08 | 0.410 |

What supports the relation:

* **Within a paper the ratio is stable.** Pooled within-paper log10 SD is
  0.283, a 1.9-fold spread. The 14 compounds of CHEMBL3632554 span 1150 to
  1.55×10⁶ nM in potency at a ratio of 20.95 with a ratio-vs-potency slope of
  −0.025. The correction is not tracking potency.
* **Between papers it moves, and that is the prediction.** 53% of the
  log-ratio variance is between papers. Per-paper median ratios are 3.0
  (CHEMBL5233242), 4.2 (CHEMBL1124568), 6.8 (CHEMBL1136127) and 20.9
  (CHEMBL3632554) — four assays, four implied [S]/Km values of about 2, 3, 6
  and 20.

What it does not establish, and why the verdict stops at "partially":

* **No record carries [S].** Every ratio above 1 is consistent with some
  substrate concentration, so internal consistency is necessary and not
  sufficient. The absolute conversion is still untested compound by compound.
* **Two pairs sit below the competitive floor** of ratio 1 (CHEMBL82985 at
  0.71, CHEMBL1162149 at 0.90). The model forbids that; the rows are flagged,
  not dropped.
* **Seven pairs are not enzyme assays at all.** One paper reports whole-cell
  antibacterial potency as its "Ki". They are labelled `non_enzyme` and
  reported separately rather than deleted, because deleting them would
  improve the median and destroy the result's meaning.

Feeding [S] and the assay Km to `pairing.assay_residuals` is the check that
would settle it, and it needs a paper that reports both.

## Ki to binding site

`scripts/ki_to_structure.py` on real coordinates of *E. coli* DHFR
(1DDR, 3DAU, 4DFR), linked to measured Ki by the PDB chemical-component
ChEMBL cross-reference.

Methotrexate in the pocket: 18 residues within 4.5 Å per subunit (Phe31 and
Asp27 among them), 91–100% of ligand atoms in van der Waals contact with
protein, and contacts including 14 H-bonds and 9 salt bridges.

Two things this is not. It is not a Shrake-Rupley SASA calculation — burial is
a contact-count fraction. And it is not a structure-affinity relationship:
with one protein and two ligands, any correlation would be anecdotal. That
second gap is now addressed from a different direction, with 2D descriptors
rather than coordinates — see the next section. The chain-selection hazard is
real too: 4DFR contains both an *E. coli* and a *Lactobacillus casei* copy,
and picking the wrong one would compare an *E. coli* affinity to the wrong
protein while looking entirely normal.

## Structure-affinity, on the congeneric series that was already here

`scripts/qsar_congeneric.py`, `src/enzkin/qsar.py`. Ten RDKit 2D descriptors
(MW, logP, TPSA, HBD, HBA, rotatable bonds, rings, aromatic rings, fraction
Csp3, molar refractivity) against log10 Ki, with leave-one-out
cross-validation and a **1000-shuffle permutation null**. RDKit 2026.09.2.
SMILES come from ChEMBL's molecule endpoint via `scripts/fetch_chembl_smiles.py`
and are committed (20 KB) so the analysis runs offline.

The permutation null is the whole point, and it is not a formality. With 21
compounds and 11 fitted parameters, noise routinely produces the kind of R²
that looks like a finding. So each shuffle refits the *entire* pipeline —
standardisation, folds, least squares — on a scrambled Ki vector, and the
question is how often chance reaches the observed score.

| paper | n | des/param | Ki span | LOO R² | best single descriptor | p_perm |
|---|---|---|---|---|---|---|
| CHEMBL1123078 (J Med Chem 1985) | 21 | 1.9 | 0.021–16 nM (762×) | **−1.021** | HBD, 0.182 | 0.678 |
| CHEMBL1129627 (J Med Chem 1996) | 73 | 6.6 | 1.35–912000 nM | **0.326** | MW, 0.222 | ≤0.001 |
| CHEMBL1124156 (J Med Chem 1988) | 72 | 6.6 | 0.012–912000 nM | **0.624** | MW, 0.458 | ≤0.001 |

(`p_perm` is floored at 1/(n_perm+1) = 0.001; on both larger series no shuffle
reached the observed score, so the report says that rather than quoting the
arithmetic.)

**The tight series fails, and that is the informative result.** The
21-compound set — the most congeneric one, 20 of 21 compounds sharing a single
Murcko scaffold — has an LOO R² of −1.021, i.e. worse than predicting the
mean, with p_perm 0.678. Its best univariate descriptor does not reach
significance either (|rho| ≤ 0.37, p ≥ 0.10). With 1.9 observations per
parameter, this series **is too small to support a descriptor model**, and the
fit is not reported as if it were one.

The two ~72-compound series do show signal, and the honesty checks say it
survives them rather than evaporating:

- **It is not tail-driven.** Dropping every compound with Ki ≥ 100 µM (1 and 2
  compounds respectively) leaves LOO R² = 0.582 and 0.736 with p_perm = 0.005
  on 200 shuffles. The correlations get *stronger*, not weaker, so they are not
  an artefact of a few barely-binding outliers.
- **The permutation null is genuinely against us.** On the 73-compound set the
  shuffled R² distribution has mean −0.201, 95th percentile −0.052 and maximum
  0.098 against an observed 0.326; on the 72-compound set, mean −0.215, 95th
  percentile −0.040, maximum 0.146 against 0.624.
- **But a single descriptor does most of the work.** MW alone reaches 0.222 and
  0.458, and the ten-descriptor model adds only +0.104 and +0.166 over it. The
  sign of the correlation is against log10 Ki (nM), so a *negative* rho means
  the descriptor raises affinity: bigger, more polar, more flexible compounds
  bind *better* here. In CHEMBL1129627 rho(MW, log10 Ki) = −0.532,
  MolRefract −0.436, HBA −0.352, TPSA −0.257, logP −0.207, RotB −0.118; the
  one exception is HBD at +0.064 (p = 0.59), so the direction is not uniform
  across descriptors, and CHEMBL1124156 repeats the pattern more strongly
  (MW −0.507). A larger, more lipophilic ligand binding tighter is consistent
  with a growing hydrophobic contact area rather than a size penalty, but 2D
  descriptors cannot tell the two apart, so no individual coefficient should be
  read as a design rule. Source: `results_qsar.txt` (univariate Spearman
  blocks, both series).

So the plain answer, which the earlier README said could not be given: **a
structure-affinity relationship is present in the 72–73 compound series and
absent — indistinguishable from chance — in the 21-compound one.** The
binding constraint is sample size, exactly as claimed, but the threshold sits
near 70 congeners, not at "impossible".

Caveats worth stating rather than burying. The 1988 and 1996 papers are
QSAR-era datasets measured by different groups on different days, and 2D
descriptors cannot see the pose. CHEMBL1124156 (the 1988 set) contains 9
Murcko scaffolds, so "congeneric" is generous there — but that confound was
tested and did not explain the result: restricted to its largest
scaffold (52 of 72 compounds) the LOO R² is 0.653 against 0.624 for the whole
set, p_perm = 0.001. What that restriction does show is that the fit is
molecular weight and essentially only molecular weight (MW alone reaches
0.644, a +0.009 gain for the other nine descriptors), so the honest reading of
that series is a molecular-weight (size/hydrophobicity) effect inside one
chemotype, not a complementarity model. Note again the sign: MW correlates
negatively with log10 Ki, i.e. the heavier compounds in this series bind
tighter, so this is not a penalty for being large. The 1996 set is the cleaner test at 66 of 73 on one scaffold, where the
ten descriptors still add +0.109 over MW alone. Neither result is a prediction
about a new compound; both are descriptions of variation inside a series that
was already measured.

Outputs: `results_qsar.txt`, `results/qsar_series.csv`,
`results_single_scaffold.txt` (the within-scaffold re-run).


## Note on how this was built

Written with substantial help from Claude (Anthropic) as a coding assistant.
The scientific questions, the validation design, and the interpretation of the
results are mine. The two findings that the simulation study turned up — that
the Lineweaver-Burk critique depends on the assay's error model, and that the
normal-approximation confidence intervals were overconfident at 87% coverage —
both came out of deliberately testing the code against known answers rather
than trusting that it worked.
