# What each planned item found, and what is still open

Moved here from README.md on 2026-11-17 to keep the README readable; the text is
unchanged apart from this note, the heading level and the pointer to the table.
"See below" refers to the sections of README.md. Every number here was in the README
before the move and comes from the committed artefacts it cites.

Updated after the published-data pass, the IC50/Ki validation, and the
congeneric-series descriptor test. None of the original items is still open;
three of them are done only as far as the available data honestly allows, and
what is left in those is written down rather than quietly dropped.

<details>
<summary>The six completed items and what each found (click to expand)</summary>

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
  is that the *form* is partially supported and the *values* are untested.**
  This item used to say the check was impossible because ChEMBL records carry
  no assay [S] or [E] as fields. That is true for most assays, but it was too
  pessimistic about what *can* be tested. The two files in this repository
  overlap: of 482 IC50 records and 266 Ki records (257 wild-type), **52
  compounds appear in both**, and for **35 of them the IC50 and the Ki come
  from the same publication**. `scripts/validate_cheng_prusoff.py` joins them
  and tests the ratio the theory predicts, because Cheng-Prusoff for a
  competitive inhibitor says IC50/Ki = 1 + [S]/Km — a property of the assay,
  not of the compound. Every number below comes from
  `results_cheng_prusoff.txt` and `results/cheng_prusoff_ratio_stats.csv`.

  One publication (CHEMBL1136127) ran IC50 at two stated substrate
  concentrations, 30 and 100 µM (written in its assay descriptions). Pooling
  those IC50 records gives a ratio that belongs to neither assay, so the join
  now keeps them apart: **36 pooled pairs become 47**, 22 of them (11
  molecules) from that publication.

  The table of the four pair sets and the bootstrap interval are in the README,
  section "Is the conversion right? Measured against real Ki values", where a
  test ties them to `results/cheng_prusoff_ratio_stats.csv`.

  The median ratio of 9.4 implies a typical [S]/Km of 8.4 — but that pooled
  number averages assay conditions whose own medians are 3.0, 4.2, 4.6, 9.7
  and 20.9, so it is a summary and not a conversion factor. What the data can
  support is the *structure*: inside each assay condition the ratio is
  reproducible to a pooled within-group log10 SD of 0.288 (1.9-fold; the
  pooled figure hides the per-group max/min spreads of 2.0× for CHEMBL3632554,
  11.5× and 11.1× for the two [S] groups of CHEMBL1136127, 6.3× for
  CHEMBL1124568 and 26.9× for the 3-pair CHEMBL5233242), and the
  groups differ from each other (one-way ANOVA, F(4,41) = 12.86,
  p = 7.2×10⁻⁷; between groups is 51% of the log-ratio variance as an adjusted share, 1 − within/total variance, which `pairing.py` reports; the unadjusted sum-of-squares share, η², is larger). That is the
  pattern a per-assay substrate factor predicts, and any per-assay systematic
  difference would predict it too, so it does not single out Cheng-Prusoff.
  The label `partially supported` is decided by `pairing.verdict` from fixed
  thresholds; it is sensitive to the 0.30 limit on the within-group SD, and
  with a limit of 0.25 the same data would be labelled `unsupported`. The
  largest paper (CHEMBL3632554, 14 compounds spanning 1150 to 1.55×10⁶ nM of
  IC50) has a ratio-vs-potency slope of −0.025 with a 95% interval of
  −0.070 to 0.020. All four groups with at least five pairs have slope
  intervals that include 0, which is the absence of a detected trend, not
  evidence of none: the ratio is the difference of two log potencies, so
  independent noise biases the IC50 slope upward and the Ki slope downward,
  and the groups hold 7–14 points. The loose join pairs across papers and has
  no within-paper structure to test; its verdict is `not testable`.

  The two-[S] publication allows a direct (1 + [S]/Km) check that needs no
  Ki. For 11 compounds the IC50 at 100 µM over the IC50 at 30 µM has median
  2.18 (range 1.45–2.77); Cheng-Prusoff bounds it between 1 and 3.33. The
  single-condition IC50/Ki ratios imply Km medians of 8.1 µM (30 µM
  condition) and 11.4 µM (100 µM condition), which would predict a ratio
  of 2.84 or 2.69 against the 2.18 measured; the Km from the IC50 pair alone
  has median 29.2 µM and is noisy. These are measured medians without
  intervals at n = 11.

  What it does not support is the *values*. For most assays no [S] is stated,
  so no pair can confirm the correction it implies; any ratio above 1 is
  consistent with some substrate concentration. Two pairs sit below the
  competitive model's floor of ratio 1 (CHEMBL82985 at 0.48, CHEMBL1162149 at
  0.90) and are kept, not dropped. Seven pairs use a Ki whose ChEMBL assay
  description says "Antibacterial activity against Escherichia coli". That
  label is **unverified**: its Ki values (0.4–1.7 nM) are of the size of enzyme
  Ki, and trimethoprim's value there (1.3 nM) is within 5% of the value in 4 of
  the other 6 documents. They are flagged `antibacterial_described`, kept in the
  headline number, and reported as a sensitivity analysis. 29 of the 482 IC50
  records are above 10⁶ nM, a range ChEMBL itself flags (all report
  `standard_units` nM, so a units filter alone cannot catch them); they are
  flagged `outside_typical_range`, one of them is in a strict pair, and dropping
  it moves n from 47 to 46 and the median from 9.38 to 9.31. Validation of the
  absolute conversion still needs a paper carrying [S] alongside IC50 and Ki;
  `pairing.assay_residuals` is the function that would run that check.

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

</details>

- **Open: what the two working series do not establish.** The signal is real
  against a permutation null and it survives restriction to a single Murcko
  scaffold (66 of 73 and 52 of 72 compounds, LOO R² 0.312 and 0.653, both
  p_perm = 0.001; `results_single_scaffold.txt`), which is evidence against
  its being pure chemotype spread, not a proof: within the scaffold the two
  ring-count descriptors are constant, so the fit uses eight descriptors, and
  a negative result there would not have shown the cause either. It is largely
  captured by molecular weight alone, and it is not independent of potency
  range: see the tail sweep below. What would settle the mechanism is a free-energy calculation
  against the actual 4DFR/1DDR poses rather than 2D descriptors, or a series
  whose congeners differ in a substitution the descriptors can resolve rather
  than in size. Neither is in the
  data here, and inventing a pose would be worse than the descriptor result.
  The other gap is coverage in the drug-relevant band. By Ki band
  (`results_qsar_tail.txt`): the 1996 series has 0 compounds below 1 nM, 33
  in 1–100 nM, 39 in 100 nM–100 µM and 1 at or above 100 µM; the 1988 series
  has 12 below 1 nM, 13 in 1–100 nM, 45 in 100 nM–100 µM and 2 at or above
  100 µM. So "the rest are weaker" is true of the 1996 series and false of the
  1988 series, where 12 compounds bind tighter than 1 nM. The 1985 series
  (0.021–16 nM) has 9 compounds below 1 nM and 12 in 1–100 nM.

