"""Structure-affinity analysis of a congeneric series.

The question this module exists to answer is narrow: given a set of
congeners -- one scaffold, one protein, one paper, one assay -- whose Ki
values span two to six orders of magnitude, can 2D structure descriptors
recover any of that variation?

The honest difficulty is sample size. A series of 21 compounds tested against
10 descriptors is 21 observations for 11 fitted parameters, and a
leave-one-out R2 computed on that is optimistic even when the descriptors
carry no information. So the cross-validated R2 is not reported on its own:
`permutation_null` shuffles the affinities, refits the whole model including
the cross-validation, and reports how often shuffled data reaches the real
score. If that fraction is not small, the model has not been shown to
generalise and must not be described as if it had.

Descriptors are RDKit 2D descriptors -- no conformer generation, no 3D, no
force field. That is deliberate: with no co-crystal for most of these
compounds, a 3D descriptor would be describing an invented pose.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised by the import guard test
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem import rdMolDescriptors as _rdmd
except ImportError:  # pragma: no cover
    Chem = None
    RDLogger = None


#: Descriptor name -> callable on an RDKit Mol. Order is the column order of
#: every matrix this module builds, so it is fixed once and referenced by
#: name elsewhere.
DESCRIPTORS = {
    "MW": lambda m: Descriptors.MolWt(m),
    "logP": lambda m: Crippen.MolLogP(m),
    "TPSA": lambda m: Descriptors.TPSA(m),
    "HBD": lambda m: Lipinski.NumHDonors(m),
    "HBA": lambda m: Lipinski.NumHAcceptors(m),
    "RotB": lambda m: Lipinski.NumRotatableBonds(m),
    "Rings": lambda m: Lipinski.RingCount(m),
    "AromaticRings": lambda m: Lipinski.NumAromaticRings(m),
    "FractionCSP3": lambda m: Lipinski.FractionCSP3(m),
    "MolRefract": lambda m: Crippen.MolMR(m),
}

DESCRIPTOR_NAMES = tuple(DESCRIPTORS)


def _rdkit():
    if Chem is None:
        raise ImportError(
            "rdkit is required for descriptor analysis; install with "
            "`pip install rdkit` (2026.09.2 or newer is what this repo used)"
        )
    return Chem


def parse_smiles(smiles):
    """RDKit Mol for a SMILES string, or None if it does not parse."""
    chem = _rdkit()
    if not smiles:
        return None
    return chem.MolFromSmiles(smiles)


def compute_descriptors(smiles):
    """The ten 2D descriptors for one SMILES, as a dict.

    Raises ValueError if the SMILES does not parse, rather than returning a
    row of NaNs: a compound silently dropped from the matrix changes n, and n
    is the whole argument here.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    return {name: float(fn(mol)) for name, fn in DESCRIPTORS.items()}


def descriptor_table(smiles_map):
    """[(chembl_id, smiles, {descriptor: value}), ...], in input order.

    Compounds whose SMILES is missing or unparseable are dropped and named in
    the return value's companion `skipped` list, so a caller can report how
    many were lost instead of quietly fitting on fewer compounds.
    """
    rows, skipped = [], []
    for cid, smi in smiles_map.items():
        try:
            rows.append((cid, smi, compute_descriptors(smi)))
        except ValueError:
            skipped.append(cid)
    return rows, skipped


def descriptor_matrix(rows):
    """(X, names) with X of shape (n_compounds, n_descriptors)."""
    names = list(DESCRIPTOR_NAMES)
    X = np.array([[r[2][name] for name in names] for r in rows], dtype=float)
    return X, names


def spearman(x, y):
    """Spearman rank correlation and its two-sided p, without scipy.stats.

    Ties are handled by ranking with average ranks, which is what
    SciPy does; the p value uses the t approximation with n-2 degrees of
    freedom, exact enough for n ~ 21 and stated so it can be checked.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        raise ValueError("spearman needs equal-length arrays of size >= 3")
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    if denom == 0:
        return 0.0, 1.0
    rho = float((rx @ ry) / denom)
    n = x.size
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho * rho))
    return rho, float(_t_two_sided_p(t, n - 2))


def _average_ranks(v):
    """Ranks with ties averaged (the standard correction, not ordinal)."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    sorted_v = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _t_two_sided_p(t, df):
    """Two-sided p for Student's t, via the regularised incomplete beta."""
    x = df / (df + t * t)
    return _betainc_reg(0.5 * df, 0.5, x)


def _betainc_reg(a, b, x):
    """Regularised incomplete beta I_x(a, b) by continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
    front = np.exp(lbeta + a * np.log(x) + b * np.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a, b, x, itmax=200, eps=3e-14):
    """Continued fraction for the incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _log_gamma(z):
    """Lanczos approximation to ln Gamma(z) for z > 0."""
    g = 7.0
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    if z < 0.5:
        return (np.log(np.pi / np.sin(np.pi * z))
                - _log_gamma(1.0 - z))
    z -= 1.0
    a = c[0]
    t = z + g + 0.5
    for i in range(1, 10):
        a += c[i] / (z + i)
    return 0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(a)


def standardise(X):
    """Column-centre and scale; columns with zero variance become zeros.

    Zero-variance columns are kept rather than dropped so that descriptor
    count and column order stay stable across series -- a descriptor that
    happens to be constant here (FractionCSP3 across one scaffold, say) is
    information about the series, not something to hide.
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd_safe


def _standardise_train_test(Xtr, Xte):
    """Standardise Xte with the training statistics, never its own."""
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def loocv_predictions(X, y, ridge=False, alpha=1.0):
    """Leave-one-out predictions for every row of X.

    Each fold refits from scratch and standardises the held-out row with the
    training fold's mean and standard deviation. That detail matters: a
    'cross-validation' that standardises once, globally, has already seen the
    held-out point.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    pred = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xtr, Xte = _standardise_train_test(X[mask], X[i:i + 1])
        A = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
        if ridge and alpha > 0:
            AtA = A.T @ A
            pen = np.eye(AtA.shape[0]) * alpha
            pen[0, 0] = 0.0
            coef = np.linalg.solve(AtA + pen, A.T @ y[mask])
        else:
            coef = np.linalg.lstsq(A, y[mask], rcond=None)[0]
        pred[i] = np.hstack([1.0, Xte[0]]) @ coef
    return pred


def r2_score(y_true, y_pred):
    """Coefficient of determination against the mean of y_true."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def loocv_r2(X, y, ridge=False, alpha=1.0):
    """Leave-one-out cross-validated R2 (can be negative; report as-is).

    A negative value means the model is worse than predicting the mean, which
    on a small descriptor set is a common and perfectly informative outcome.
    """
    return r2_score(y, loocv_predictions(X, y, ridge=ridge, alpha=alpha))


def permutation_null(X, y, observed_r2, n_perm=1000, seed=0, ridge=False,
                     alpha=1.0):
    """Shuffle y `n_perm` times, refit, and see how often chance matches.

    Returns a dict with the null distribution, the permutation p value
    `(1 + #{null >= observed}) / (n_perm + 1)`, and the null's mean and 95th
    percentile. The whole pipeline -- standardisation, folds, refit -- is
    re-run on each shuffle, so what is being compared is models, not
    numbers.

    This is the test that decides whether a cross-validated R2 in this
    setting means anything. With n = 21 and ten descriptors, a shuffled null
    that reaches the real score 20% of the time is not a subtlety about
    variance: it says the score is what chance produces.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        null[k] = loocv_r2(X, rng.permutation(y), ridge=ridge, alpha=alpha)
    exceed = int((null >= observed_r2).sum())
    return {
        "null_r2": null,
        "n_perm": int(n_perm),
        "p_perm": (1.0 + exceed) / (n_perm + 1.0),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "null_max": float(null.max()),
        "seed": int(seed),
    }


def univariate_scan(X, names, y):
    """Spearman rho and p for each descriptor against y, sorted by |rho|."""
    out = []
    for j, name in enumerate(names):
        rho, p = spearman(X[:, j], y)
        out.append({"descriptor": name, "rho": rho, "p_value": p})
    out.sort(key=lambda d: -abs(d["rho"]))
    return out


def collinearity(X, names):
    """Largest absolute pairwise descriptor correlation, and between whom.

    Reported because a multivariable fit on ten descriptors that are really
    three is more overfit than it looks, and a reader deserves to see that
    before trusting a coefficient.
    """
    X = np.asarray(X, dtype=float)
    best = {"pair": None, "abs_r": 0.0}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho, _ = spearman(X[:, i], X[:, j])
            if abs(rho) > best["abs_r"]:
                best = {"pair": (names[i], names[j]), "abs_r": float(abs(rho)),
                        "rho": float(rho)}
    return best
