"""Review fixes for the QSAR layer: cross-validation leakage and silent skips.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from enzkin import datasets, qsar

ROOT = Path(__file__).resolve().parents[1]


# --- C11: pin the training-only standardisation where it can matter ----------
# Ordinary least squares predictions are invariant to column scaling, so a
# test that uses plain OLS (the old one) cannot see whether the held-out row
# leaked into the standardisation. Ridge is not scale-invariant, and the
# standardisation helper can be pinned directly.

def _reference_ridge_loo(X, y, alpha, standardise):
    """Independent ridge LOO. standardise(Xtr, Xte) -> (Ztr, Zte)."""
    n = X.shape[0]
    out = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        Ztr, Zte = standardise(X[m], X[i:i + 1])
        A = np.hstack([np.ones((Ztr.shape[0], 1)), Ztr])
        pen = np.eye(A.shape[1]) * alpha
        pen[0, 0] = 0.0
        coef = np.linalg.solve(A.T @ A + pen, A.T @ y[m])
        out[i] = np.hstack([1.0, Zte[0]]) @ coef
    return out


def _train_only(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _leaky_global(X_all):
    mu, sd = X_all.mean(0), X_all.std(0)

    def f(Xtr, Xte):
        return (Xtr - mu) / sd, (Xte - mu) / sd
    return f


def _data():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(14, 3)) * np.array([1.0, 30.0, 0.2])
    X[0, 1] += 400.0                       # one extreme row
    y = 0.4 * X[:, 0] + 0.01 * X[:, 1] + rng.normal(0, 0.5, 14)
    return X, y


def test_standardise_train_test_uses_training_statistics_only():
    Xtr = np.array([[0.0], [2.0], [4.0]])
    Xte = np.array([[100.0]])
    Ztr, Zte = qsar._standardise_train_test(Xtr, Xte)
    mu, sd = 2.0, np.sqrt(8.0 / 3.0)
    assert Ztr[:, 0] == pytest.approx((Xtr[:, 0] - mu) / sd)
    assert Zte[0, 0] == pytest.approx((100.0 - mu) / sd)
    # the held-out value must not have moved the statistics: standardising
    # with train+test would give a different number
    both = np.vstack([Xtr, Xte])
    leaky = (100.0 - both.mean()) / both.std()
    assert abs(Zte[0, 0] - leaky) > 0.5


def test_loocv_ridge_predictions_equal_the_training_only_reference():
    X, y = _data()
    got = qsar.loocv_predictions(X, y, ridge=True, alpha=3.0)
    want = _reference_ridge_loo(X, y, 3.0, _train_only)
    assert got == pytest.approx(want, abs=1e-10)


def test_the_reference_discriminates_leaky_from_clean_standardisation():
    """Guard the guard: with ridge, a global (leaky) standardisation gives
    visibly different predictions, so the equality test above can fail."""
    X, y = _data()
    clean = _reference_ridge_loo(X, y, 3.0, _train_only)
    leaky = _reference_ridge_loo(X, y, 3.0, _leaky_global(X))
    assert np.max(np.abs(clean - leaky)) > 1e-3


def test_loocv_fails_if_standardisation_is_made_global(monkeypatch):
    X, y = _data()
    clean = qsar.loocv_predictions(X, y, ridge=True, alpha=3.0)
    monkeypatch.setattr(qsar, "_standardise_train_test", _leaky_global(X))
    leaky = qsar.loocv_predictions(X, y, ridge=True, alpha=3.0)
    assert np.max(np.abs(clean - leaky)) > 1e-3


# --- C13: skips that would hide a regression ---------------------------------

def _all_test_sources():
    return {p.name: p.read_text() for p in (ROOT / "tests").glob("test_*.py")}


def test_no_test_skips_on_missing_tracked_inputs_or_filtered_records():
    """The structure and single-scaffold tests used pytest.skip when a PDB
    file, a wild-type MTX record or the committed report was missing. All of
    those are tracked inputs: missing means broken, so those tests must fail."""
    src = _all_test_sources()
    assert "pytest.skip(" not in src["test_structure.py"]
    assert "pytest.skip(" not in src["test_qsar_single_scaffold.py"]


def test_wild_type_methotrexate_record_set_is_not_empty():
    wt = [r for r in datasets.load_chembl_dhfr_ki()
          if r["molecule_chembl_id"] == "CHEMBL34259"]
    allr = [r for r in datasets.load_chembl_dhfr_ki(wild_type_only=False)
            if r["molecule_chembl_id"] == "CHEMBL34259"]
    assert len(wt) >= 1
    # the filter removes the point mutants and keeps nothing it should not
    assert len(allr) > len(wt)
    assert max(r["ki_nM"] for r in wt) < min(
        r["ki_nM"] for r in allr if r not in wt)


def test_tracked_inputs_the_skipif_guards_depend_on_exist():
    for rel in ("data/chembl_dhfr/ecoli_dhfr_smiles.csv", "results_qsar.txt",
                "results_single_scaffold.txt", "data/pdb/1DDR.pdb",
                "data/pdb/4DFR.pdb", "data/pdb/3DAU.pdb", "data/pdb/1RX2.pdb"):
        assert (ROOT / rel).exists(), f"{rel} is missing"


def test_rdkit_is_installed_so_the_descriptor_tests_actually_run():
    """Most descriptor tests skip silently without RDKit. Fail loudly here
    instead, so a green run cannot mean 'the QSAR layer was never executed'."""
    assert qsar.Chem is not None, "RDKit is not installed: QSAR tests were skipped"
