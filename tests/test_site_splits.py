"""
Unit tests for the fixed site groups of the spatial settings.

These guard the properties a held-out-site split must have, and in particular
make the `spatial-easy40-v2` test group reproducible: it is a uniform sample
(no replacement, seed 20260726) of 40 sites drawn from the spatial-easy40
train+test pool, i.e. every site minus the 20 validation sites.

Run:  python tests/test_site_splits.py [--path data]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader import _SPATIAL_TEST_GROUPS, _SPATIAL_VAL_GROUPS

V2_SEED = 20260726
V2_N_TEST = 40


def _all_sites(path):
    """Sorted site ids from `<path>/sites/*.csv`, or None if the data is absent."""
    sites_dir = os.path.join(path, "sites")
    if not os.path.isdir(sites_dir):
        return None
    return sorted(f[:-4] for f in os.listdir(sites_dir) if f.endswith(".csv"))


def test_groups_are_well_formed():
    for setting, test_group in _SPATIAL_TEST_GROUPS.items():
        val_group = _SPATIAL_VAL_GROUPS[setting]
        assert len(test_group) == len(set(test_group)), f"{setting}: duplicate test sites"
        assert len(val_group) == len(set(val_group)), f"{setting}: duplicate val sites"
        overlap = set(test_group) & set(val_group)
        assert not overlap, f"{setting}: test/val overlap {sorted(overlap)}"
    assert set(_SPATIAL_TEST_GROUPS) == set(_SPATIAL_VAL_GROUPS), \
        "every setting needs both a test and a val group"
    print("ok  test_groups_are_well_formed")


def test_v2_shares_v1_validation_set():
    # The whole point of the re-draw: only the test sites change.
    assert (_SPATIAL_VAL_GROUPS['spatial-easy40-v2']
            == _SPATIAL_VAL_GROUPS['spatial-easy40']), \
        "spatial-easy40-v2 must reuse the spatial-easy40 validation sites verbatim"
    assert (set(_SPATIAL_TEST_GROUPS['spatial-easy40-v2'])
            != set(_SPATIAL_TEST_GROUPS['spatial-easy40'])), \
        "spatial-easy40-v2 must hold out a different set of test sites"
    print("ok  test_v2_shares_v1_validation_set")


def test_v2_is_reproducible_from_the_seed(path):
    """Re-draw the v2 test group from scratch and compare to the stored list."""
    all_sites = _all_sites(path)
    if all_sites is None:
        print(f"skip test_v2_is_reproducible_from_the_seed (no {path}/sites)")
        return

    val = _SPATIAL_VAL_GROUPS['spatial-easy40-v2']
    stored = _SPATIAL_TEST_GROUPS['spatial-easy40-v2']

    # Candidates = the spatial-easy40 train + test sites (everything but val).
    pool = [s for s in all_sites if s not in set(val)]
    assert len(pool) == len(all_sites) - len(val)
    assert set(_SPATIAL_TEST_GROUPS['spatial-easy40']) <= set(pool)

    rng = np.random.default_rng(V2_SEED)
    redrawn = sorted(rng.choice(np.array(pool), size=V2_N_TEST, replace=False).tolist())
    assert redrawn == sorted(stored), (
        "stored spatial-easy40-v2 test group does not match a fresh draw with "
        f"seed {V2_SEED}; missing={sorted(set(redrawn) - set(stored))}, "
        f"unexpected={sorted(set(stored) - set(redrawn))}")
    print("ok  test_v2_is_reproducible_from_the_seed")


def test_groups_reference_real_sites(path):
    """Every named site must exist, and train must not be starved."""
    all_sites = _all_sites(path)
    if all_sites is None:
        print(f"skip test_groups_reference_real_sites (no {path}/sites)")
        return

    known = set(all_sites)
    for setting, test_group in _SPATIAL_TEST_GROUPS.items():
        val_group = _SPATIAL_VAL_GROUPS[setting]
        unknown = (set(test_group) | set(val_group)) - known
        assert not unknown, f"{setting}: unknown site ids {sorted(unknown)}"
        n_train = len(known - set(test_group) - set(val_group))
        assert n_train > 0, f"{setting}: no training sites left"
        # The three splits must partition the sites exactly.
        assert n_train + len(test_group) + len(val_group) == len(known), \
            f"{setting}: splits do not partition the site list"
    print("ok  test_groups_reference_real_sites")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data",
                        help="Data directory containing sites/")
    args = parser.parse_args()

    test_groups_are_well_formed()
    test_v2_shares_v1_validation_set()
    test_v2_is_reproducible_from_the_seed(args.path)
    test_groups_reference_real_sites(args.path)
    print("\nAll site split tests passed.")
