"""
Unit tests for the LSTM sequence windowing and loss masking.

These cover the two subtle pieces of the LSTM baseline:
  * windows never cross a site (or year-split) boundary, and eval windows tile
    every post-warmup step exactly once (no gaps, no double-counting);
  * the training loss mask ignores BOTH warmup steps and missing targets.

Run:  python tests/test_lstm_windowing.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader import _train_window_starts, _eval_window_specs
from models.lstm import _WindowDataset


def test_train_window_starts():
    # Overlapping windows, right-aligned final window covers the tail.
    starts = _train_window_starts(length=1000, window=720, stride=168)
    assert starts[0] == 0
    assert starts[-1] == 1000 - 720, starts          # tail covered
    assert all(s + 720 <= 1000 for s in starts)      # never runs off the block
    # Series shorter than one window -> a single padded window at 0.
    assert _train_window_starts(500, 720, 168) == [0]
    assert _train_window_starts(0, 720, 168) == []
    print("ok  test_train_window_starts")


def test_eval_windows_tile_exactly_once():
    length, window, warmup = 2000, 720, 168
    specs = _eval_window_specs(length, window, warmup)
    covered = np.zeros(length, dtype=int)
    for (start, own_lo, own_hi) in specs:
        assert start >= 0 and start + own_hi <= length      # stays in the block
        assert own_lo >= warmup or start > 0                # warmup is context only
        covered[start + own_lo:start + own_hi] += 1
    # Every step from warmup onward is predicted exactly once; earlier steps 0.
    assert (covered[:warmup] == 0).all(), "first warmup steps must be unpredicted"
    assert (covered[warmup:] == 1).all(), "post-warmup steps must be covered once"
    # A block too short to clear warmup emits nothing.
    assert _eval_window_specs(warmup, window, warmup) == []
    print("ok  test_eval_windows_tile_exactly_once")


def test_eval_windows_cover_warmup():
    # cover_warmup=True: coverage is [0, length) -- every step scored once,
    # including the leading warmup region (test set matches the flat baselines).
    length, window, warmup = 2000, 720, 168
    specs = _eval_window_specs(length, window, warmup, cover_warmup=True)
    covered = np.zeros(length, dtype=int)
    for (start, own_lo, own_hi) in specs:
        assert start >= 0 and start + own_hi <= length      # stays in the block
        covered[start + own_lo:start + own_hi] += 1
    assert (covered == 1).all(), "every step must be covered exactly once"
    # A block shorter than warmup still emits a single window covering it all.
    short = _eval_window_specs(warmup, window, warmup, cover_warmup=True)
    covered_short = np.zeros(warmup, dtype=int)
    for (start, own_lo, own_hi) in short:
        covered_short[start + own_lo:start + own_hi] += 1
    assert (covered_short == 1).all(), "short block must be fully covered"
    # An empty block emits nothing regardless.
    assert _eval_window_specs(0, window, warmup, cover_warmup=True) == []
    print("ok  test_eval_windows_cover_warmup")


def test_loss_mask_ignores_warmup_and_missing():
    window, warmup = 10, 3
    length = 10
    # valid targets everywhere EXCEPT one missing step at t=5.
    valid = np.ones(length, dtype=bool)
    valid[5] = False
    target = np.arange(length, dtype=np.float32)
    target[~valid] = np.nan                       # missing target is NaN
    block = {'feats': np.zeros((length, 2), np.float32),
             'target': target, 'valid': valid}
    ds = _WindowDataset([block], [(0, 0)], window=window, warmup=warmup)
    Xw, yw, mw = ds[0]
    mw = mw.numpy().astype(bool)

    expected = valid.copy()
    expected[:warmup] = False                     # warmup ignored
    assert np.array_equal(mw, expected), (mw, expected)
    assert not mw[5], "missing target must be masked"
    assert mw[3] and mw[9], "valid post-warmup steps must be kept"
    # No NaNs leak into the tensors the loss sees.
    assert np.isfinite(yw.numpy()).all()
    assert np.isfinite(Xw.numpy()).all()
    print("ok  test_loss_mask_ignores_warmup_and_missing")


def test_short_block_padding_mask():
    # Block shorter than the window: padded steps must never count in the loss.
    window, warmup, length = 10, 3, 6
    block = {'feats': np.ones((length, 2), np.float32),
             'target': np.ones(length, np.float32),
             'valid': np.ones(length, bool)}
    ds = _WindowDataset([block], [(0, 0)], window=window, warmup=warmup)
    _Xw, _yw, mw = ds[0]
    mw = mw.numpy().astype(bool)
    assert mw[length:].sum() == 0, "padded steps must be masked out"
    assert mw[warmup:length].all(), "real post-warmup steps must be kept"
    print("ok  test_short_block_padding_mask")


if __name__ == "__main__":
    test_train_window_starts()
    test_eval_windows_tile_exactly_once()
    test_eval_windows_cover_warmup()
    test_loss_mask_ignores_warmup_and_missing()
    test_short_block_padding_mask()
    print("\nAll LSTM windowing/masking tests passed.")
