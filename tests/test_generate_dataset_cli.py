"""Lightweight tests for `backend.scripts.generate_dataset`'s argparse CLI.

Pure argument-parsing checks — no Postgres, no subprocess. BUILD_PLAN.md:
"the seed is mandatory for a fair eval," so `--seed` must have no default
and argparse must error (`SystemExit`) when it's omitted.
"""

from __future__ import annotations

import pytest

from backend.scripts.generate_dataset import parse_args


def test_seed_is_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_count_defaults_to_30_dev_size():
    args = parse_args(["--seed", "42"])
    assert args.seed == 42
    assert args.count == 30


def test_count_and_seed_both_parse_as_ints():
    args = parse_args(["--count", "100", "--seed", "42"])
    assert args.count == 100
    assert args.seed == 42
