"""Tests for configuration validation."""

from __future__ import annotations

import pytest

from poolguard.config import PoolGuardConfig


def test_invalid_smd_threshold() -> None:
    with pytest.raises(ValueError, match="smd_threshold"):
        PoolGuardConfig(smd_threshold=0.0)


def test_invalid_ks_alpha() -> None:
    with pytest.raises(ValueError, match="ks_alpha"):
        PoolGuardConfig(ks_alpha=1.5)


def test_invalid_bootstrap_n() -> None:
    with pytest.raises(ValueError, match="bootstrap_n"):
        PoolGuardConfig(bootstrap_n=50)


def test_invalid_bootstrap_ci() -> None:
    with pytest.raises(ValueError, match="bootstrap_ci"):
        PoolGuardConfig(bootstrap_ci=1.0)
