from __future__ import annotations

import re

import pytest

from src.utils.run_naming import timestamped_run_name, validate_run_name


def test_timestamped_run_name_is_sortable_and_has_microseconds() -> None:
    assert re.fullmatch(r"\d{8}_\d{6}_\d{6}", timestamped_run_name())


def test_timestamped_run_name_can_include_a_safe_partition_tag() -> None:
    assert re.fullmatch(
        r"win1s_ov50_\d{8}_\d{6}_\d{6}",
        timestamped_run_name("win1s_ov50"),
    )


def test_timestamped_run_name_rejects_an_unsafe_partition_tag() -> None:
    with pytest.raises(ValueError):
        timestamped_run_name("../shared")


@pytest.mark.parametrize("value", ["", "../old", "nested/run", ".", ".."])
def test_validate_run_name_rejects_unsafe_path_components(value: str) -> None:
    with pytest.raises(ValueError):
        validate_run_name(value)


def test_validate_run_name_accepts_descriptive_name() -> None:
    assert validate_run_name("ablation_no_prior") == "ablation_no_prior"
