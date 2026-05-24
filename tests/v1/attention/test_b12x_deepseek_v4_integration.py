# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla import b12x_integration as b12x


@pytest.fixture(autouse=True)
def reset_b12x_status():
    for name in b12x._SUBSYSTEM_ORDER:
        b12x._b12x_subsystem_status[name] = {
            "state": "not_checked",
            "detail": "",
            "active": False,
        }
    yield


def _set_full_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    strict: bool = False,
    compressed_mla: bool | None = None,
    indexer: bool | None = None,
    mhc: bool | None = None,
    wo_projection: bool | None = None,
    moe: bool | None = None,
) -> None:
    monkeypatch.setattr(b12x.envs, "VLLM_USE_B12X_DEEPSEEK_V4", True)
    monkeypatch.setattr(b12x.envs, "VLLM_USE_B12X_DEEPSEEK_V4_STRICT", strict)
    monkeypatch.setattr(
        b12x.envs,
        "VLLM_USE_B12X_DEEPSEEK_V4_COMPRESSED_MLA",
        compressed_mla,
    )
    monkeypatch.setattr(b12x.envs, "VLLM_USE_B12X_DEEPSEEK_V4_INDEXER", indexer)
    monkeypatch.setattr(b12x.envs, "VLLM_USE_B12X_DEEPSEEK_V4_MHC", mhc)
    monkeypatch.setattr(
        b12x.envs,
        "VLLM_USE_B12X_DEEPSEEK_V4_WO_PROJECTION",
        wo_projection,
    )
    monkeypatch.setattr(b12x.envs, "VLLM_USE_B12X_DEEPSEEK_V4_MOE", moe)


def test_b12x_subsystem_defaults_are_env_gated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(b12x.envs, "VLLM_USE_B12X_DEEPSEEK_V4", False)

    assert not b12x.subsystem_enabled("MHC")
    assert not b12x.subsystem_enabled("COMPRESSED_MLA")

    _set_full_env(monkeypatch)

    assert b12x.subsystem_enabled("MHC")
    assert not b12x.subsystem_enabled("COMPRESSED_MLA")
    assert not b12x.subsystem_enabled("INDEXER")
    assert not b12x.subsystem_enabled("WO_PROJECTION")
    assert not b12x.subsystem_enabled("MOE")


def test_b12x_status_rows_include_full_matrix(monkeypatch: pytest.MonkeyPatch):
    _set_full_env(
        monkeypatch,
        strict=True,
        compressed_mla=True,
        indexer=True,
        mhc=True,
        wo_projection=True,
        moe=True,
    )

    rows = b12x.b12x_status_rows()

    assert [row["subsystem"] for row in rows] == [
        "compressed MLA",
        "paged MQA indexer",
        "mHC",
        "WO projection",
        "MoE",
    ]
    assert all(row["requested"] for row in rows)
    assert all(row["strict"] for row in rows)


def test_b12x_strict_fallback_raises_and_records_status(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_full_env(monkeypatch, strict=True, compressed_mla=True)

    with pytest.raises(RuntimeError, match="compressed MLA requested in strict mode"):
        b12x.b12x_subsystem_fallback(
            "COMPRESSED_MLA",
            "compressed MLA",
            "metadata unsupported",
        )

    row = next(
        row
        for row in b12x.b12x_status_rows()
        if row["subsystem"] == "compressed MLA"
    )
    assert row["state"] == "fallback"
    assert row["detail"] == "metadata unsupported"


def test_flatten_page_metadata_uses_raw_compressed_token_lengths():
    block_table = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([[128, 256], [384, 512]], dtype=torch.int32)

    flat_table, flat_lens = b12x._flatten_page_metadata(
        block_table=block_table,
        seq_lens=seq_lens,
    )

    assert flat_table.tolist() == [
        [10, 11, 12],
        [10, 11, 12],
        [20, 21, 22],
        [20, 21, 22],
    ]
    assert flat_lens.tolist() == [128, 256, 384, 512]
