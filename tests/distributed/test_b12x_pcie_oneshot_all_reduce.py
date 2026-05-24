# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from vllm.distributed.device_communicators import custom_all_reduce as car


class _FakeB12xRuntime:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_exchange_group(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls()

    def for_stream(self):
        return SimpleNamespace(should_allreduce=lambda inp: True)

    def all_reduce(self, inp, *, out=None):
        return inp if out is None else out

    @contextmanager
    def capture(self):
        yield

    def close(self):
        pass


def _patch_common_runtime(monkeypatch, *, same_node: bool) -> object:
    cpu_group = object()
    device_group = object()

    def get_backend(group):
        return "nccl" if group is device_group else "gloo"

    def all_gather(gather_list, tensor, group):
        for rank, gathered in enumerate(gather_list):
            gathered.copy_(torch.tensor([rank], dtype=torch.int))

    monkeypatch.setattr(car, "custom_ar", False)
    monkeypatch.setattr(car.dist, "get_backend", get_backend)
    monkeypatch.setattr(car.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(car.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(car.dist, "all_gather", all_gather)
    monkeypatch.setattr(
        car, "in_the_same_node_as", lambda group, source_rank=0: [same_node, True]
    )
    monkeypatch.setattr(car, "_can_p2p", lambda rank, world_size: True)
    monkeypatch.setattr(car, "_load_b12x_pcie_oneshot_runtime", lambda: _FakeB12xRuntime)
    monkeypatch.setattr(car.envs, "VLLM_USE_B12X_PCIE_ONESHOT_ALLREDUCE", True)
    monkeypatch.setattr(car.envs, "VLLM_B12X_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "64KB")
    monkeypatch.setattr(car.envs, "CUDA_VISIBLE_DEVICES", None)
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(car.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(car.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(car.current_platform, "device_count", lambda: 2)
    monkeypatch.setattr(car.current_platform, "get_device_capability", lambda: None)
    monkeypatch.setattr(car.current_platform, "is_fully_connected", lambda ids: False)

    return cpu_group, device_group


def test_parse_b12x_pcie_oneshot_max_size() -> None:
    assert car._parse_b12x_pcie_oneshot_max_size("64KB") == 64 * 1024
    assert car._parse_b12x_pcie_oneshot_max_size("2M") == 2 * 1024 * 1024
    assert car._parse_b12x_pcie_oneshot_max_size(4096) == 4096


def test_b12x_pcie_oneshot_runtime_uses_device_group(monkeypatch) -> None:
    _FakeB12xRuntime.calls.clear()
    cpu_group, device_group = _patch_common_runtime(monkeypatch, same_node=True)

    communicator = car.CustomAllreduce(
        group=cpu_group,
        device=torch.device("cuda:0"),
        max_size=8 * 1024 * 1024,
        device_group=device_group,
    )

    assert not communicator.disabled
    assert communicator.max_size == 64 * 1024
    assert _FakeB12xRuntime.calls == [
        {
            "exchange_group": device_group,
            "device": torch.device("cuda:0"),
            "eager_buffer_bytes": 64 * 1024,
            "max_size": 64 * 1024,
        }
    ]


def test_b12x_pcie_oneshot_stays_disabled_cross_node(monkeypatch) -> None:
    _FakeB12xRuntime.calls.clear()
    cpu_group, device_group = _patch_common_runtime(monkeypatch, same_node=False)

    communicator = car.CustomAllreduce(
        group=cpu_group,
        device=torch.device("cuda:0"),
        device_group=device_group,
    )

    assert communicator.disabled
    assert _FakeB12xRuntime.calls == []
