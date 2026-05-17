# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_b12x_fused_moe


class FlashInferB12xW4A16Experts(mk.FusedMoEExpertsModular):
    """FlashInfer B12x SM12x W4A16 MoE experts for MXFP4 DeepSeek V4."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.use_mxfp4_w4a16

        self.hidden_dim = moe_config.hidden_dim
        self.hidden_dim_unpadded = (
            moe_config.hidden_dim_unpadded or moe_config.hidden_dim
        )
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.topk = moe_config.experts_per_token
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.device = torch.device(moe_config.device)
        self.out_dtype = moe_config.in_dtype
        self.max_capture_size = (
            get_current_vllm_config().compilation_config.max_cudagraph_capture_size
        )

        self._w1_alpha: torch.Tensor | None = None
        self._w2_alpha: torch.Tensor | None = None
        self._prepared_weights = None
        self._expert_map_cache_key: tuple[int, torch.device, torch.dtype] | None = None
        self._expert_map_cache: torch.Tensor | None = None

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and has_flashinfer_b12x_fused_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        del w1, w2
        if a1.dim() == 2:
            assert topk_ids.size(0) == a1.size(0), (
                f"{topk_ids.size(0)} != {a1.size(0)}"
            )
            M = a1.size(0)
        else:
            assert a1.dim() == 3
            assert a1.size(0) == self.local_num_experts, (
                f"{a1.size(0)} != {self.local_num_experts}"
            )
            M = a1.size(1)

        assert topk_ids.dim() == 2
        return (
            self.local_num_experts,
            M,
            2 * self.intermediate_size_per_partition,
            self.hidden_dim,
            topk_ids.size(1),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, topk, global_num_experts, local_num_experts, expert_tokens_meta
        assert activation == MoEActivation.SILU
        assert K == self.hidden_dim
        return (0,), (0,), (M, self.hidden_dim_unpadded)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.w1_scale is None or self.w2_scale is None:
            raise ValueError("FlashInfer B12x requires MXFP4 weight scales.")
        if self.w1_bias is not None or self.w2_bias is not None:
            raise NotImplementedError("FlashInfer B12x W4A16 does not support MoE bias.")

        device = layer.w13_weight.device
        self._w1_alpha = torch.ones(
            (self.local_num_experts,), dtype=torch.float32, device=device
        )
        self._w2_alpha = torch.ones(
            (self.local_num_experts,), dtype=torch.float32, device=device
        )

        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
            prepare_w4a16_packed_weights,
        )

        self._prepared_weights = prepare_w4a16_packed_weights(
            layer.w13_weight,
            self.w1_scale,
            self._w1_alpha,
            layer.w2_weight,
            self.w2_scale,
            self._w2_alpha,
            activation="silu",
            params_dtype=self.out_dtype,
            source_format="compressed_tensors",
        )

        # The prepared FlashInfer object is the only representation used at
        # runtime. Release the converted MXFP4 tensors immediately so full
        # DeepSeek V4 does not keep both layouts resident on GB10 unified memory.
        empty_weight = torch.empty(
            (self.local_num_experts, 0, 0),
            dtype=layer.w13_weight.dtype,
            device=device,
        )
        replace_parameter(layer, "w13_weight", empty_weight)
        replace_parameter(layer, "w2_weight", empty_weight)

        empty_scale = torch.empty(
            (0,),
            dtype=self.w1_scale.dtype,
            device=device,
        )
        replace_parameter(layer, "w13_weight_scale", empty_scale)
        replace_parameter(layer, "w2_weight_scale", empty_scale)
        self.quant_config._w1.scale = empty_scale
        self.quant_config._w2.scale = empty_scale

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _normalize_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)
        if not topk_ids.is_contiguous():
            topk_ids = topk_ids.contiguous()
        return topk_ids

    def _normalize_topk_weights(self, topk_weights: torch.Tensor) -> torch.Tensor:
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.float()
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()
        return topk_weights

    def _normalize_expert_map(
        self,
        expert_map: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if expert_map is None:
            return None
        key = (expert_map.data_ptr(), device, expert_map.dtype)
        if self._expert_map_cache_key == key and self._expert_map_cache is not None:
            return self._expert_map_cache
        normalized = expert_map
        if normalized.device != device or normalized.dtype != torch.int32:
            normalized = normalized.to(device=device, dtype=torch.int32)
        if not normalized.is_contiguous():
            normalized = normalized.contiguous()
        self._expert_map_cache_key = key
        self._expert_map_cache = normalized
        return normalized

    def _get_workspace(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
    ):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _get_cached_workspace,
        )

        routed_rows = int(topk_ids.size(0)) * int(self.topk)
        workspace = _get_cached_workspace(
            backend="w4a16",
            state_E=self.local_num_experts,
            weight_E=self.global_num_experts,
            routed_rows=max(1, routed_rows),
            k=int(hidden_states.size(1)),
            n=self.intermediate_size_per_partition,
            num_topk=self.topk,
            device=hidden_states.device,
            quant_mode="w4a16",
            activation="silu",
        )
        normalized_expert_map = self._normalize_expert_map(
            expert_map, hidden_states.device
        )
        if normalized_expert_map is not None:
            workspace.expert_map = normalized_expert_map
        return workspace

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        del a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta
        del apply_router_weight_on_input
        assert activation == MoEActivation.SILU
        assert hidden_states.dtype == torch.bfloat16
        assert self._w1_alpha is not None and self._w2_alpha is not None

        topk_ids = self._normalize_topk_ids(topk_ids)
        topk_weights = self._normalize_topk_weights(topk_weights)
        workspace = self._get_workspace(hidden_states, topk_ids, expert_map)

        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
        )

        launch_sm120_moe(
            a=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w1_weight=w1,
            w1_weight_sf=self.w1_scale,
            w1_alpha=self._w1_alpha,
            fc2_input_scale=None,
            w2_weight=w2,
            w2_weight_sf=self.w2_scale,
            w2_alpha=self._w2_alpha,
            num_experts=global_num_experts,
            top_k=self.topk,
            num_local_experts=self.local_num_experts,
            scatter_output=output,
            activation="silu",
            quant_mode="w4a16",
            source_format="compressed_tensors",
            _workspace=workspace,
            _prepared_weights=self._prepared_weights,
        )
