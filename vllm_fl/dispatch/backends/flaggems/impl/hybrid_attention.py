"""Use FlagGems for prefill-only batches and Triton when decoding is mixed in.

This backend is experimental.  The FlagGems cache layout is kept for writes;
Triton receives a strided view of the same KV storage for decode reads.
"""

from dataclasses import dataclass

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

from .attention import (
    AttentionFLBackend,
    AttentionFLImpl,
    AttentionFLMetadata,
    AttentionFLMetadataBuilder,
)


@dataclass
class HybridAttentionMetadata(AttentionFLMetadata):
    triton_metadata: TritonAttentionMetadata | None = None


class HybridAttentionMetadataBuilder(AttentionFLMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.triton_builder = TritonAttentionMetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, device
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        flaggems = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        triton = None
        use_triton = flaggems.max_query_len == 1
        if not use_triton and self.dcp_world_size == 1 and not flaggems.use_cascade:
            # A chunked prefill may share its batch with single-token decode.
            # Route that entire batch to Triton so decode does not wait for the
            # FlagGems kernel. query_start_loc_cpu is already on the host.
            starts = common_attn_metadata.query_start_loc_cpu
            query_lens = starts[1:] - starts[:-1]
            use_triton = bool(torch.any(query_lens == 1))
        if use_triton and not flaggems.use_cascade and self.dcp_world_size == 1:
            triton = self.triton_builder.build(
                common_prefix_len, common_attn_metadata, fast_build=fast_build
            )
        return HybridAttentionMetadata(**vars(flaggems), triton_metadata=triton)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        metadata = self.build(0, common_attn_metadata)
        if metadata.triton_metadata is not None:
            # Triton's decode graph capture must not iterate over max_model_len.
            metadata.triton_metadata.seq_lens.fill_(1)
        return metadata


class HybridAttentionImpl(AttentionFLImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.triton_impl = TritonAttentionImpl(*args, **kwargs)

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        if (
            self.attn_type == AttentionType.DECODER
            and attn_metadata is not None
            and attn_metadata.triton_metadata is not None
        ):
            # FlagGems stores [K/V, blocks, slots, heads, dim], while Triton
            # expects [blocks, K/V, slots, heads, dim].  This is only a view;
            # unified_attention uses the tensor strides for KV reads.
            return self.triton_impl.forward(
                layer,
                query,
                key,
                value,
                kv_cache.transpose(0, 1),
                attn_metadata.triton_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )


class HybridAttentionBackend(AttentionFLBackend):
    @staticmethod
    def get_impl_cls():
        return HybridAttentionImpl

    @staticmethod
    def get_builder_cls():
        return HybridAttentionMetadataBuilder
