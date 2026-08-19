# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
)
from vllm.v1.worker.mamba_checkpoint_utils import (
    make_mamba_checkpoint_metadata_cpu,
)


def test_mrv2_mamba_checkpoint_metadata_reaches_common_metadata() -> None:
    offsets, state_indices = make_mamba_checkpoint_metadata_cpu(
        req_ids=["req-a", "req-b"],
        num_reqs_padded=4,
        num_scheduled_tokens={"req-a": 2048, "req-b": 512},
        checkpoints={
            1: {"req-a": (17, 1536)},
            3: {"req-a": (23, 1536), "req-b": (29, 128)},
        },
    )
    metadata = MambaHybridAttnMetadata(
        is_prefilling=torch.tensor([True, True, False, False]),
        checkpoint_offsets_cpu=offsets,
        checkpoint_state_indices_by_group_cpu=state_indices,
    )

    group_one = metadata.get_extra_common_attn_kwargs(1, 2)
    assert group_one["mamba_checkpoint_offsets_cpu"].tolist() == [1536, 128]
    assert group_one["mamba_checkpoint_state_indices_cpu"].tolist() == [17, -1]

    group_three = metadata.get_extra_common_attn_kwargs(3, 2)
    assert group_three["mamba_checkpoint_offsets_cpu"].tolist() == [1536, 128]
    assert group_three["mamba_checkpoint_state_indices_cpu"].tolist() == [23, 29]

    group_without_mamba = metadata.get_extra_common_attn_kwargs(0, 2)
    assert "mamba_checkpoint_state_indices_cpu" not in group_without_mamba
