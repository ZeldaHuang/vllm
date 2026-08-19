# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def make_mamba_checkpoint_metadata_cpu(
    req_ids: list[str],
    num_reqs_padded: int,
    num_scheduled_tokens: dict[str, int],
    checkpoints: dict[int, dict[str, tuple[int, int]]],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Build per-request checkpoint offsets and state destinations."""
    checkpoint_offsets_cpu = torch.full(
        (num_reqs_padded,), -1, dtype=torch.int32, device="cpu"
    )
    state_indices_by_group_cpu: dict[int, torch.Tensor] = {}
    checkpoint_offsets: dict[str, int] = {}

    for kv_cache_gid, req_to_checkpoint in checkpoints.items():
        state_indices_cpu = torch.full(
            (num_reqs_padded,), -1, dtype=torch.int32, device="cpu"
        )
        for req_idx, req_id in enumerate(req_ids):
            checkpoint = req_to_checkpoint.get(req_id)
            if checkpoint is None:
                continue
            block_id, checkpoint_offset = checkpoint
            state_indices_cpu[req_idx] = block_id
            previous = checkpoint_offsets.setdefault(req_id, checkpoint_offset)
            assert previous == checkpoint_offset, (
                "Mamba groups requested inconsistent checkpoint offsets for "
                f"{req_id}: {previous} and {checkpoint_offset}"
            )
        state_indices_by_group_cpu[kv_cache_gid] = state_indices_cpu

    for req_idx, req_id in enumerate(req_ids):
        num_scheduled = num_scheduled_tokens.get(req_id, 0)
        request_checkpoint_offset = checkpoint_offsets.get(req_id)
        if request_checkpoint_offset is None:
            continue
        assert 0 < request_checkpoint_offset < num_scheduled, (
            f"Mamba checkpoint offset {request_checkpoint_offset} for {req_id} must "
            f"fall strictly inside its {num_scheduled}-token model input"
        )
        checkpoint_offsets_cpu[req_idx] = request_checkpoint_offset

    return checkpoint_offsets_cpu, state_indices_by_group_cpu
