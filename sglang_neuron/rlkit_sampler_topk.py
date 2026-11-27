# SPDX-License-Identifier: Apache-2.0
"""RLKIT: the topk_topp sampler (sampling.sample_filtered_sharded_logits, v0.1.0) with its shard-local unfiltered draw
done by the temperature sampler's NKI kernel (rlkit_sampler_kernel.gumbel_partials, used unchanged).

Selected by RLKIT_SAMPLER_KERNEL_TOPK=1 (the guarded branch in sampling.sample_filtered_sharded_logits). Unset or 0,
the released function body runs.

What the kernel replaces. In the released function the shard-local stage over [R, W] is: the padded-vocabulary mask,
the per-column hash noise and the Gumbel score (_tempered_scores), the shard maximum of the score and the smallest
column holding it (_first_max), the raw logit at the winner, two log-sum-exps and the shard's top-`candidates` pool
(_pool_topk). Here the noise, the score, the maximum and the winner id come from gumbel_partials, merged per row with the
engine's tie rule (_max_smallest_id), exactly as in rlkit_sampler_kernel.sample_sharded_logits_kernel.

What stays the engine's torch code, on purpose:
- both log-sum-exps (torch.logsumexp, as released). The tempered one feeds the pool probabilities, and so the top-p and
  min-p keep mask; a kernel exp-sum differs from torch in the last bits and could move a row across a top-p boundary.
  The kernel's own log-sum-exp partials (m, s) are computed and not used.
- the shard pool top-k (_pool_topk: the rotational NKI top-k on a device, torch.topk on CPU), the raw-logit gather, and
  everything after the payload (all-gather, shard winner, pool sort, probabilities, keep mask, pool draw with torch
  noise, logprobs, top logprobs): sampling.py v0.1.0 lines 379-408, copied verbatim (the selftest checks the text).

So a restricted sampled row (top-k below TOP_K_ALL, top-p below 1 or min-p above 0, temperature above 0) gets its
token and logprob from code that is the released code on the same inputs; the kernel's result is only read for
temperature-0 rows (max and smallest id: exact) and for unrestricted sampled rows (Gumbel noise from the kernel's own
log, which in the NKI simulator is not torch.log bit for bit, so a near-tie can resolve differently).
Limits: at most 128 rows per call (P_MAX partitions; the kernel has no row tiling), so a server with this switch
needs --max-running-requests <= 128. More rows raise a RuntimeError that names this switch; the released path has no
such limit. Row keys must lie in [0, 2**31), as noise_row_keys makes them (see the comment at the keys below).
Status: CPU and NKI simulator tests only (tests/rlkit_sampler/selftest_sampler_topk.py). Not run on a device.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang_neuron.rlkit_sampler_kernel import BIG, NPART, P_MAX, _LANE, _device_partials, n_chunks, segments

logger = logging.getLogger(__name__)

_logged = []


@torch._dynamo.assume_constant_result       # trace-time log, as rlkit_sampler_kernel._log_first_use
def _log_first_use(rows: int, width: int, segs: int, part_shape: str, rank_info: str) -> None:
    if not _logged:
        _logged.append(1)
        logger.info("[rlkit] topk_topp sampler by the kernel: rows=%d shard_width=%d segments=%d partitions=%d "
                    "partials=%s pid=%d %s", rows, width, segs, rows * segs, part_shape, os.getpid(), rank_info)


def sample_filtered_sharded_logits_kernel(logits, group, rank, vocabulary_size, temperatures, row_keys, filters,
                                          top_logprobs, candidates, partials_fn=None):
    """Same contract and result as sampling.sample_filtered_sharded_logits (v0.1.0, lines 337-408); the caller has
    checked the vocabulary size and 0 <= top_logprobs <= candidates <= shard width. partials_fn: the shard-local
    stage (default: the NKI kernel; the selftest passes the CPU reference or the simulator)."""
    from sglang_neuron import sampling as S
    from sglang_neuron.parallel import collectives as neuron_collectives

    # the engine's names the verbatim tail below uses
    _shard_winner, _pool_topk, _max_smallest_id = S._shard_winner, S._pool_topk, S._max_smallest_id
    gumbel_noise, TOP_K_ALL, return_original_logprob = S.gumbel_noise, S.TOP_K_ALL, S.return_original_logprob

    if vocabulary_size >= 2**24:                                     # v0.1.0 lines 358-359
        raise ValueError("Packed float32 token IDs require a vocabulary smaller than 2**24")
    rows, width = logits.shape
    rank64 = rank.to(torch.int64)
    logits = logits.float()
    # the engine's mask, always applied as sampling._tempered_scores does, also when no column is padded (then it
    # masks nothing). Skipping it when width * world_size == vocabulary_size gave other temperature-0 logprobs than the
    # released path on Qwen3-0.6B at TP4 (one device run).
    ids = torch.arange(width, dtype=torch.int64, device=logits.device) + rank64 * width
    logits = logits.masked_fill(ids.unsqueeze(0) >= vocabulary_size, float("-inf"))
    # per-row columns, as sampling._tempered_scores (v0.1.0 lines 316-319)
    temperature = temperatures.reshape(rows, 1).to(torch.float32)
    sampled = temperature > 0
    inverse = torch.where(sampled, 1.0 / torch.where(sampled, temperature, torch.ones_like(temperature)),
                          torch.ones_like(temperature))
    gate = torch.where(sampled, torch.full_like(temperature, -1.0), torch.zeros_like(temperature))

    # the kernel's layout and call, as rlkit_sampler_kernel.sample_sharded_logits_kernel
    if rows > P_MAX:               # segments() would also raise, but its message names RLKIT_SAMPLER_KERNEL, the other switch
        raise RuntimeError(f"RLKIT topk_topp sampler kernel: {rows} rows exceed {P_MAX} partitions; this patch has no "
                           "row tiling (no silent fallback: unset RLKIT_SAMPLER_KERNEL_TOPK, or serve with "
                           f"--max-running-requests <= {P_MAX})")
    segs = segments(rows, width)
    seg_w = width // segs
    parts_n = rows * segs
    seg_index = torch.arange(segs, dtype=torch.int64, device=logits.device).repeat(rows)
    base = (seg_index * seg_w + rank64 * width).to(torch.float32).reshape(parts_n, 1)

    def per_part(v):
        return v.reshape(rows, 1).repeat_interleave(segs, dim=0).reshape(parts_n, 1)

    # Row keys must lie in [0, 2**31), as noise_row_keys (_mix31) makes them. The kernel masks them with _LANE; the
    # released gumbel_noise does not, so a key outside that range would give other unfiltered tokens than the released
    # path. No assert: it would add an op to the traced graph for a case no caller produces.
    keys = per_part(row_keys.to(torch.int32) & _LANE)
    fn = partials_fn or _device_partials
    if partials_fn is None:
        _log_first_use(rows, width, segs, f"[{parts_n},{n_chunks(seg_w) * NPART}]",
                       f"rank_in_group={getattr(group, 'rank_in_group', '?')}")
    # lse scale 1: the kernel's (m, s) partials are not used here (module docstring)
    ones = torch.ones_like(inverse)
    parts = fn(logits.reshape(parts_n, seg_w), keys, per_part(inverse), per_part(gate), per_part(ones), base)
    parts = parts.reshape(rows, segs * n_chunks(seg_w), NPART)
    value, token_f = _max_smallest_id(parts[:, :, 0], BIG - parts[:, :, 1], vocabulary_size - 1)
    local_id = (token_f.to(torch.int64) - rank64 * width).clamp(0, width - 1)
    token = local_id + rank64 * width                                # v0.1.0 line 373

    # the rest of the shard-local stage: the engine's code (v0.1.0 lines 370-371, 374-378)
    pool_vals, pool_ids = _pool_topk(logits, candidates)
    pool_ids = pool_ids.to(torch.int64)
    raw = logits.gather(1, local_id)
    original = return_original_logprob()
    scale = torch.ones_like(inverse) if original else inverse
    lse = torch.logsumexp(logits * scale, dim=-1, keepdim=True)
    tempered_lse = torch.logsumexp(logits * inverse, dim=-1, keepdim=True) if original else lse

    # from here on: sampling.py v0.1.0 lines 379-408, verbatim
    payload = [value, token.float(), raw, lse, tempered_lse, pool_vals,
               (pool_ids + rank.to(torch.int64) * width).float()]
    gathered = neuron_collectives.all_gather(group, torch.cat(payload, dim=-1), dim=-1)
    per_rank = gathered.reshape(rows, group.world_size, 5 + 2 * candidates)
    _, unfiltered_token, winner = _shard_winner(per_rank, vocabulary_size)
    unfiltered_raw = per_rank[:, :, 2].masked_fill(~winner, float("-inf")).amax(dim=-1, keepdim=True)
    global_lse = torch.logsumexp(per_rank[:, :, 3], dim=-1)
    global_tempered_lse = torch.logsumexp(per_rank[:, :, 4], dim=-1, keepdim=True)
    vals, order = _pool_topk(per_rank[:, :, 5:5 + candidates].reshape(rows, -1).contiguous(), candidates)
    order = order.to(torch.int64)
    pool_tokens = per_rank[:, :, 5 + candidates:].reshape(rows, -1).gather(1, order)
    probs = torch.exp(vals * inverse - global_tempered_lse)
    strictly_before = torch.ones(candidates, candidates, dtype=torch.float32, device=logits.device).triu(1)
    exclusive = probs @ strictly_before
    top_k, top_p, min_p = filters[:, 0:1], filters[:, 1:2], filters[:, 2:3]
    pool_rank = torch.arange(candidates, dtype=torch.float32, device=logits.device).unsqueeze(0)
    keep = (pool_rank < top_k) & (exclusive <= top_p) & (probs >= min_p * probs[:, :1])
    pool_score = torch.where(keep, vals * inverse + gumbel_noise(row_keys, pool_tokens.to(torch.int32)),
                             torch.full_like(vals, float("-inf")))
    _, pool_token = _max_smallest_id(pool_score, pool_tokens, vocabulary_size - 1)
    pool_raw = vals.masked_fill(pool_tokens != pool_token, float("-inf")).amax(dim=-1, keepdim=True)
    restricted = sampled & ((top_k < TOP_K_ALL) | (top_p < 1) | (min_p > 0))
    token_ids = torch.where(restricted, pool_token, unfiltered_token).squeeze(1)
    raw_token = torch.where(restricted, pool_raw, unfiltered_raw).squeeze(1)
    scale = scale.reshape(rows)
    result = {"token_ids": token_ids.to(torch.int32), "logprobs": raw_token * scale - global_lse}
    if top_logprobs:
        result["top_vals"] = vals[:, :top_logprobs] * scale.unsqueeze(-1) - global_lse.unsqueeze(-1)
        result["top_ids"] = pool_tokens[:, :top_logprobs].to(torch.int32)
    return result
