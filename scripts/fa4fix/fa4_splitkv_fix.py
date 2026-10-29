"""Runtime fix for split-KV in vLLM 0.32's FA4 (CuTe DSL) SM90 forward kernel.

flash_fwd_sm90.py builds BlockInfo without num_splits (it defaults to 1), and
BlockInfo.get_n_block_min_max uses that field instead of its num_splits argument.
Split 0 therefore gets the whole KV range and splits 1..N-1 get none, so a decode step
runs on one CTA per KV head. The output is still correct (the combine kernel ignores
empty splits), which hides the problem.

This module replaces get_n_block_min_max with a version that uses the num_splits
argument the SM90 kernel passes. The installed package is not modified.
Upstream-style one-line fix: patches/fa4_sm90_splitkv.patch.
"""
from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from vllm.vllm_flash_attn.cute import block_info as _bi


@cute.jit
def get_n_block_min_max(self, seqlen_info, m_block: Int32, split_idx: Int32 = 0,
                        num_splits: Int32 = 1) -> Tuple[Int32, Int32]:
    n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
    if const_expr(self.is_causal or (self.is_local and self.window_size_right is not None)):
        m_idx_max = (m_block + 1) * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
        n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = n_idx if const_expr(self.is_causal) else n_idx + self.window_size_right
        n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
    n_block_min = 0
    if const_expr(self.is_local and self.window_size_left is not None):
        m_idx_min = m_block * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_left = n_idx - self.window_size_left
        n_block_min = cutlass.max(n_idx_left // self.tile_n, 0)
    if cutlass.const_expr(self.is_split_kv):
        if const_expr(self.pack_split_idx):
            num_splits = split_idx >> 16
            split_idx = split_idx & 0xFFFF
        # (original: `else: num_splits = self.num_splits`, which is 1 on SM90)
        if const_expr(self.num_n_blocks_per_split is not None):
            num_n_blocks_per_split = self.num_n_blocks_per_split
        else:
            num_n_blocks_per_split = (
                Int32(0)
                if n_block_max <= n_block_min
                else (n_block_max - n_block_min + num_splits - 1) // num_splits
            )
        n_block_min = n_block_min + split_idx * num_n_blocks_per_split
        n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
    return n_block_min, n_block_max


_bi.BlockInfo.get_n_block_min_max = get_n_block_min_max
