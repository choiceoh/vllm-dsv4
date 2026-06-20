"""vllm PR #43961 (MERGED) — Corrupted MLA + linear attention.

Adds MLAAttentionSpec to the new_block_ids reporting gate in BOTH
SingleTypeKVCacheManager.allocate_new_computed_blocks and .allocate_new_blocks.
Without it, newly allocated MLA blocks (DeepSeek-V4 uses MLA) are never reported
to attention metadata once the KV cache fills, corrupting the tail page ->
garbage output (GSM8K collapses across epochs in the upstream repro).

Aiden base relationship (verified against image dsv4-tiera:local):
  - MLAAttentionManager (base line 620) extends FullAttentionManager (619) which
    extends SingleTypeKVCacheManager (31). NEITHER subclass overrides
    allocate_new_blocks / allocate_new_computed_blocks, so MLA specs flow through
    these base-class methods -> the gate here is exactly the right place.
  - The Aiden eagle hack (eagle_extra_cache_blocks, _protect_prompt_blocks,
    _protected_prompt_block_ids) lives in SlidingWindowManager._cache_block_mask
    / MLAAttentionManager.cache_blocks — a prefix-cache-mask & eviction-protection
    concern, ORTHOGONAL to new_block_ids allocation tracking. No conflict.
  - MLAAttentionSpec is already imported (base line 22). No import change needed.

The two anchors are byte-identical; we disambiguate by the preceding extend(...)
line so each replacement is unique (count==1 each).
"""
p = "/opt/env/lib/python3.12/site-packages/vllm/v1/core/single_type_kv_cache_manager.py"
s = open(p).read()

# Site 1: allocate_new_computed_blocks (preceded by extend(allocated_blocks))
OLD1 = (
    "            req_blocks.extend(allocated_blocks)\n"
    "            if type(self.kv_cache_spec) in (FullAttentionSpec, TQFullAttentionSpec):\n"
    "                self.new_block_ids.extend(b.block_id for b in allocated_blocks)\n"
)
NEW1 = (
    "            req_blocks.extend(allocated_blocks)\n"
    "            if type(self.kv_cache_spec) in (\n"
    "                FullAttentionSpec,\n"
    "                TQFullAttentionSpec,\n"
    "                MLAAttentionSpec,\n"
    "            ):\n"
    "                self.new_block_ids.extend(b.block_id for b in allocated_blocks)\n"
)

# Site 2: allocate_new_blocks (preceded by extend(new_blocks))
OLD2 = (
    "            req_blocks.extend(new_blocks)\n"
    "            if type(self.kv_cache_spec) in (FullAttentionSpec, TQFullAttentionSpec):\n"
    "                self.new_block_ids.extend(b.block_id for b in new_blocks)\n"
)
NEW2 = (
    "            req_blocks.extend(new_blocks)\n"
    "            if type(self.kv_cache_spec) in (\n"
    "                FullAttentionSpec,\n"
    "                TQFullAttentionSpec,\n"
    "                MLAAttentionSpec,\n"
    "            ):\n"
    "                self.new_block_ids.extend(b.block_id for b in new_blocks)\n"
)

assert s.count(OLD1) == 1, f"#43961 site1 anchor count = {s.count(OLD1)} (expected 1)"
assert s.count(OLD2) == 1, f"#43961 site2 anchor count = {s.count(OLD2)} (expected 1)"
# MLAAttentionSpec must already be importable in this module.
assert "    MLAAttentionSpec,\n" in s, "#43961: MLAAttentionSpec import missing — refusing to patch"

s = s.replace(OLD1, NEW1)
s = s.replace(OLD2, NEW2)
open(p, "w").write(s)
print("single_type_kv_cache_manager.py patched for #43961 OK (both sites)")
