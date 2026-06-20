"""vllm PR #43991 (MERGED) — [Model Runner V2] Use actual batch max_seq_len for
attn metadata.

Follow-up to #40654 (already present in the Aiden base's DefaultModelState).
Propagates "use actual batch max_seq_len, not max_model_len" to the two V2 paths
it missed: MambaHybridModelState.prepare_attn and the eagle/MTP draft
_build_draft_attn_metadata. Handing max_model_len to FlashInfer makes TRTLLM
attention walk past valid block-table entries.

dsv4 relevance: dsv4 runs an MTP draft through eagle/speculator.py, so the
speculator hunk is the directly-relevant fix. The mamba_hybrid hunk is for
Mamba/linear-attention hybrids (harmless to dsv4) and is applied for fidelity.

Eagle-hack collision check: the Aiden eagle hack (eagle_extra_cache_blocks /
_protect_prompt_blocks) lives in single_type_kv_cache_manager.py (KV-block
eviction protection) — a different file & mechanism. The draft_max_seq_len
change here is orthogonal. The base speculator.py regions are clean upstream V2.

All anchors verified count==1 against image dsv4-tiera:local.
"""

# ---------------------------------------------------------------------------
# File 1: model_states/default.py — drop the int() cast (substance already in
# base; this is the cosmetic delta the PR makes for consistency).
# ---------------------------------------------------------------------------
p1 = "/opt/env/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_states/default.py"
s1 = open(p1).read()
OLD1 = "            max_seq_len = int(seq_lens_cpu_upper_bound[:num_reqs].max().item())\n"
NEW1 = "            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()\n"
assert s1.count(OLD1) == 1, f"#43991 default.py anchor count = {s1.count(OLD1)} (expected 1)"
s1 = s1.replace(OLD1, NEW1)
open(p1, "w").write(s1)
print("default.py patched for #43991 OK")

# ---------------------------------------------------------------------------
# File 2: model_states/mamba_hybrid.py — compute real max_seq_len + use it.
# Base structure diverges from the PR diff (no seq_lens_cpu_upper_bound block,
# uses `if not for_capture:` for other state). We insert the upper-bound block
# right after the max_query_len line (PR placement) and swap the build call arg.
# ---------------------------------------------------------------------------
p2 = "/opt/env/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
s2 = open(p2).read()
OLD2A = "        max_query_len = input_batch.num_scheduled_tokens.max().item()\n"
NEW2A = (
    "        max_query_len = input_batch.num_scheduled_tokens.max().item()\n"
    "        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound\n"
    "        if for_capture:\n"
    "            # Capture with worst-case max_seq_len so the graph is valid at any replay.\n"
    "            max_seq_len = self.max_model_len\n"
    "        else:\n"
    "            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()\n"
)
OLD2B = "            max_seq_len=self.max_model_len,\n"
NEW2B = "            max_seq_len=max_seq_len,\n"
assert s2.count(OLD2A) == 1, f"#43991 mamba_hybrid maxqlen anchor count = {s2.count(OLD2A)} (expected 1)"
assert s2.count(OLD2B) == 1, f"#43991 mamba_hybrid build-arg anchor count = {s2.count(OLD2B)} (expected 1)"
s2 = s2.replace(OLD2A, NEW2A)
s2 = s2.replace(OLD2B, NEW2B)
open(p2, "w").write(s2)
print("mamba_hybrid.py patched for #43991 OK")

# ---------------------------------------------------------------------------
# File 3: spec_decode/eagle/speculator.py — the dsv4-relevant fix.
#   (a) init: add self.draft_max_seq_len = self.max_model_len
#   (b) _build_draft_attn_metadata: max_seq_len=self.draft_max_seq_len
#   (c) propose(): compute draft_max_seq_len from real batch max + spec steps
# ---------------------------------------------------------------------------
p3 = "/opt/env/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py"
s3 = open(p3).read()

OLD3A = (
    "        self.max_model_len = vllm_config.model_config.max_model_len\n"
    "        # We need to get the hidden size from the draft model config because\n"
)
NEW3A = (
    "        self.max_model_len = vllm_config.model_config.max_model_len\n"
    "        self.draft_max_seq_len = self.max_model_len\n"
    "        # We need to get the hidden size from the draft model config because\n"
)
OLD3B = "            max_seq_len=self.max_model_len,\n"
NEW3B = "            max_seq_len=self.draft_max_seq_len,\n"
OLD3C = "        max_query_len = input_batch.num_scheduled_tokens.max()\n"
NEW3C = (
    "        max_query_len = input_batch.num_scheduled_tokens.max()\n"
    "        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()\n"
    "        self.draft_max_seq_len = min(\n"
    "            max_seq_len + self.num_speculative_steps, self.max_model_len\n"
    "        )\n"
)
assert s3.count(OLD3A) == 1, f"#43991 speculator init anchor count = {s3.count(OLD3A)} (expected 1)"
assert s3.count(OLD3B) == 1, f"#43991 speculator build-arg anchor count = {s3.count(OLD3B)} (expected 1)"
assert s3.count(OLD3C) == 1, f"#43991 speculator propose anchor count = {s3.count(OLD3C)} (expected 1)"
s3 = s3.replace(OLD3A, NEW3A)
s3 = s3.replace(OLD3B, NEW3B)
s3 = s3.replace(OLD3C, NEW3C)
open(p3, "w").write(s3)
print("speculator.py patched for #43991 OK")
