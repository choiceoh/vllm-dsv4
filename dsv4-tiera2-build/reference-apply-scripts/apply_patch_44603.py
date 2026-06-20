"""vllm PR #44603 (MERGED) — fix: pad dummy run query_start_loc.

In _dummy_run, query_start_loc beyond num_reqs is left stale, so it is not a
monotonic sequence; the MLA indexer's repeat_interleave then sees a negative
repeat count and the kernel aborts (`Assertion repeat >= 0 failed`). The real
prepare path (base line 1981) already fills the tail with cu_num_tokens[-1];
this applies the same fill to the dummy-run path.

dsv4 relevance: the upstream repro is GLM-5.1-FP8 in P/D disaggregation, but the
fault is in the MLA indexer dummy run during CUDA-graph capture, which dsv4 (MLA
+ indexer) also exercises. Low cost, additive, exact anchor — applied as the
optional P2.

num_reqs_padded is in scope at the anchor (used a few lines below for
commit_block_table). Anchor verified count==1 against image dsv4-tiera:local
(the real-prepare-path sibling at base line 1978 uses cu_num_tokens, a different
variable, so it is not matched).
"""
p = "/opt/env/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"
s = open(p).read()

OLD = (
    "                self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens\n"
    "                self.query_start_loc.copy_to_gpu()\n"
)
NEW = (
    "                self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens\n"
    "                self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(\n"
    "                    cum_num_tokens[-1]\n"
    "                )\n"
    "                self.query_start_loc.copy_to_gpu()\n"
)
assert s.count(OLD) == 1, f"#44603 anchor count = {s.count(OLD)} (expected 1)"
s = s.replace(OLD, NEW)
open(p, "w").write(s)
print("gpu_model_runner.py patched for #44603 OK")
