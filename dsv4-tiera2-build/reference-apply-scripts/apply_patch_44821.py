"""vllm PR #44821 (MERGED) — fix: prefix DeepSeek V4 MTP projections.

Pass explicit prefix= to the DeepSeek-V4 MTP e_proj / h_proj ReplicatedLinear
layers in BOTH the nvidia and amd implementations. Without it, compressed-tensors
receives an empty layer_name while constructing the draft model and cannot match
its ignore/target rules (these projections use fp8 linear quant per the module
docstring, so prefix-based matching is relevant for dsv4). Harmless if the quant
matcher does not rely on prefix.

Anchors verified against image dsv4-tiera:local:
  - nvidia/mtp.py e_proj@82 h_proj@89 ; amd/mtp.py e_proj@79 h_proj@86
  - `prefix` is a constructor parameter in both files (in scope).
The e_proj and h_proj blocks are byte-identical within a file, so we anchor on
the preceding `self.<name> = ReplicatedLinear(` line for uniqueness.
"""

E_OLD = (
    "{NAME} = ReplicatedLinear(\n"
    "            config.hidden_size,\n"
    "            config.hidden_size,\n"
    "            bias=False,\n"
    "            return_bias=False,\n"
    "            quant_config=quant_config,\n"
    "        )\n"
)
E_NEW = (
    "{NAME} = ReplicatedLinear(\n"
    "            config.hidden_size,\n"
    "            config.hidden_size,\n"
    "            bias=False,\n"
    "            return_bias=False,\n"
    "            quant_config=quant_config,\n"
    '            prefix=f"{{prefix}}.{SHORT}",\n'
    "        )\n"
)

base = "/opt/env/lib/python3.12/site-packages/vllm/models/deepseek_v4"
files = [f"{base}/nvidia/mtp.py", f"{base}/amd/mtp.py"]

for p in files:
    s = open(p).read()
    for name, short in (("        self.e_proj", "e_proj"), ("        self.h_proj", "h_proj")):
        old = E_OLD.format(NAME=name)
        new = E_NEW.format(NAME=name, SHORT=short)
        cnt = s.count(old)
        assert cnt == 1, f"#44821 {p} {short} anchor count = {cnt} (expected 1)"
        s = s.replace(old, new)
    open(p, "w").write(s)
    print(f"patched for #44821 OK: {p}")
