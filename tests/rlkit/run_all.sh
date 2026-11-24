#!/bin/bash
# Runs every CPU test of the RL kit (LoRA merge kernel, factor layout, host path, lazy delta) and prints one line each.
# Needs a Python with nki 0.7.0 (CPU simulator), torch (CPU), safetensors and numpy; sglang is not needed.
#   PY=~/nki_sim_venv/bin/python bash tests/rlkit/run_all.sh        (RUN_8B=1 adds the Qwen3-8B shapes; slow)
cd "$(dirname "$0")" || exit 2
PY=${PY:-python3}
mkdir -p logs
fail=0
for t in test_merge_kernel_sim test_k_chunks test_factor_layout negative_control_factor_layout test_end_to_end_review \
         test_lazy_delta test_host_kernel_path; do
    timeout 3000 "$PY" "$t.py" > "logs/$t.log" 2>&1
    code=$?
    last=$(grep -v -i "warn" "logs/$t.log" | tail -1 | cut -c1-150)
    echo "$t: exit $code | $last"
    [ $code -ne 0 ] && fail=1
done
exit $fail
