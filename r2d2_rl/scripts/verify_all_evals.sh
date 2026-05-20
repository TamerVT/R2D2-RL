#!/usr/bin/env bash
# Pre-demo sanity: run Eval 1, 2, 3 in the sim using the SCRIPTED align_grasp
# policy (no learned checkpoint required), confirm each one returns successes
# matching its goal count, and dump the summary JSONs.
#
# Use this before plugging in --sb3-align-grasp-checkpoint <path> to confirm
# the hybrid pipeline itself is healthy.

set -uo pipefail

cd "$(dirname "$0")/../.."   # land in project3/

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not on PATH"
    exit 2
fi

# shellcheck disable=SC1091
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-p3-rcs

export MUJOCO_GL=egl

OUT_ROOT="r2d2_rl/outputs/verify_all_evals_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT_ROOT"

OVERALL_EXIT=0
for EVAL in eval1 eval2 eval3; do
    OUT="$OUT_ROOT/$EVAL"
    echo
    echo "=================================================="
    echo " Running $EVAL"
    echo "=================================================="
    python r2d2_rl/scripts/run_eval_sequence.py \
        --config "r2d2_rl/configs/hybrid_control_rl/$EVAL.yaml" \
        --output-dir "$OUT" \
        --no-use-pregrasp-regressor \
        --save-images
    RC=$?
    if [ "$RC" -ne 0 ]; then
        echo "[$EVAL] FAILED (exit code $RC)"
        OVERALL_EXIT=1
    else
        echo "[$EVAL] PASS"
    fi
done

echo
echo "=================================================="
echo " Summary"
echo "=================================================="
for EVAL in eval1 eval2 eval3; do
    SUMMARY="$OUT_ROOT/$EVAL/summary.json"
    if [ -f "$SUMMARY" ]; then
        python -c "import json; d=json.load(open('$SUMMARY')); print(f'  $EVAL: {d[\"num_successes\"]}/{d[\"num_goals\"]}  all_pass={d[\"all_pass\"]}')"
    else
        echo "  $EVAL: NO SUMMARY"
    fi
done

if [ "$OVERALL_EXIT" -eq 0 ]; then
    echo "ALL EVALS PASS"
else
    echo "ONE OR MORE EVALS FAILED -- see $OUT_ROOT"
fi
exit "$OVERALL_EXIT"
