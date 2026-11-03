#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim launcher: override with `export ISAAC_PYTHON=/path/to/isaacsim/python.sh`.
PYTHON_SH="${ISAAC_PYTHON:-isaacsim/python.sh}"
# Bundle root is the directory containing this script (portable, no hardcoding).
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLAY_SCRIPT="${BUNDLE_ROOT}/scripts/replay_whip_traj_wire_end_calibration.py"

TRAJ_CSV="${BUNDLE_ROOT}/data/trajectory/whip_traj_medium.csv"
REF_CSV="${BUNDLE_ROOT}/data/reference/whipping_medium_1_001_stacked_transformed.csv"
ENGINE_CFG="${BUNDLE_ROOT}/config/rho700_E220000.json"
RUN_NAME="whip_medium_$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${BUNDLE_ROOT}/outputs/runs"
VIDEO_FPS=60
HEADLESS=0
MAKE_VIDEO=1
PHYSICS_GPU=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_replay_whip_medium.sh [options] [extra replay args...]

Options:
  --traj-csv PATH      Override trajectory CSV
  --ref-csv PATH       Override reference CSV
  --engine-cfg PATH    Override CoSim engine config JSON
  --run-name NAME      Run folder name (default: whip_medium_<timestamp>)
  --runs-root DIR      Parent directory for run outputs
  --video-fps N        Video frame rate (default: 60)
  --headless           Run Isaac Sim headless
  --physics-gpu N      Pass --physics_gpu N to replay script
  --no-video           Skip --make_video
  -h, --help           Show this help

Examples:
  ./run_replay_whip_medium.sh
  ./run_replay_whip_medium.sh --headless --video-fps 30
  ./run_replay_whip_medium.sh --run-name test_run --physics-gpu 0 -- --max_frames 3000
  ./run_replay_whip_medium.sh --engine-cfg /path/to/other_engine_cfg.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --traj-csv)
      TRAJ_CSV="$2"
      shift 2
      ;;
    --ref-csv)
      REF_CSV="$2"
      shift 2
      ;;
    --engine-cfg)
      ENGINE_CFG="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --runs-root)
      RUNS_ROOT="$2"
      shift 2
      ;;
    --video-fps)
      VIDEO_FPS="$2"
      shift 2
      ;;
    --headless)
      HEADLESS=1
      shift
      ;;
    --physics-gpu)
      PHYSICS_GPU="$2"
      shift 2
      ;;
    --no-video)
      MAKE_VIDEO=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "${PYTHON_SH}" ]]; then
  echo "Error: Isaac Sim launcher not executable: ${PYTHON_SH}" >&2
  exit 1
fi
if [[ ! -f "${REPLAY_SCRIPT}" ]]; then
  echo "Error: Replay script not found: ${REPLAY_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${TRAJ_CSV}" ]]; then
  echo "Error: Trajectory CSV not found: ${TRAJ_CSV}" >&2
  exit 1
fi
if [[ ! -f "${REF_CSV}" ]]; then
  echo "Error: Reference CSV not found: ${REF_CSV}" >&2
  exit 1
fi
if [[ ! -f "${ENGINE_CFG}" ]]; then
  echo "Error: Engine config JSON not found: ${ENGINE_CFG}" >&2
  exit 1
fi

RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
RAW_DIR="${RUN_DIR}/raw"
PLOTS_DIR="${RUN_DIR}/plots"
COMPARE_DIR="${RUN_DIR}/comparison"
VIDEOS_DIR="${RUN_DIR}/videos"
LOGS_DIR="${RUN_DIR}/logs"

mkdir -p "${RAW_DIR}" "${PLOTS_DIR}" "${COMPARE_DIR}" "${VIDEOS_DIR}" "${LOGS_DIR}"

OUT_CSV="${RAW_DIR}/whip_wire_end_positions.csv"
OUT_PLOT="${PLOTS_DIR}/whip_wire_end_trajectory.png"
COMPARE_OUT_CSV="${COMPARE_DIR}/whip_reference_comparison.csv"
OUT_VIDEO="${VIDEOS_DIR}/whip_wire_end_trajectory.mp4"
LOG_FILE="${LOGS_DIR}/replay.log"

CMD=(
  "${PYTHON_SH}"
  "${REPLAY_SCRIPT}"
  "--traj_csv" "${TRAJ_CSV}"
  "--ref_csv" "${REF_CSV}"
  "--engine_cfg" "${ENGINE_CFG}"
  "--out_csv" "${OUT_CSV}"
  "--out_plot" "${OUT_PLOT}"
  "--compare_out_csv" "${COMPARE_OUT_CSV}"
  "--out_video" "${OUT_VIDEO}"
  "--video_fps" "${VIDEO_FPS}"
)

if [[ "${MAKE_VIDEO}" -eq 1 ]]; then
  CMD+=("--make_video")
fi
if [[ "${HEADLESS}" -eq 1 ]]; then
  CMD+=("--headless")
fi
if [[ -n "${PHYSICS_GPU}" ]]; then
  CMD+=("--physics_gpu" "${PHYSICS_GPU}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Run directory: ${RUN_DIR}"
echo "Trajectory CSV: ${TRAJ_CSV}"
echo "Reference CSV : ${REF_CSV}"
echo "Engine config : ${ENGINE_CFG}"
echo "Output CSV    : ${OUT_CSV}"
echo "Output plot   : ${OUT_PLOT}"
echo "Compare CSV   : ${COMPARE_OUT_CSV}"
echo "Output video  : ${OUT_VIDEO}"
echo "Log file      : ${LOG_FILE}"
echo

{
  echo "[run] $(date -Iseconds)"
  echo "[pwd] $(pwd)"
  printf '[cmd]'
  printf ' %q' "${CMD[@]}"
  echo
  "${CMD[@]}"
} 2>&1 | tee "${LOG_FILE}"

status=${PIPESTATUS[0]}
if [[ ${status} -ne 0 ]]; then
  echo "Replay failed with exit code ${status}. See log: ${LOG_FILE}" >&2
  exit "${status}"
fi

echo "Replay completed successfully."
echo "Outputs are under: ${RUN_DIR}"
