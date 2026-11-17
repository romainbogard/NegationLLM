#!/usr/bin/env bash
#
# NegLLM — environment bootstrapper
#
# Creates the Python virtual environment, installs the dependency stack, verifies it,
# and registers a Jupyter kernel. No dependencies of its own: it has to run *before*
# there is an environment, so it is plain POSIX-ish bash (macOS ships bash 3.2, so no
# associative arrays, no mapfile, no ${var,,}).
#
#   ./setup/setup.sh                 create .venv and install everything
#   ./setup/setup.sh --force         delete and recreate the environment
#   ./setup/setup.sh --no-jupyter    skip the Jupyter kernel registration
#   ./setup/setup.sh --ascii         plain ASCII output (no unicode box drawing)
#   ./setup/setup.sh --help
#
# The script lives in setup/ but operates on the repository root: the environment belongs next to
# code/, not inside setup/. It can therefore be invoked from anywhere.
#
set -uo pipefail

PROJECT="NegationLLM"
TAGLINE="the geometry of 'not' - experiment environment"   # ASCII only: --ascii targets non-UTF8 terminals
VENV_DIR=".venv"          # conventional hidden directory (tools and .gitignore expect it)
VENV_PROMPT="negllm"      # what the shell prompt shows once activated: (negllm)
REQ_FILE=""               # defaults to <script dir>/requirements.txt, resolved after arg parsing
KERNEL_NAME="negllm"
KERNEL_LABEL="Python (NegationLLM)"
PY_MIN_MAJOR=3
PY_MIN_MINOR=9

ORIG_PWD="$PWD"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" || exit 1
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)" || exit 1
cd "$PROJECT_ROOT" || exit 1
LOG="$PROJECT_ROOT/.setup.log"

FORCE=0
WITH_JUPYTER=1
USE_UNICODE=1
STEP_TOTAL=6
STEP_NUM=0

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
${PROJECT} setup

usage: ./setup/setup.sh [options]

options:
  --force            remove an existing ${VENV_DIR} and rebuild it from scratch
  --no-jupyter       do not register a Jupyter kernel
  --requirements F   use F instead of setup/requirements.txt
  --ascii            disable unicode glyphs (for terminals that mangle them)
  -h, --help         show this message

The environment is created at the repository root, whichever directory you run this from.

After it finishes:
  source ${VENV_DIR}/bin/activate
  jupyter lab code/
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --force)        FORCE=1 ;;
    --no-jupyter)   WITH_JUPYTER=0; STEP_TOTAL=5 ;;
    --ascii)        USE_UNICODE=0 ;;
    --requirements) shift; REQ_FILE="${1:-}" ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "unknown option: $1"; echo; usage; exit 2 ;;
  esac
  shift
done

# A --requirements path is relative to where the user stood; the default sits beside this script.
if [ -z "$REQ_FILE" ]; then
  REQ_FILE="$SCRIPT_DIR/requirements.txt"
else
  case "$REQ_FILE" in /*) ;; *) REQ_FILE="$ORIG_PWD/$REQ_FILE" ;; esac
fi
REQ_LABEL="${REQ_FILE#"$PROJECT_ROOT"/}"

# ---------------------------------------------------------------------------
# presentation layer
# ---------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
  ESC=$(printf '\033')
  BOLD="${ESC}[1m"; DIM="${ESC}[2m"; RESET="${ESC}[0m"
  RED="${ESC}[31m"; GREEN="${ESC}[32m"; YELLOW="${ESC}[33m"
  BLUE="${ESC}[34m"; MAGENTA="${ESC}[35m"; CYAN="${ESC}[36m"; GREY="${ESC}[90m"
  IS_TTY=1
else
  BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""
  BLUE=""; MAGENTA=""; CYAN=""; GREY=""
  IS_TTY=0
fi

case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
  *UTF-8*|*utf8*|*UTF8*) : ;;
  *) USE_UNICODE=0 ;;
esac

if [ "$USE_UNICODE" -eq 1 ]; then
  TL="╭"; TR="╮"; BL="╰"; BR="╯"; HZ="─"; VT="│"
  OK="✔"; BAD="✖"; ARROW="›"; DOT="•"; BAR_FULL="█"; BAR_EMPTY="░"; ELL="…"
  # An ARRAY, not a space-separated string: the dependency loop sets IFS to newline, which would
  # stop word-splitting from separating the frames and print them all on one line.
  SPIN_FRAMES=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
else
  TL="+"; TR="+"; BL="+"; BR="+"; HZ="-"; VT="|"
  OK="OK"; BAD="XX"; ARROW=">"; DOT="*"; BAR_FULL="#"; BAR_EMPTY="."; ELL="..."
  SPIN_FRAMES=("|" "/" "-" "\\")
fi

WIDTH=64

repeat() { # $1 char, $2 count
  local out="" i=0
  while [ "$i" -lt "$2" ]; do out="$out$1"; i=$((i + 1)); done
  printf '%s' "$out"
}

# Visible width: strip ANSI, then count CHARACTERS. bash's ${#var} is locale-aware and counts
# characters, whereas awk's length() counts bytes on macOS — which silently misaligns every box
# line containing a glyph like ✔ (3 bytes, 1 column).
vlen() {
  local stripped
  stripped=$(printf '%s' "$1" | sed "s/$(printf '\033')\[[0-9;]*m//g")
  printf '%s' "${#stripped}"
}

box_top()    { printf '%s%s%s%s%s\n' "$GREY" "$TL" "$(repeat "$HZ" $((WIDTH - 2)))" "$TR" "$RESET"; }
box_bottom() { printf '%s%s%s%s%s\n' "$GREY" "$BL" "$(repeat "$HZ" $((WIDTH - 2)))" "$BR" "$RESET"; }
box_line() { # $1 content (may contain ANSI)
  local content="$1"
  local vis maxw=$((WIDTH - 3))
  vis=$(vlen "$content")
  if [ "$vis" -gt "$maxw" ]; then   # never let over-long content break the frame
    local plain
    plain=$(printf '%s' "$content" | sed "s/$(printf '\033')\[[0-9;]*m//g")
    content="${plain:0:$((maxw - ${#ELL}))}${ELL}"
    vis=$(vlen "$content")
  fi
  local pad=$((maxw - vis))
  [ "$pad" -lt 0 ] && pad=0
  printf '%s%s%s %s%s%s%s\n' "$GREY" "$VT" "$RESET" "$content" "$(repeat ' ' "$pad")" "$GREY$VT" "$RESET"
}

shorten_path() { # $1 path, $2 max characters — keeps the informative tail
  local p="$1" max="$2" n
  case "$p" in "$HOME"/*) p="~${p#"$HOME"}" ;; esac
  n=${#p}
  [ "$n" -gt "$max" ] && p="${ELL}${p:$((n - max + ${#ELL}))}"
  printf '%s' "$p"
}
box_sep() { printf '%s%s%s%s%s\n' "$GREY" "$VT" "$(repeat "$HZ" $((WIDTH - 2)))" "$VT" "$RESET"; }

banner() {
  printf '\n'
  box_top
  box_line "${BOLD}${MAGENTA}${PROJECT}${RESET}${GREY} ${DOT} ${RESET}${DIM}${TAGLINE}${RESET}"
  box_sep
  box_line "${DIM}python${RESET}  ${PY_LABEL:-?}"
  box_line "${DIM}target${RESET}  $(shorten_path "${PWD}/${VENV_DIR}" 52)"
  box_line "${DIM}log${RESET}     $(shorten_path "$LOG" 52)"
  box_bottom
  printf '\n'
}

step() { # $1 label
  STEP_NUM=$((STEP_NUM + 1))
  STEP_LABEL="$1"
  printf '%s[%d/%d]%s %s%s%s\n' "$GREY" "$STEP_NUM" "$STEP_TOTAL" "$RESET" "$BOLD" "$1" "$RESET"
}

say()  { printf '      %s%s%s\n' "$DIM" "$1" "$RESET"; }
good() { printf '      %s%s%s %s\n' "$GREEN" "$OK" "$RESET" "$1"; }
warn() { printf '      %s%s%s %s\n' "$YELLOW" "!" "$RESET" "$1"; }
fail() { printf '      %s%s%s %s\n' "$RED" "$BAD" "$RESET" "$1"; }

cursor_hide() { [ "$IS_TTY" -eq 1 ] && printf '%s' "$(tput civis 2>/dev/null)"; }
cursor_show() { [ "$IS_TTY" -eq 1 ] && printf '%s' "$(tput cnorm 2>/dev/null)"; }

cleanup() { cursor_show; }
trap cleanup EXIT
trap 'cursor_show; printf "\n%s%s interrupted%s\n" "$YELLOW" "$BAD" "$RESET"; exit 130' INT

spin_wait() { # $1 pid, $2 label — animates until the pid exits, returns its status
  local pid="$1" label="$2" rc=0
  if [ "$IS_TTY" -eq 0 ]; then
    wait "$pid" || rc=$?
    return $rc
  fi
  cursor_hide
  while kill -0 "$pid" 2>/dev/null; do
    for f in "${SPIN_FRAMES[@]}"; do
      kill -0 "$pid" 2>/dev/null || break
      printf '\r      %s%s%s %s' "$CYAN" "$f" "$RESET" "$label"
      sleep 0.08
    done
  done
  wait "$pid" || rc=$?
  printf '\r%s\r' "$(repeat ' ' $((WIDTH + 12)))"
  cursor_show
  return $rc
}

progress_bar() { # $1 current, $2 total, $3 label
  local cur="$1" total="$2" label="$3" width=26
  [ "$total" -le 0 ] && total=1
  local filled=$((cur * width / total))
  local pct=$((cur * 100 / total))
  [ "$IS_TTY" -eq 0 ] && return 0
  printf '\r      %s%s%s%s %s%3d%%%s  %s%-26.26s%s' \
    "$CYAN" "$(repeat "$BAR_FULL" "$filled")" "$GREY" "$(repeat "$BAR_EMPTY" $((width - filled)))" \
    "$BOLD" "$pct" "$RESET" "$DIM" "$label" "$RESET"
}

die() { # $1 message
  printf '\n'
  box_top
  box_line "${RED}${BOLD}${BAD} setup failed${RESET}"
  box_sep
  box_line "$1"
  box_line "${DIM}last lines of ${LOG}:${RESET}"
  box_bottom
  printf '\n'
  [ -f "$LOG" ] && tail -n 12 "$LOG" | sed "s/^/      ${GREY}/;s/$/${RESET}/"
  printf '\n'
  exit 1
}

# ---------------------------------------------------------------------------
# 1. preflight
# ---------------------------------------------------------------------------
: > "$LOG"

PY_BIN=""
for candidate in python3.12 python3.11 python3.13 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver=$("$candidate" -c 'import sys; print("%d %d"%sys.version_info[:2])' 2>/dev/null) || continue
    maj=${ver% *}; min=${ver#* }
    if [ "$maj" -gt "$PY_MIN_MAJOR" ] || { [ "$maj" -eq "$PY_MIN_MAJOR" ] && [ "$min" -ge "$PY_MIN_MINOR" ]; }; then
      PY_BIN="$candidate"
      PY_LABEL="$("$candidate" -c 'import sys,platform; print("%s  (%s %s)"%(platform.python_version(), platform.system(), platform.machine()))')"
      break
    fi
  fi
done

banner

step "Preflight checks"
if [ -z "$PY_BIN" ]; then
  fail "no python >= ${PY_MIN_MAJOR}.${PY_MIN_MINOR} found on PATH"
  die "Install Python ${PY_MIN_MAJOR}.${PY_MIN_MINOR}+ (e.g. 'brew install python@3.12') and re-run."
fi
good "interpreter  ${BOLD}${PY_BIN}${RESET} ${DIM}${PY_LABEL}${RESET}"

if [ ! -f "$REQ_FILE" ]; then
  fail "missing ${REQ_LABEL}"
  die "Expected the dependency list at ${REQ_FILE}."
fi
PKG_LIST=$(grep -v '^[[:space:]]*#' "$REQ_FILE" | grep -v '^[[:space:]]*$' | sed 's/[[:space:]]*#.*$//' | sed 's/[[:space:]]*$//')
PKG_COUNT=$(printf '%s\n' "$PKG_LIST" | grep -c . || true)
good "requirements ${BOLD}${PKG_COUNT}${RESET} package(s) from ${DIM}${REQ_LABEL}${RESET}"

if "$PY_BIN" -c 'import venv' >/dev/null 2>&1; then
  good "venv module  available"
else
  fail "the 'venv' module is unavailable"
  die "On Debian/Ubuntu: sudo apt install python3-venv"
fi
printf '\n'

# ---------------------------------------------------------------------------
# 2. virtual environment
# ---------------------------------------------------------------------------
step "Virtual environment"
if [ -d "$VENV_DIR" ] && [ "$FORCE" -eq 1 ]; then
  say "removing existing ${VENV_DIR} (--force)"
  rm -rf "$VENV_DIR"
fi

if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
  good "reusing existing ${BOLD}${VENV_DIR}${RESET} ${DIM}(use --force to rebuild)${RESET}"
else
  ("$PY_BIN" -m venv --prompt "$VENV_PROMPT" "$VENV_DIR") >>"$LOG" 2>&1 &
  spin_wait $! "creating ${VENV_DIR} ..." || die "Could not create the virtual environment."
  good "created ${BOLD}${VENV_DIR}${RESET} ${DIM}(prompt: ${VENV_PROMPT})${RESET}"
fi

VPY="$VENV_DIR/bin/python"
VPIP="$VENV_DIR/bin/pip"
[ -x "$VPY" ] || die "The environment looks broken: ${VPY} is missing."
printf '\n'

# ---------------------------------------------------------------------------
# 3. pip
# ---------------------------------------------------------------------------
step "Package manager"
("$VPY" -m pip install --upgrade pip setuptools wheel) >>"$LOG" 2>&1 &
spin_wait $! "upgrading pip, setuptools, wheel ..." || die "Could not upgrade pip."
good "pip $("$VPIP" --version 2>/dev/null | awk '{print $2}') ready"
printf '\n'

# ---------------------------------------------------------------------------
# 4. dependencies
# ---------------------------------------------------------------------------
step "Dependencies"
say "installing one at a time so a failure points at the culprit"
IDX=0
FAILED=""
OLDIFS="$IFS"; IFS='
'
for spec in $PKG_LIST; do
  IDX=$((IDX + 1))
  name=$(printf '%s' "$spec" | sed 's/[<>=!~;[].*$//' | sed 's/[[:space:]]*$//')
  progress_bar "$((IDX - 1))" "$PKG_COUNT" "$name"
  ("$VPY" -m pip install "$spec") >>"$LOG" 2>&1 &
  if spin_wait $! "installing ${name} ($IDX/$PKG_COUNT) ..."; then
    :
  else
    FAILED="$FAILED $name"
  fi
  progress_bar "$IDX" "$PKG_COUNT" "$name"
done
IFS="$OLDIFS"
[ "$IS_TTY" -eq 1 ] && printf '\r%s\r' "$(repeat ' ' $((WIDTH + 12)))"

if [ -n "$FAILED" ]; then
  fail "failed:${FAILED}"
  die "Some packages could not be installed."
fi
good "all ${BOLD}${PKG_COUNT}${RESET} packages installed"
printf '\n'

# ---------------------------------------------------------------------------
# 5. verification
# ---------------------------------------------------------------------------
step "Verification"
VERIFY=$("$VPY" - "$REQ_FILE" <<'PY' 2>>"$LOG"
import importlib, importlib.metadata as md, re, sys, pathlib

IMPORT_NAME = {"scikit-learn": "sklearn", "pyarrow": "pyarrow", "jupyterlab": "jupyterlab"}
req = pathlib.Path(sys.argv[1])
lines = [l.split("#")[0].strip() for l in req.read_text().splitlines()]
names = [re.split(r"[<>=!~;\[]", l)[0].strip() for l in lines if l]

for n in names:
    mod = IMPORT_NAME.get(n, n.replace("-", "_"))
    try:
        importlib.import_module(mod)
        ok = "ok"
    except Exception:
        ok = "fail"
    try:
        ver = md.version(n)
    except Exception:
        ver = "?"
    print(f"{ok}\t{n}\t{ver}")

device = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
except Exception:
    device = "n/a"
print(f"device\t{device}\t")
PY
) || true

DEVICE="n/a"
BAD_IMPORTS=0
OLDIFS="$IFS"; IFS='
'
for line in $VERIFY; do
  status=$(printf '%s' "$line" | cut -f1)
  pname=$(printf '%s' "$line" | cut -f2)
  pver=$(printf '%s' "$line" | cut -f3)
  case "$status" in
    ok)     printf '      %s%s%s %-16s %s%s%s\n' "$GREEN" "$OK" "$RESET" "$pname" "$DIM" "$pver" "$RESET" ;;
    fail)   printf '      %s%s%s %-16s %s%s%s\n' "$RED" "$BAD" "$RESET" "$pname" "$DIM" "import failed" "$RESET"
            BAD_IMPORTS=$((BAD_IMPORTS + 1)) ;;
    device) DEVICE="$pname" ;;
  esac
done
IFS="$OLDIFS"
[ "$BAD_IMPORTS" -gt 0 ] && die "${BAD_IMPORTS} package(s) installed but do not import."
printf '\n'

# ---------------------------------------------------------------------------
# 6. jupyter kernel
# ---------------------------------------------------------------------------
if [ "$WITH_JUPYTER" -eq 1 ]; then
  step "Jupyter kernel"
  ("$VPY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_LABEL") >>"$LOG" 2>&1 &
  if spin_wait $! "registering kernel '${KERNEL_LABEL}' ..."; then
    good "kernel ${BOLD}${KERNEL_LABEL}${RESET} registered"
  else
    warn "kernel registration failed (not fatal) — see ${LOG}"
  fi
  printf '\n'
fi

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
ELAPSED=$SECONDS
MINS=$((ELAPSED / 60)); SECS=$((ELAPSED % 60))

case "$DEVICE" in
  mps)  DEV_NOTE="${GREEN}mps${RESET} ${DIM}(Apple Silicon GPU)${RESET}" ;;
  cuda) DEV_NOTE="${GREEN}cuda${RESET} ${DIM}(NVIDIA GPU)${RESET}" ;;
  cpu)  DEV_NOTE="${YELLOW}cpu${RESET} ${DIM}(no GPU detected — fine, the notebooks are small)${RESET}" ;;
  *)    DEV_NOTE="${DIM}${DEVICE}${RESET}" ;;
esac

box_top
box_line "${GREEN}${BOLD}${OK} environment ready${RESET}${GREY}  ${DOT}  ${RESET}${DIM}${MINS}m${SECS}s${RESET}"
box_sep
box_line "${DIM}environment${RESET}    ${BOLD}${VENV_PROMPT}${RESET} ${DIM}(${VENV_DIR})${RESET}"
box_line "${DIM}torch device${RESET}   ${DEV_NOTE}"
box_line "${DIM}packages${RESET}       ${PKG_COUNT} installed"
[ "$WITH_JUPYTER" -eq 1 ] && box_line "${DIM}kernel${RESET}         ${KERNEL_LABEL}"
box_sep
box_line "${BOLD}next${RESET}"
box_line "  ${CYAN}${ARROW}${RESET} source ${VENV_DIR}/bin/activate"
box_line "  ${CYAN}${ARROW}${RESET} jupyter lab code/"
box_sep
box_line "${DIM}00_load_nevir.ipynb${RESET}          fetch NevIR ${ARROW} parquet"
box_line "${DIM}01_negation_subspace_rank.ipynb${RESET}  the rank experiment"
box_bottom
printf '\n'
