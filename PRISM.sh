#!/usr/bin/env bash
# PRISM launcher.
#
# Resolves a usable Python interpreter, activates a project-local
# virtualenv if one is present, then runs ``python -m PRISM``.
# Forwards any extra arguments to the Python entry point.
#
# Works under bash on Linux / macOS / Git Bash / WSL / MSYS2.

set -euo pipefail

# Resolve the directory containing this script so PRISM can be launched
# from anywhere (and so symlinks to PRISM.sh keep working).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Activate a project-local virtualenv if one exists.  We look in the
# usual places without imposing a particular convention.
for venv_path in ".venv" "venv" ".env"; do
    if [[ -f "${venv_path}/bin/activate" ]]; then
        # POSIX-style venv (Linux / macOS / WSL).
        # shellcheck disable=SC1090
        source "${venv_path}/bin/activate"
        break
    fi
    if [[ -f "${venv_path}/Scripts/activate" ]]; then
        # Windows-style venv used by Git Bash / MSYS2.
        # shellcheck disable=SC1090
        source "${venv_path}/Scripts/activate"
        break
    fi
done

# Resolve a Python interpreter.  We prefer interpreters that reliably
# expose PyQt6: ``py -3.10`` on Windows tends to match what users have
# installed for this project, then fall back to plain ``python3`` /
# ``python`` for POSIX environments.
PYTHON=""
if command -v py >/dev/null 2>&1; then
    if py -3.10 -c "import PyQt6" >/dev/null 2>&1; then
        PYTHON="py -3.10"
    elif py -c "import PyQt6" >/dev/null 2>&1; then
        PYTHON="py"
    fi
fi

if [[ -z "${PYTHON}" ]]; then
    for cand in python3 python; do
        if command -v "${cand}" >/dev/null 2>&1; then
            if "${cand}" -c "import PyQt6" >/dev/null 2>&1; then
                PYTHON="${cand}"
                break
            fi
        fi
    done
fi

if [[ -z "${PYTHON}" ]]; then
    cat >&2 <<'EOF'
PRISM: could not find a Python interpreter with PyQt6 installed.

Install the runtime dependencies, e.g.:

    python -m pip install -r requirements.txt

Then re-run ./PRISM.sh.
EOF
    exit 1
fi

exec ${PYTHON} -m PRISM "$@"
