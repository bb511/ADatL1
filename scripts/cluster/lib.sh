#!/bin/bash
# Helpers shared by submit_search.sh, submit_pareto.sh and submit_pareto_ngt.sh.
#
# The block parser below is the one piece that must stay in step with
# scripts/optuna/make_pareto_scripts.py: it defines what a "command" in a
# run*_{search,pareto}.sh file is. It used to be copy-pasted into all three
# submitters, which meant a change to the generated layout had to be made in
# three places or one submitter would silently stop finding commands.
#
# Source it as:  . "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Where the training env lives on clariden; quoted in the preflight message.
CLARIDEN_ENV=${CLARIDEN_ENV:-/users/podagiu/.conda/envs/adl1t}

# A block opens at the commented 'python3 src/train.py' line and runs to the
# first line without a trailing continuation. Emits one flat command per block,
# each with a leading space so callers can concatenate directly.
extract_cmds() {
    awk '
        /^# python3 src\/train\.py/ { collecting = 1; cmd = ""; next }
        collecting {
            line = $0
            sub(/^# ?/, "", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            ends = (line ~ /\\$/)
            sub(/[ \t]*\\$/, "", line)
            if (line != "") cmd = cmd " " line
            if (!ends) { collecting = 0; print cmd }
        }
    ' "$1"
}

# The value of an override, e.g. field "$cmd" 'run_name' or
# field "$cmd" 'hydra\.sweeper\.study_name'. Empty when the command has none.
field() { printf "%s" "$1" | grep -oE "$2=[^ ]+" | head -1 | cut -d= -f2- || true; }

# $1 = 1-based index, $2 = name. Matches against the caller's $only, a
# comma-separated list of name substrings or indices; an empty $only selects
# everything. Written with explicit ifs rather than 'test && action': under
# 'set -e' a bare failing && list aborts the script.
selected() {
    if [ -z "$only" ]; then return 0; fi
    local idx="$1" name="$2" s
    local IFS=','
    for s in $only; do
        if [ -n "$s" ]; then
            case "$s" in
                # An all-digit selector is an index only. Matching it as a
                # substring too would make '--only 1' also pick *_t1, *_t10, ...
                *[!0-9]*) case "$name" in *"$s"*) return 0 ;; esac ;;
                *) if [ "$s" = "$idx" ]; then return 0; fi ;;
            esac
        fi
    done
    return 1
}

# Every command is relative to the repository root, and the log directories are
# created there too, so refuse to run from anywhere else.
require_repo_root() {
    [ -f .project-root ] || { echo "run this from the repository root" >&2; exit 1; }
}

# The generated blocks carry a literal '/path/to/adl1t_data/parquet_files'
# placeholder that each submitter rewrites with the real location. Catching a
# survivor here is what stops a whole batch from failing one job at a time.
# $1 = assembled command, $2 = label for the message, $3 = how to fix it.
assert_no_placeholder() {
    case "$1" in
        *"/path/to/"*)
            echo "error: $2 still contains a '/path/to/...' placeholder." >&2
            [ -z "${3:-}" ] || echo "       $3" >&2
            exit 1 ;;
    esac
}

# Preflight: the drivers are nohup'd, so an unimportable dependency shows up
# only as a tiny log per driver and nothing ever reaches the scheduler.
# Forgetting to activate the env once cost 24 silent failures on the search
# side, so it is checked up front. $1 = python, $2 = modules, rest = hint lines.
require_imports() {
    local py="$1" mods="$2" line
    shift 2
    "$py" -c "import $mods" >/dev/null 2>&1 && return 0
    echo "error: '$py' cannot import $mods - the training env is not active." >&2
    for line in "$@"; do echo "       $line" >&2; done
    exit 1
}
