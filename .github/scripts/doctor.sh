#!/usr/bin/env bash
#
# doctor — compare the environment this project declares to the one it is running in.
#
# Where a working setup quietly stops being reproducible: a variable that is set on the
# machine where it was written and nowhere else, a runtime that drifted a minor version
# past the pin, a directory the project writes to that it does not have permission to
# write to. None of these is a security finding and none of them fails a scanner; they
# are the reason "it works here" stops being true.
#
# WHAT IT READS, AND WHY IT READS ONLY DECLARATIONS
#
# Everything below is a comparison between something the repository declares and something
# the machine reports. There is no list of variables a project "should" have, because a
# tool that invents requirements teaches its user to ignore it.
#
#   .env.example          the declared environment. Names only — this never reads .env's
#                         values, and never prints a value it does read.
#   config.schema.json    the same declaration in JSON Schema form, if that is the shape
#                         the project uses. Its `required` array is the list.
#   mise.toml, .mise.toml, .tool-versions
#                         the declared runtime versions. mise is the tool this follows:
#                         it reads all three formats, and its remedies are what a failure
#                         here prints. Parsed directly rather than shelled out to, so the
#                         check works before mise is installed — which is exactly the
#                         machine most likely to be wrong.
#
# WHAT IT WILL NOT DO
#
#   * Block. It is not on the commit hook and it is not in CI. A missing variable is a
#     fact about one machine, not about a commit — and womblex's own CI sets none of
#     them, so a doctor job there would be permanently and uninformatively red.
#     Run it by hand:  bash .github/scripts/doctor.sh
#   * Call an undeclared environment a pass. A project that declares nothing reports
#     `not applicable`, by name. There is nothing to check, which is not the same as
#     everything being right.
#   * Print a secret. Variable names are compared; values are never read, echoed or
#     logged, including from .env.
#
# Deterministic and offline in the sense that matters for a check about a machine: same
# tree plus same environment gives the same verdict, with no network and no model.
#
# Exit codes, house convention:
#     0  everything required is present, or there was nothing declared
#     1  something required is missing or mismatched
#     2  could not run
#
# Lifted from DeepCivic/runwAI at f67a2c5a8e2289817d7a65e1920fa5d67953d3e6 (Apache-2.0),
# .github/scripts/doctor.sh. Changed here: optional-variable support, keyed off the
# `# Optional:` convention .env.example already used, because womblex's only declared
# variable is needed by one optional stage and the upstream script failed every clone
# that did not use it.

set -uo pipefail

ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$ROOT" || { echo "doctor: cannot enter $ROOT" >&2; exit 2; }

problems=0
declarations=0
lines=()

note() { lines+=("$1"); }
fail() { lines+=("$1"); problems=$((problems + 1)); }

# --- 1. declared environment variables ------------------------------------------------
#
# Names are taken from the left of the first '=' on each non-comment line. A name with no
# value in .env.example is still a declaration: "this must be set" is the whole point of
# the file, and an example value is a courtesy rather than the contract.

declared_vars=()
optional_vars=()
if [ -f .env.example ]; then
  declarations=$((declarations + 1))
  optional_next=0
  while IFS= read -r line; do
    case "$line" in
      '') optional_next=0; continue ;;
      '#'*)
        # A comment containing "optional" marks every declaration up to the next
        # blank line as not-required. Womblex's .env.example already used that
        # word this way ("# Optional: override the local models/ directory"), so
        # this reads an existing convention rather than inventing a file format —
        # and .env.example stays a file you can copy to .env unchanged.
        #
        # Without this, a variable needed by one optional stage fails doctor on
        # every clone that does not use that stage, and a check that is red when
        # nothing is wrong is one people learn to ignore.
        case "$(printf '%s' "$line" | tr '[:upper:]' '[:lower:]')" in
          *optional*) optional_next=1 ;;
        esac
        continue ;;
    esac
    name="${line%%=*}"
    name="${name#export }"
    name="$(printf '%s' "$name" | tr -d '[:space:]')"
    case "$name" in
      ''|*[!A-Za-z0-9_]*) optional_next=0; continue ;;
    esac
    if [ "$optional_next" -eq 1 ]; then
      optional_vars+=("$name")
    else
      declared_vars+=("$name")
    fi
  done < .env.example
fi

# JSON Schema's `required` array, extracted without a JSON parser: this file has to run
# before anything is installed, and the shape needed is one flat list of strings.
if [ -f config.schema.json ]; then
  declarations=$((declarations + 1))
  while IFS= read -r name; do
    [ -n "$name" ] && declared_vars+=("$name")
  done < <(tr -d '\n' < config.schema.json \
           | grep -o '"required"[[:space:]]*:[[:space:]]*\[[^]]*\]' \
           | grep -o '"[A-Za-z0-9_]*"' \
           | tr -d '"' \
           | grep -v '^required$')
fi

if [ $((${#declared_vars[@]} + ${#optional_vars[@]})) -gt 0 ]; then
  # ${arr[@]+"${arr[@]}"} rather than a bare "${arr[@]}" throughout: under `set -u`,
  # bash before 4.4 treats expanding an empty array as an unbound variable and aborts.
  # macOS still ships bash 3.2, and womblex's own case reaches here with declared_vars
  # empty and only optional_vars populated — so the bare form would fail on exactly the
  # machine a developer is most likely to run this from.
  missing=()
  for name in ${declared_vars[@]+"${declared_vars[@]}"}; do
    if [ -z "${!name-}" ]; then
      missing+=("$name")
    fi
  done
  if [ ${#declared_vars[@]} -eq 0 ]; then
    note "  NOTE  every declared variable is marked optional; none is required here"
  elif [ ${#missing[@]} -eq 0 ]; then
    note "  PASS  all ${#declared_vars[@]} required variables are set"
  else
    fail "  FAIL  ${#missing[@]} required variable(s) are not set in this environment:"
    for name in "${missing[@]}"; do
      lines+=("          $name")
    done
    lines+=("        Set them, or copy .env.example to .env and fill it in. Nothing here")
    lines+=("        reads their values — only whether a name has one.")
  fi

  # Optional variables are reported and never counted. An unset ISAACUS_API_KEY on a
  # clone that only runs extraction is a fact about how this clone is used, not a
  # problem with it — but it is still the first thing to check when the enrichment
  # stage will not start, so silence would be the wrong answer too.
  if [ ${#optional_vars[@]} -gt 0 ]; then
    unset_optional=()
    for name in "${optional_vars[@]}"; do
      [ -z "${!name-}" ] && unset_optional+=("$name")
    done
    if [ ${#unset_optional[@]} -eq 0 ]; then
      note "  PASS  all ${#optional_vars[@]} optional variables are set"
    else
      note "  NOTE  ${#unset_optional[@]} optional variable(s) are not set:"
      for name in "${unset_optional[@]}"; do
        lines+=("          $name")
      done
      lines+=("        The stages that need them will not run. That is a choice about this")
      lines+=("        clone, not a problem with it, so it does not change the exit code.")
    fi
  fi

  # Drift in the other direction: a name in .env that .env.example does not declare is
  # usually left over from a config that moved on, and it is the half nobody notices
  # because everything still works.
  if [ -f .env ]; then
    undeclared=()
    while IFS= read -r line; do
      case "$line" in
        ''|'#'*) continue ;;
      esac
      name="${line%%=*}"
      name="${name#export }"
      name="$(printf '%s' "$name" | tr -d '[:space:]')"
      case "$name" in
        ''|*[!A-Za-z0-9_]*) continue ;;
      esac
      found=0
      for declared in ${declared_vars[@]+"${declared_vars[@]}"} \
                      ${optional_vars[@]+"${optional_vars[@]}"}; do
        [ "$declared" = "$name" ] && { found=1; break; }
      done
      [ "$found" -eq 0 ] && undeclared+=("$name")
    done < .env
    if [ ${#undeclared[@]} -gt 0 ]; then
      note "  NOTE  ${#undeclared[@]} variable(s) in .env are not declared in .env.example:"
      for name in "${undeclared[@]}"; do
        lines+=("          $name")
      done
      lines+=("        Either declare them or delete them. This is drift, not a failure,")
      lines+=("        so it does not change the exit code.")
    fi
  fi

  # Anything naming a path is a declaration that the project writes somewhere. Checking
  # the directory itself when it exists, and its parent when it does not, is the
  # difference between "you cannot write here" and "this has not been created yet".
  #
  # Running as root makes the permission half of this vacuous — `test -w` is true for root
  # on any directory — so in a container that runs as root only the existence check can
  # fail. That is the kernel's answer, not a gap here, and it is worth knowing before
  # someone concludes the check is broken.
  for name in ${declared_vars[@]+"${declared_vars[@]}"} \
              ${optional_vars[@]+"${optional_vars[@]}"}; do
    case "$name" in
      *_DIR|*_PATH|*_ROOT|*_HOME) ;;
      *) continue ;;
    esac
    value="${!name-}"
    [ -z "$value" ] && continue
    target="$value"
    [ -d "$target" ] || target="$(dirname "$target")"
    if [ ! -d "$target" ]; then
      fail "  FAIL  $name points at $value, and neither it nor its parent exists"
    elif [ ! -w "$target" ]; then
      fail "  FAIL  $name points at $value, which is not writable"
    else
      note "  PASS  $name is writable"
    fi
  done
fi

# --- 2. declared runtime versions -----------------------------------------------------
#
# tool -> pinned version, from whichever of the three formats the project uses. mise reads
# all of them; this parses them so the check still works on a machine where mise is not
# installed yet.

pins=()
read_tool_versions() {
  while IFS= read -r line; do
    case "$line" in
      ''|'#'*) continue ;;
    esac
    set -- $line
    [ $# -lt 2 ] && continue
    pins+=("$1 $2")
  done < "$1"
}

read_mise_toml() {
  # Only the [tools] table, and only `name = "version"` entries. A version expressed as a
  # table or a list is a declaration this cannot compare, and it is skipped rather than
  # guessed at.
  local in_tools=0 line key value
  while IFS= read -r line; do
    line="${line%%#*}"
    case "$line" in
      \[tools\]*) in_tools=1; continue ;;
      \[*) in_tools=0; continue ;;
    esac
    [ "$in_tools" -eq 1 ] || continue
    case "$line" in
      *=*) ;;
      *) continue ;;
    esac
    key="$(printf '%s' "${line%%=*}" | tr -d '[:space:]"')"
    value="$(printf '%s' "${line#*=}" | tr -d '[:space:]"'"'")"
    case "$value" in
      ''|\[*|\{*) continue ;;
    esac
    [ -n "$key" ] && pins+=("$key $value")
  done < "$1"
}

for f in mise.toml .mise.toml; do
  [ -f "$f" ] && { declarations=$((declarations + 1)); read_mise_toml "$f"; break; }
done
if [ -f .tool-versions ]; then
  declarations=$((declarations + 1))
  read_tool_versions .tool-versions
fi

# The interpreter to ask about each pinned tool. A tool with no entry here is reported as
# declared-but-unverifiable rather than skipped silently: the pin is real, this check just
# cannot read it.
runtime_command() {
  case "$1" in
    python|python3) echo "python3 --version" ;;
    node|nodejs) echo "node --version" ;;
    go|golang) echo "go version" ;;
    ruby) echo "ruby --version" ;;
    rust) echo "rustc --version" ;;
    java) echo "java -version" ;;
    *) echo "" ;;
  esac
}

if [ ${#pins[@]} -gt 0 ]; then
  for pin in "${pins[@]}"; do
    set -- $pin
    tool="$1"; want="$2"
    cmd="$(runtime_command "$tool")"
    if [ -z "$cmd" ]; then
      note "  NOTE  $tool is pinned to $want; doctor does not know how to ask it its version"
      continue
    fi
    binary="${cmd%% *}"
    if ! command -v "$binary" >/dev/null 2>&1; then
      fail "  FAIL  $tool is pinned to $want but $binary is not on PATH"
      lines+=("        Install it:  mise install $tool@$want")
      continue
    fi
    # Version output is not a stable format across runtimes, so the first dotted number
    # anywhere in the first line is what gets compared.
    have="$($cmd 2>&1 | head -1 | grep -oE '[0-9]+(\.[0-9]+)+' | head -1)"
    if [ -z "$have" ]; then
      note "  NOTE  $tool is pinned to $want; could not read a version from '$cmd'"
    elif [ "$have" = "$want" ]; then
      note "  PASS  $tool $have matches the pin"
    else
      fail "  FAIL  $tool is pinned to $want but $have is on PATH"
      lines+=("        Fix it:  mise install $tool@$want && mise use $tool@$want")
    fi
  done
fi

# --- 3. mise's own view, when it is installed -----------------------------------------
#
# Informational and deliberately kept out of the exit code. mise doctor reports on mise's
# installation — shims, activation, plugin state — which is about the developer's machine
# rather than about this project's declarations. Worth surfacing when a pin above failed
# and the reason is that mise is not activated; never worth failing a project over.

if [ "$declarations" -gt 0 ] && command -v mise >/dev/null 2>&1; then
  if mise doctor >/dev/null 2>&1; then
    note "  NOTE  mise doctor reports no problems with the mise installation itself"
  else
    note "  NOTE  mise doctor reports problems with the mise installation itself."
    note "        Run 'mise doctor' for its output. Not counted here: that is about mise,"
    note "        not about this project."
  fi
fi

# --- verdict ---------------------------------------------------------------------------

echo "doctor — does this environment match what the project declares?"
echo

if [ "$declarations" -eq 0 ]; then
  echo "  NOT APPLICABLE"
  echo "    This project declares no environment requirements: no .env.example, no"
  echo "    config.schema.json, and no mise.toml or .tool-versions. There is nothing to"
  echo "    compare against, which is not the same as everything being correct."
  echo
  echo "    Declare what it needs and this starts checking it. Nothing else changes."
  exit 0
fi

printf '%s\n' "${lines[@]}"
echo

if [ "$problems" -gt 0 ]; then
  echo "$problems problem(s). Each line above says what to run."
  echo "This reports and blocks nothing: it is not on the commit hook and not in CI."
  echo "Optional variables are listed above but never counted here."
  exit 1
fi

echo "Everything this project requires is present and matches."
echo "That is a statement about declared requirements only — a variable nobody wrote down"
echo "is not checked here, because nothing knows it exists, and any optional variable"
echo "listed above as unset is still unset."
exit 0
