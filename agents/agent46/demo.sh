#!/usr/bin/env bash
# Demo helper for the Regression Court on Atlas.
# Loads the Atlas settings from .env itself (so it works from any fresh terminal) and never prints credentials.
#
#   ./demo.sh court                     start the Regression Court against Atlas on http://127.0.0.1:8766
#   ./demo.sh seed                      3 Nominated + 2 Filed demo cases (adapts to the live policy)
#   ./demo.sh run parts-theft [pace]    narrated simulated incident stream; leaves a proposed version to approve
#   ./demo.sh run abandoned-lookalike   a plausible fix that the safety check rejects
#   ./demo.sh status                    what the demo created, which policy is live, any candidate waiting
#   ./demo.sh reset                     remove only the simulated data
#
# Scenarios: parts-theft, abandoned-lookalike (staged-accident and cloned-vehicle are already fixed in v8).
# Set DEMO_BY to change the name recorded in the audit log (default PAVAN).
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "demo.sh: .env not found in $(pwd)" >&2; exit 1; }
get() { grep "^$1=" .env | head -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"; }
URI="$(get ATLAS_EVOLVE_URI)"; DB="$(get ATLAS_EVOLVE_DB)"
if [ -z "$URI" ] || [ -z "$DB" ]; then
  echo "demo.sh: ATLAS_EVOLVE_URI and ATLAS_EVOLVE_DB must both be set in .env" >&2; exit 1
fi

export EVOLVE_MONGODB_URI="$URI"
export EVOLVE_DOCKET_PATH="$PWD/.evolve_cache/atlas_docket.json"
mkdir -p .evolve_cache
if [ ! -f "$EVOLVE_DOCKET_PATH" ]; then
  if [ -f evals/scenarios/docket.json ]; then cp evals/scenarios/docket.json "$EVOLVE_DOCKET_PATH"; else echo '[]' > "$EVOLVE_DOCKET_PATH"; fi
fi
BY="${DEMO_BY:-PAVAN}"
UI="${DEMO_UI_URL:-http://127.0.0.1:8766}"

cli() { uv run python -m harness_evolve.cli --allow-remote --by "$BY" --db "$DB" "$@" 2>&1 | sed -E 's#mongodb(\+srv)?://[^ ]*#<uri>#g'; }

case "${1:-help}" in
  court)  exec uv run python -m harness_evolve.court_ui --allow-remote --db "$DB" --port 8766 ;;
  seed)   cli demo seed ;;
  run)    cli demo run --scenario "${2:?usage: ./demo.sh run <scenario> [pace]}" --pace "${3:-2}" --ui-url "$UI" ;;
  status) cli demo status ;;
  reset)  cli demo reset --yes ;;
  *)      sed -n '2,13p' "$0" ;;
esac
