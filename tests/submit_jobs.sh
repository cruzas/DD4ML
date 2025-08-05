#!/bin/bash
set -euo pipefail

CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config|-c)
      CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 --config <name>" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Usage: $0 --config <name>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="$SCRIPT_DIR/job_configs/${CONFIG}.conf"
if [[ ! -f "$CONF_FILE" ]]; then
  echo "Config file not found: $CONF_FILE" >&2
  exit 1
fi

source "$CONF_FILE"
source "$SCRIPT_DIR/submit_jobs_common.sh"
