

set -euo pipefail

AGENT_URL=${AGENT_URL:-http://127.0.0.1:8001}
MODE=${1:-stream} # stream|nonstream

echo "Testing against: $AGENT_URL (mode=$MODE)" >&2

build_payload() {
python - << 'PY'
import json

# Empty board: 100 zeros (EMPTY)
state = [0]*100
# current turn RED=1 or BLUE=2 (make BLUE=2 so AI moves)
state += [2]  # [100]
# scores
state += [1, 1]  # [101],[102]
# placed tiles count
state += [0]  # [103]

valid_actions = list(range(100))
print(json.dumps({"state": state, "valid_actions": valid_actions}))
PY
}

if [[ "$MODE" == "nonstream" ]]; then
  echo "\n-- Non-streaming /move timing --" >&2
  build_payload | \
    curl -sS -o /dev/null -w $'status=%{http_code}\nnamelookup=%{time_namelookup}\nconnect=%{time_connect}\nappconnect=%{time_appconnect}\nstarttransfer=%{time_starttransfer}\nredirect=%{time_redirect}\npretransfer=%{time_pretransfer}\ntotal=%{time_total}\n' \
      -H 'Content-Type: application/json' \
      -X POST "$AGENT_URL/move" --data-binary @-
else
  echo "\n-- Streaming /move_stream timestamps --" >&2
  # Print millisecond timestamps for each SSE line as it arrives
  build_payload | \
    curl -sS -N -H 'Accept: text/event-stream' -H 'Content-Type: application/json' -X POST "$AGENT_URL/move_stream" --data-binary @- \
    | awk '{ cmd="date +%H:%M:%S.%3N"; cmd | getline t; close(cmd); print t " " $0; fflush(); }'
fi

echo "\nDone." >&2
