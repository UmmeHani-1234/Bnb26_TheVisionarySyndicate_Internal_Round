"""Entry point: run the agent on one request and print a readable execution timeline.

    python main.py
    python main.py "I need a laptop around ₹60,000 with at least 16GB RAM."
    python main.py --json                       # also print the raw events as JSON
    python main.py --save-events events.json    # write the raw events to a file
"""

import argparse
import json
import sys

from agent.agent import ConfigError, LaptopAgent
from agent.events import format_timeline

DEFAULT_REQUEST = "Find a laptop under ₹70,000 suitable for programming and gaming."


def main(argv=None) -> int:
    # Make ₹ and box-drawing characters safe on Windows consoles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run the Black Box target agent once.")
    parser.add_argument("request", nargs="?", default=DEFAULT_REQUEST, help="Natural-language laptop request.")
    parser.add_argument("--json", action="store_true", help="Also print the raw event list as JSON.")
    parser.add_argument("--save-events", metavar="PATH", help="Write the raw event list to a JSON file.")
    args = parser.parse_args(argv)

    from recorder.recorder import ExecutionRecorder

    try:
        recorder = ExecutionRecorder()
        agent = LaptopAgent(sinks=[recorder.record])
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    result = agent.run(args.request)

    print(format_timeline(result.events, result.request, result.final_response))

    events = [e.to_dict() for e in result.events]
    if args.json:
        print(json.dumps(events, indent=2, ensure_ascii=False))
    if args.save_events:
        with open(args.save_events, "w", encoding="utf-8") as handle:
            json.dump(events, handle, indent=2, ensure_ascii=False)
        print(f"Saved {len(events)} events to {args.save_events}")

    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
