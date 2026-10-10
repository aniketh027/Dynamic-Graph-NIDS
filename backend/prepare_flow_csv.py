"""Offline exact 80/84-column CSV preview, preparation and linkage verification."""
import argparse
import json
import sys

from backend.app.services.csv_compatibility import CompatibilityError, prepare, preview, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("preview", help="inspect only the source header; write nothing")
    inspect.add_argument("--source", required=True)
    convert = commands.add_parser("prepare", help="prepare a new output and .provenance.json sidecar")
    convert.add_argument("--source", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--exporter")
    convert.add_argument("--exporter-version")
    check = commands.add_parser("verify", help="verify output/sidecar linkage, without authenticating an exporter")
    check.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preview":
            result = preview(args.source)
        elif args.command == "prepare":
            result = prepare(args.source, args.output, exporter=args.exporter, exporter_version=args.exporter_version)
        else:
            result = verify(args.output)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    except (CompatibilityError, OSError):
        # Native exceptions can contain full paths. Compatibility diagnostics never contain cells.
        exc = sys.exception()
        message = str(exc) if isinstance(exc, CompatibilityError) else "File access failed; check the paths and permissions."
        print(f"CSV preparation failed: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
