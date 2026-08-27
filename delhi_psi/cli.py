"""delhi-psi <stage> --config <profile-or-path> [--data-dir D] [--out-dir O].

Exit codes: 0 success, 1 a validation or IO failure, 2 a bad configuration or
a usage error (argparse's own code).
"""

import argparse
import logging
import sys

from delhi_psi import pipeline
from delhi_psi.config import ConfigError, load_config
from delhi_psi.validate import ValidationError

log = logging.getLogger("delhi_psi")

STAGES = {"preprocess": pipeline.preprocess, "compute": pipeline.compute}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="delhi-psi", description="Delhi Public Services Index pipeline")
    parser.add_argument("stage", choices=sorted(STAGES),
                        help="pipeline stage to run")
    parser.add_argument("--config", default="code-2025",
                        help="shipped profile name or path to a YAML file")
    parser.add_argument("--data-dir", default=None, help="input data root")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: the data dir)")
    parser.add_argument("--log-level", default="INFO",
                        help="logging level (default: INFO)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), "INFO"),
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config, data_dir=args.data_dir,
                          out_dir=args.out_dir)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        result = STAGES[args.stage](cfg)
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, OSError) as exc:
        print(f"input/output error: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
