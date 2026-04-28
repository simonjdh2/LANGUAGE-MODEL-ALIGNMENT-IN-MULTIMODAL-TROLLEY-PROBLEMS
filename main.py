"""
main.py — CLI entry point.

Usage:
    python3 main.py text              # run text arm (1000 scenarios)
    python3 main.py text 5            # run text arm (5 scenarios)
    python3 main.py image 5           # run image arm (5 scenarios)
    python3 main.py report            # generate HTML report from latest CSVs

API keys must be set as environment variables:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
"""

import argparse
from pathlib import Path

from scenario_generator import N_SCENARIOS
from text_arm import run_text_experiment
from image_arm import run_image_experiment
from report import generate_report, find_latest_pair

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Trolley Problem Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t_p = sub.add_parser("text",  help="run text arm")
    t_p.add_argument("n", nargs="?", type=int, default=N_SCENARIOS)

    i_p = sub.add_parser("image", help="run image arm")
    i_p.add_argument("n", nargs="?", type=int, default=N_SCENARIOS)

    r_p = sub.add_parser("report", help="generate HTML report")
    r_p.add_argument("text_csv",  nargs="?", help="path to text arm CSV")
    r_p.add_argument("image_csv", nargs="?", help="path to image arm CSV")
    r_p.add_argument("--output-dir", help="directory to write the HTML report")

    args = p.parse_args()

    if args.cmd == "text":
        run_text_experiment(args.n)

    elif args.cmd == "image":
        run_image_experiment(args.n)

    elif args.cmd == "report":
        if args.text_csv and args.image_csv:
            generate_report(Path(args.text_csv), Path(args.image_csv), output_dir=args.output_dir)
        else:
            t_csv, i_csv = find_latest_pair()
            print(f"Auto-detected: {t_csv.name} + {i_csv.name}")
            generate_report(t_csv, i_csv, output_dir=args.output_dir)
