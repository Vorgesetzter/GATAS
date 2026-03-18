import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from Trainer.RunLogger import RunLogger


def main():
    parser = argparse.ArgumentParser(description="Aggregate run_summary.json files into all_results.json and all_results.csv")
    parser.add_argument("results_dir", help="Directory containing sentence_*/run_*/run_summary.json files")
    parser.add_argument("--output_dir", default=None, help="Where to write all_results.json/csv (default: results_dir)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = args.output_dir or str(results_dir)

    summaries = []
    for json_path in sorted(results_dir.rglob("run_summary.json")):
        with open(json_path) as f:
            summaries.append(json.load(f))

    print(f"[*] Found {len(summaries)} runs.")
    RunLogger.aggregate_results(summaries, output_dir=output_dir)


if __name__ == "__main__":
    main()
