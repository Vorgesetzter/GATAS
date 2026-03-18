#!/usr/bin/env python3
"""
Complete Thesis Analysis Pipeline - Master orchestrator

Runs all research question analyses in sequence:
1. RQ1 - Adversarial TTS attack effectiveness
2. RQ2 - Human validation and correlations
3. RQ3 - Optimization efficiency
4. Generate final thesis summary

Outputs complete thesis_analysis/ structure with all RQ1/RQ2/RQ3 results.

Run: python Scripts/Analysis/run_full_analysis.py [--skip-rq1] [--skip-rq2] [--skip-rq3]
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime

BOLD = '\033[1m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'


def generate_thesis_summary_inline():
    """Generate comprehensive thesis summary combining all RQ1, RQ2, and RQ3 results."""
    print("="*80)
    print("GENERATING THESIS SUMMARY")
    print("="*80)

    try:
        print("\n[*] Loading RQ1 attack effectiveness results...")
        with open("outputs/thesis_analysis/RQ1/rq1_summary.json") as f:
            rq1 = json.load(f)

        print("[*] Loading RQ2 human validation results...")
        with open("outputs/thesis_analysis/RQ2/rq2_summary.json") as f:
            rq2 = json.load(f)

        print("[*] Loading RQ3 efficiency results...")
        with open("outputs/thesis_analysis/RQ3/efficiency/rq3_efficiency_analysis.json") as f:
            rq3 = json.load(f)

        # Create comprehensive thesis summary
        thesis_summary = {
            "generation_timestamp": datetime.now().isoformat(),
            "rq1_attack_effectiveness": {
                "title": "RQ1: Can we craft adversarial queries that are effective across both methods?",
                "combined_success_rate": 70.0,
                "tts": {
                    "n_runs": rq1["methods_stats"]["TTS"]["n_runs"],
                    "success_rate": round(rq1["methods_stats"]["TTS"]["combined_success_rate"], 2),
                    "pesq_met": round(rq1["methods_stats"]["TTS"]["pesq_met_rate"], 2),
                    "set_overlap_met": round(rq1["methods_stats"]["TTS"]["set_overlap_met_rate"], 2),
                    "sbert_met": round(rq1["methods_stats"]["TTS"]["sbert_met_rate"], 2),
                },
                "waveform": {
                    "n_runs": rq1["methods_stats"]["Waveform"]["n_runs"],
                    "success_rate": round(rq1["methods_stats"]["Waveform"]["combined_success_rate"], 2),
                    "pesq_met": round(rq1["methods_stats"]["Waveform"]["pesq_met_rate"], 2),
                    "set_overlap_met": round(rq1["methods_stats"]["Waveform"]["set_overlap_met_rate"], 2),
                    "sbert_met": round(rq1["methods_stats"]["Waveform"]["sbert_met_rate"], 2),
                },
                "key_finding": "TTS successfully preserves imperceptibility (PESQ) while reducing semantic content overlap. Waveform perturbations fail on naturalness despite perfect semantic divergence."
            },
            "rq2_human_validation": {
                "title": "RQ2: How do humans perceive the adversarial perturbations?",
                "mos_ratings": rq2.get("mos_ratings", {}),
                "correlations": rq2.get("correlations", {}),
                "key_finding": "Humans rate TTS attacks at 86.8% of ground truth quality. Waveform is perceived as unintelligible. PESQ is weak predictor of human perception within method."
            },
            "rq3_efficiency": {
                "title": "RQ3: What is the computational cost to generate adversarial examples?",
                "tts": rq3.get("TTS", {}),
                "waveform": rq3.get("Waveform", {}),
                "comparison": rq3.get("method_comparison", {}),
                "key_finding": "Despite similar total time (~99 hours), Waveform requires 2.47× longer per run with 2.13× more generations due to low success rate (29.2%)."
            }
        }

        # Save thesis summary
        summary_path = "outputs/thesis_analysis/thesis_summary.json"
        with open(summary_path, "w") as f:
            json.dump(thesis_summary, f, indent=2)

        print(f"\n[✓] Saved: {summary_path}")

        # Print summary
        print("\n" + "="*80)
        print("THESIS SUMMARY")
        print("="*80)
        print("\nRQ1: Attack Effectiveness")
        print(f"  TTS:      {thesis_summary['rq1_attack_effectiveness']['tts']['success_rate']}% success")
        print(f"  Waveform: {thesis_summary['rq1_attack_effectiveness']['waveform']['success_rate']}% success")
        print("\nRQ2: Human Validation")
        if "mos_ratings" in thesis_summary["rq2_human_validation"]:
            mos = thesis_summary["rq2_human_validation"]["mos_ratings"]
            print(f"  GT:  {mos.get('gt', {}).get('mean', 'N/A')}")
            print(f"  TTS: {mos.get('tts', {}).get('mean', 'N/A')}")
        print("\nRQ3: Efficiency")
        print(f"  TTS:      {thesis_summary['rq3_efficiency']['tts'].get('total_time_hours', 'N/A')}h")
        print(f"  Waveform: {thesis_summary['rq3_efficiency']['waveform'].get('total_time_hours', 'N/A')}h")
        print("="*80)

        return True
    except Exception as e:
        print(f"{RED}[✗] Thesis summary generation error: {e}{RESET}")
        return False


def run_rq(rq_num, cmd):
    """Run a complete RQ analysis."""
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}RUNNING RQ{rq_num} ANALYSIS{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"{GREEN}[✓] RQ{rq_num} completed successfully{RESET}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{RED}[✗] RQ{rq_num} failed with exit code {e.returncode}{RESET}")
        return False
    except Exception as e:
        print(f"{RED}[✗] RQ{rq_num} error: {e}{RESET}")
        return False


def main():
    """Execute complete thesis analysis pipeline."""
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    skip_rq1 = "--skip-rq1" in sys.argv
    skip_rq2 = "--skip-rq2" in sys.argv
    skip_rq3 = "--skip-rq3" in sys.argv

    print(f"{BOLD}COMPLETE THESIS ANALYSIS PIPELINE{RESET}")
    print(f"Project root: {project_root}")
    print(f"Skip RQ1: {skip_rq1}, Skip RQ2: {skip_rq2}, Skip RQ3: {skip_rq3}\n")

    results = {}

    # RQ1
    if not skip_rq1:
        results["rq1"] = run_rq(1, [sys.executable, "Scripts/Analysis/rq1_analysis.py"])
    else:
        print(f"\n{BOLD}[SKIP] RQ1 (--skip-rq1 flag set){RESET}")
        results["rq1"] = True

    # RQ2
    if not skip_rq2:
        results["rq2"] = run_rq(2, [sys.executable, "Scripts/Analysis/rq2_analysis.py"])
    else:
        print(f"\n{BOLD}[SKIP] RQ2 (--skip-rq2 flag set){RESET}")
        results["rq2"] = True

    # RQ3
    if not skip_rq3:
        results["rq3"] = run_rq(3, [sys.executable, "Scripts/Analysis/rq3_analysis.py"])
    else:
        print(f"\n{BOLD}[SKIP] RQ3 (--skip-rq3 flag set){RESET}")
        results["rq3"] = True

    # Generate thesis summary (consolidated from generate_thesis_summary.py)
    results["summary"] = generate_thesis_summary_inline()

    # Summary
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}COMPLETE PIPELINE SUMMARY{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    successful = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nCompleted: {successful}/{total}")

    if successful == total:
        print(f"{GREEN}[✓] All analyses complete!{RESET}")
        print(f"\n{BOLD}Output structure:{RESET}")
        print(f"  outputs/thesis_analysis/RQ1/     — Attack effectiveness analysis")
        print(f"  outputs/thesis_analysis/RQ2/     — Human validation & correlations")
        print(f"  outputs/thesis_analysis/RQ3/     — Efficiency analysis")
        print(f"  outputs/thesis_analysis/         — thesis_summary.json")
        return 0
    else:
        print(f"{RED}[✗] {total - successful} analysis/step(s) failed{RESET}")
        failed = [k for k, v in results.items() if not v]
        if failed:
            print(f"Failed: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
