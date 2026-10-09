import os
import sys
import json

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.test_benchmark import ISLSystemBenchmark
from src.models.signer_independent_eval import SignerIndependentEvaluator

def main():
    print("=================================================================")
    print(" EXECUTING ISL QUANTITATIVE & SIGNER-INDEPENDENT BENCHMARK       ")
    print("=================================================================\n")

    # 1. Full System Evaluation
    benchmark = ISLSystemBenchmark()
    report = benchmark.run_full_benchmark()

    static = report.get("static_classification", {})
    unknown = report.get("unknown_rejection", {})
    translation = report.get("translation_evaluation", {})
    perf = report.get("performance_profile", {})

    print("\n--- 1. FULL ISL STATIC & ALPHABET CLASSIFICATION (42 CLASSES) ---")
    print(f"Accuracy:           {static.get('accuracy', 0) * 100:.2f}%")
    print(f"Macro Precision:    {static.get('macro_precision', 0) * 100:.2f}%")
    print(f"Macro Recall:       {static.get('macro_recall', 0) * 100:.2f}%")
    print(f"Macro F1-Score:     {static.get('macro_f1', 0) * 100:.2f}%")
    print(f"Total Test Samples: {static.get('total_samples', 0)}")

    print("\n--- 2. UNKNOWN SIGN REJECTION METRICS ---")
    print(f"True Rejection Rate (TRR):   {unknown.get('true_rejection_rate', 0) * 100:.2f}%")
    print(f"False Acceptance Rate (FAR):  {unknown.get('false_acceptance_rate', 0) * 100:.2f}%")
    print(f"False Rejection Rate (FRR):  {unknown.get('false_rejection_rate', 0) * 100:.2f}%")

    print("\n--- 3. ISL-TO-ENGLISH TRANSLATION EVALUATION ---")
    print(f"Parallel Corpus Size: {translation.get('corpus_size', 0)} sentences")
    print(f"Average Word Error Rate (WER): {translation.get('average_wer', 0) * 100:.2f}%")
    print(f"Exact Match Ratio:             {translation.get('exact_match_ratio', 0) * 100:.2f}%")

    print("\n--- 4. PIPELINE LATENCY & THROUGHPUT ---")
    print(f"Inference Latency:      {perf.get('average_inference_latency_ms', 0):.2f} ms")
    print(f"Estimated Throughput:   {perf.get('estimated_throughput_fps', 0):.1f} FPS")

    # 2. Signer-Independent Disjoint Split Benchmark (Data Collection Rule Enforcement)
    print("\n--- 5. SIGNER-INDEPENDENT (DISJOINT SIGNER SPLITS) BENCHMARK ---")
    signer_eval = SignerIndependentEvaluator()
    si_report = signer_eval.run_disjoint_signer_evaluation(test_signer_id="signer_04")
    si_metrics = si_report.get("signer_independent_metrics", {})
    
    print(f"Train Signers:                     {', '.join(si_report.get('train_signers', []))}")
    print(f"Held-Out Unseen Test Signer:       {si_report.get('held_out_test_signer', '')}")
    print(f"Signer-Independent Accuracy:       {si_metrics.get('accuracy', 0) * 100:.2f}%")
    print(f"Signer-Independent Macro F1:       {si_metrics.get('macro_f1', 0) * 100:.2f}%")
    print(f"Signer-Dependent Baseline:         {si_report.get('signer_dependent_accuracy_baseline', 0) * 100:.2f}%")
    print(f"Generalization Drop:               {si_report.get('generalization_gap', 0) * 100:.2f}%")
    print(f"Conclusion: {si_report.get('finding_summary', '')}")

    print("\n=================================================================")
    print(" Evaluation complete. Reports saved to data/evaluation_report.json and data/signer_independent_report.json")
    print("=================================================================")

if __name__ == "__main__":
    main()
