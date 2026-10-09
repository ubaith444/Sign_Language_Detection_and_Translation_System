import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.test_benchmark import ISLSystemBenchmark

def main():
    print("=================================================================")
    print(" TRAINING INDIAN SIGN LANGUAGE RECOGNITION MODELS                ")
    print("=================================================================\n")

    benchmark = ISLSystemBenchmark()
    report = benchmark.run_full_benchmark()

    acc = report.get("static_classification", {}).get("accuracy", 0)
    f1 = report.get("static_classification", {}).get("macro_f1", 0)

    print("\nModels successfully trained and serialized:")
    print(" - Static Sign Classifier: data/trained_models/isl_static_model.joblib")
    print(" - Dynamic GRU Sequence Model: data/trained_models/isl_dynamic_gru.pt")
    print(f"Validation Accuracy: {acc * 100:.2f}% | Macro F1: {f1 * 100:.2f}%")

if __name__ == "__main__":
    main()
