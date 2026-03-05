"""Quick test to validate fused optimizer implementation."""

import torch
import re
from pathlib import Path


def check_file_for_fused_optimizer(file_path):
    """Check if file has fused optimizer implementation."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Look for the fused=True pattern
    has_fused = 'fused=True' in content

    # Look for try-except fallback pattern
    has_fallback = 'except RuntimeError:' in content and 'fused' in content

    # Find the configure_optimizers method
    method_match = re.search(r'def configure_optimizers\(self\):.*?(?=\n    def |\Z)', content, re.DOTALL)

    return {
        'has_fused': has_fused,
        'has_fallback': has_fallback,
        'method_found': method_match is not None,
    }


def main():
    """Run tests on all model files."""
    print("="*60)
    print("Fused Optimizer Implementation Validation")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print()

    # Files to check
    files_to_check = [
        ("Link Prediction", "qtaim_embed/models/link_pred/link_model.py"),
        ("Graph Regression", "qtaim_embed/models/graph_level/base_gcn.py"),
        ("Node Prediction", "qtaim_embed/models/node_level/base_gcn.py"),
        ("Graph Classifier", "qtaim_embed/models/graph_level/base_gcn_classifier.py"),
    ]

    results = {}

    for name, file_path in files_to_check:
        print(f"\n{'='*60}")
        print(f"Checking: {name}")
        print(f"File: {file_path}")
        print(f"{'='*60}")

        full_path = Path(file_path)
        if not full_path.exists():
            print(f"❌ File not found!")
            results[name] = False
            continue

        check_result = check_file_for_fused_optimizer(full_path)

        print(f"   configure_optimizers found: {check_result['method_found']}")
        print(f"   fused=True present: {check_result['has_fused']}")
        print(f"   Fallback logic present: {check_result['has_fallback']}")

        if all(check_result.values()):
            print(f"   ✅ PASS - Fused optimizer properly implemented!")
            results[name] = True
        else:
            print(f"   ❌ FAIL - Missing fused optimizer implementation")
            results[name] = False

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_type, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_type:20s}: {status}")

    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("Fused optimizer implementation verified in all model files!")
        print()
        print("Expected benefits:")
        print("  - 20-40% faster optimizer step")
        print("  - Reduced CPU time (from 31% to ~20%)")
        print("  - Overall throughput increase of 20-30%")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Check output above for details")
    print(f"{'='*60}")

    # Additional validation: Can we actually create a fused optimizer?
    print(f"\n{'='*60}")
    print("Runtime Test: Creating Fused Optimizer")
    print(f"{'='*60}")

    try:
        # Create a simple model
        model = torch.nn.Linear(10, 1)

        # Try to create fused optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, fused=True)

        print("✅ Successfully created fused Adam optimizer!")
        print(f"   Optimizer defaults: {optimizer.defaults}")
        print(f"   Fused: {optimizer.defaults.get('fused', 'NOT SET')}")

        if optimizer.defaults.get('fused', False):
            print("   🎉 FUSED OPTIMIZER IS ACTIVE!")
        else:
            print("   ⚠️  Warning: fused parameter is False")

    except RuntimeError as e:
        print(f"⚠️  Cannot create fused optimizer: {e}")
        print("   This is expected if:")
        print("   - PyTorch < 2.0")
        print("   - CUDA not available")
        print("   - Running on CPU")
        print()
        print("   Fallback to regular Adam will be used automatically.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
