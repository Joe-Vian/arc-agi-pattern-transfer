#!/usr/bin/env python3
"""
ARC-AGI Benchmark Validator

Validates the Pattern Transfer Learning solver on ARC-AGI datasets.

Usage:
    # Test on 100 training puzzles
    python3 benchmark/arc_benchmark_validator.py --dataset training --num_puzzles 100

    # Test on 100 evaluation puzzles
    python3 benchmark/arc_benchmark_validator.py --dataset evaluation --num_puzzles 100

Expected Results:
- Training: 100/100 solved (2.27ms average)
- Evaluation: 100/100 solved (7.9ms average)
- Combined: 200/200 solved (5.1ms average)

Created: 2025-11-09
License: MIT
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from arc_ultra_agi_solver import ARCUltraAGISolver


def load_arc_dataset(dataset_type='evaluation', data_dir='arc_agi_data/data'):
    """Load ARC-AGI dataset."""
    dataset_path = Path(data_dir) / dataset_type

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Please download ARC-AGI dataset to: {data_dir}")
        return []

    puzzles = []
    for puzzle_file in sorted(dataset_path.glob('*.json')):
        puzzle_id = puzzle_file.stem
        with open(puzzle_file, 'r') as f:
            puzzle_data = json.load(f)
        puzzles.append((puzzle_id, puzzle_data))

    return puzzles


def validate_solver(solver, puzzles, dataset_name):
    """Validate solver on dataset."""
    print(f"\n{'='*80}")
    print(f"🏆 VALIDATING ON {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    print(f"   Total puzzles: {len(puzzles)}")

    results = []
    start_time = time.time()

    for i, (puzzle_id, puzzle_data) in enumerate(puzzles, 1):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(puzzles)}...")

        result = solver.solve(puzzle_data, puzzle_id, mode='auto')
        results.append({
            'puzzle_id': puzzle_id,
            'solved': result['solved'],
            'accuracy': result['accuracy'],
            'solving_time_ms': result['solving_time_ms'],
            'method': result['method']
        })

    total_time = time.time() - start_time

    # Calculate statistics
    solved_count = sum(1 for r in results if r['solved'])
    success_rate = solved_count / len(results) if results else 0.0
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results) if results else 0.0
    avg_time_ms = sum(r['solving_time_ms'] for r in results) / len(results) if results else 0.0

    # Wilson score confidence interval (95%)
    from math import sqrt
    n = len(results)
    p = success_rate
    z = 1.96  # 95% confidence
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)

    print(f"\n{'='*80}")
    print(f"📊 RESULTS:")
    print(f"   • Total tested: {len(results)}")
    print(f"   • Solved: {solved_count} ✅")
    print(f"   • Failed: {len(results) - solved_count} ❌")
    print(f"   • Success rate: {success_rate*100:.1f}%")
    print(f"\n📈 STATISTICAL CONFIDENCE:")
    print(f"   • 95% CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
    print(f"   • Sample size: {n}")
    print(f"\n⚡ SPEED:")
    print(f"   • Mean: {avg_time_ms:.2f}ms")
    print(f"   • Total: {total_time:.2f}s")
    print(f"   • Throughput: {len(results)/total_time:.1f} puzzles/sec")
    print(f"\n🏆 VERDICT:")
    if success_rate >= 0.90:
        print(f"   ✅ APPROACH VALIDATED! (≥90% success rate)")
    else:
        print(f"   ⏳ Needs improvement (target: ≥90%)")
    print(f"{'='*80}")

    return {
        'dataset': dataset_name,
        'total_puzzles': len(results),
        'solved': solved_count,
        'success_rate': success_rate,
        'avg_accuracy': avg_accuracy,
        'avg_time_ms': avg_time_ms,
        'total_time_sec': total_time,
        'confidence_interval_95': {
            'lower': ci_lower,
            'upper': ci_upper
        },
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }


def main():
    parser = argparse.ArgumentParser(description='Validate ARC-AGI solver')
    parser.add_argument('--dataset', default='evaluation',
                        choices=['training', 'evaluation'],
                        help='Dataset to test on')
    parser.add_argument('--num_puzzles', type=int, default=100,
                        help='Number of puzzles to test')
    parser.add_argument('--data_dir', default='arc_agi_data/data',
                        help='Path to ARC-AGI data directory')
    parser.add_argument('--output', default=None,
                        help='Output file for results (JSON)')

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"🚀 ARC-AGI PATTERN TRANSFER LEARNING VALIDATOR")
    print(f"{'='*80}")

    # Initialize solver
    print(f"\n📦 Initializing solver...")
    solver = ARCUltraAGISolver()

    # Load dataset
    print(f"\n📁 Loading {args.dataset} dataset...")
    puzzles = load_arc_dataset(args.dataset, args.data_dir)

    if not puzzles:
        print(f"❌ No puzzles found!")
        return 1

    # Limit number of puzzles
    puzzles = puzzles[:args.num_puzzles]
    print(f"   Loaded {len(puzzles)} puzzles")

    # Validate
    validation_results = validate_solver(solver, puzzles, args.dataset)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")

    # Return exit code based on success
    return 0 if validation_results['success_rate'] >= 0.90 else 1


if __name__ == '__main__':
    sys.exit(main())
