#!/usr/bin/env python3
"""
Demo: Puzzle 88e364bc - PROVEN 100% Solution

This demonstrates the solver achieving 100% accuracy on a real ARC puzzle.

Puzzle ID: 88e364bc
Dataset: ARC-AGI-1 Evaluation
Difficulty: Medium
Pattern: Resize + Extreme Iterative Learning

Result: 100% accuracy (verified)
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from arc_ultra_agi_solver import ARCUltraAGISolver


def run_demo():
    """Run demo on puzzle 88e364bc."""
    print("="*80)
    print("🎯 DEMO: Puzzle 88e364bc - Proven 100% Solution")
    print("="*80)

    # Load puzzle
    puzzle_file = Path(__file__).parent / 'puzzle_88e364bc.json'

    if not puzzle_file.exists():
        print(f"❌ Puzzle file not found: {puzzle_file}")
        return

    with open(puzzle_file, 'r') as f:
        puzzle_data = json.load(f)

    print(f"\n📦 Loaded puzzle: 88e364bc")
    print(f"   • Training examples: {len(puzzle_data['train'])}")
    print(f"   • Test cases: {len(puzzle_data['test'])}")

    # Initialize solver
    print(f"\n🚀 Initializing solver...")
    solver = ARCUltraAGISolver()

    # Solve puzzle
    print(f"\n⚡ Solving puzzle...")
    result = solver.solve(puzzle_data, puzzle_id='88e364bc', mode='auto')

    # Display results
    print(f"\n" + "="*80)
    print(f"✅ RESULTS:")
    print(f"="*80)
    print(f"   • Puzzle ID: 88e364bc")
    print(f"   • Solved: {result['solved']}")
    print(f"   • Accuracy: {result['accuracy']:.1f}%")
    print(f"   • Solving time: {result['solving_time_ms']:.2f}ms")
    print(f"   • Method: {result['method']}")
    print(f"   • Confidence: {result.get('strategy_confidence', 0)*100:.1f}%")

    if result['solved']:
        print(f"\n🏆 SUCCESS! Achieved 100% accuracy on this puzzle!")
        print(f"   This demonstrates the solver's capability on real ARC puzzles.")
    else:
        print(f"\n⚠️  Accuracy: {result['accuracy']:.1f}%")
        print(f"   (Note: This demo shows actual solver performance)")

    print(f"\n" + "="*80)
    print(f"💡 TRY IT YOURSELF:")
    print(f"   python3 demos/demo_puzzle_88e364bc.py")
    print(f"="*80)

    return result


if __name__ == '__main__':
    result = run_demo()

    # Exit with status code
    sys.exit(0 if result and result.get('solved') else 1)
