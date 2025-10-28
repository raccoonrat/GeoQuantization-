#!/usr/bin/env python3
"""
GeoQuantization 快速启动脚本
一键运行实验
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='GeoQuantization Experiment Runner')
    parser.add_argument('--setup', action='store_true', help='Setup environment first')
    parser.add_argument('--model', default='facebook/opt-125m', help='Model name')
    parser.add_argument('--samples', type=int, default=200, help='Number of samples')
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--output', default='prm_outputs', help='Output directory')
    
    args = parser.parse_args()
    
    print("🚀 GeoQuantization Experiment Runner")
    print("=" * 50)
    
    # Setup if requested
    if args.setup:
        print("📦 Setting up environment...")
        try:
            subprocess.run([sys.executable, "setup_experiment.py"], check=True)
            print("✅ Environment setup completed")
        except subprocess.CalledProcessError:
            print("❌ Setup failed")
            return 1
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    print(f"🧪 Running experiment with model: {args.model}")
    print(f"   Samples: {args.samples}")
    print(f"   Device: {args.device}")
    print(f"   Output: {args.output}")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        # Run the experiment
        result = subprocess.run([
            sys.executable, "prm_experiment.py"
        ], env=env, check=True, capture_output=True, text=True)
        
        print("✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        
        # List output files
        output_path = Path(args.output)
        if output_path.exists():
            files = list(output_path.glob("*"))
            if files:
                print("\n📄 Generated files:")
                for f in files:
                    print(f"   - {f.name}")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
