#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_dashboard():
    """Run the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / "gui" / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)])

def export_model(args):
    """Export model to TFLite format"""
    export_script = Path(__file__).parent / "microcontroller" / "export_model.py"
    cmd = ["python", str(export_script)]
    
    if args.model:
        cmd.extend(["--model", args.model])
    if args.out_dir:
        cmd.extend(["--out-dir", args.out_dir])
    if args.no_quantize:
        cmd.append("--no-quantize")
    if args.no_optimize:
        cmd.append("--no-optimize")
    if args.platform:
        cmd.extend(["--platform", args.platform])
    if args.generate_header:
        cmd.append("--generate-header")
    
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="MIND: Neural EMG Decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the dashboard
  python main.py dashboard
  
  # Export model to TFLite
  python main.py export --model models/emg_decoder.h5 --platform arduino
  
  # Export model without quantization
  python main.py export --no-quantize --generate-header
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Run the Streamlit dashboard"
    )
    
    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export model to TFLite format"
    )
    export_parser.add_argument(
        "--model",
        help="Path to input Keras model"
    )
    export_parser.add_argument(
        "--out-dir",
        help="Output directory for exported files"
    )
    export_parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization"
    )
    export_parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable optimizations"
    )
    export_parser.add_argument(
        "--platform",
        choices=["arduino", "esp32", "generic"],
        help="Target platform"
    )
    export_parser.add_argument(
        "--generate-header",
        action="store_true",
        help="Generate Arduino header file"
    )
    
    args = parser.parse_args()
    
    if args.command == "dashboard":
        run_dashboard()
    elif args.command == "export":
        export_model(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
