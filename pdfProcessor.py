#!/usr/bin/env python3
"""
Simple Batch PDF to Markdown Processor

This script processes PDF files one by one using MinerU's magic-pdf tool,
converting them to markdown files and saving them to a specified output directory.
"""

import os
import argparse
import subprocess
from pathlib import Path
import time


def process_pdf(pdf_path, output_dir, conda_env=None, verbose=False):
    """
    Process a single PDF file using magic-pdf.
    
    Args:
        pdf_path (Path): Path to the PDF file
        output_dir (Path): Directory to save the output
        conda_env (str, optional): Conda environment name where magic-pdf is installed
        verbose (bool): Whether to print verbose output
    
    Returns:
        bool: True if successful, False otherwise
    """
    pdf_name = pdf_path.name
    
    if verbose:
        print(f"Processing: {pdf_name}")
    
    try:
        # Create the base command for magic-pdf
        base_cmd = ["magic-pdf", "-p", str(pdf_path), "-o", str(output_dir)]
        
        # If conda environment is specified, prepend conda run command
        if conda_env:
            cmd = ["conda", "run", "-n", conda_env, "--no-capture-output"] + base_cmd
        else:
            cmd = base_cmd
        
        # Execute the command
        if verbose:
            print(f"Running command: {' '.join(cmd)}")
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        # Check if command was successful
        if result.returncode == 0:
            if verbose:
                print(f"Successfully processed: {pdf_name}")
            return True
        else:
            print(f"Error processing {pdf_name}:")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"Exception while processing {pdf_name}: {str(e)}")
        return False


def generate_shell_script(input_dir, output_dir, conda_env):
    """
    Generate a simple shell script to process PDFs using the conda environment.
    
    Args:
        input_dir (str): Directory containing PDF files
        output_dir (str): Directory to save the output files
        conda_env (str): Conda environment name where magic-pdf is installed
        
    Returns:
        str: Path to the generated shell script
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return None
    
    # Create a shell script
    script_path = Path.cwd() / "process_pdfs.sh"
    
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# Activate conda environment {conda_env}\n")
        f.write(f"source \"$(conda info --base)/etc/profile.d/conda.sh\"\n")
        f.write(f"conda activate {conda_env}\n\n")
        
        f.write("# Process PDF files\n")
        f.write(f"echo \"Found {len(pdf_files)} PDF files to process\"\n\n")
        
        for i, pdf in enumerate(pdf_files):
            f.write(f"echo \"Processing {i+1}/{len(pdf_files)}: {pdf.name}\"\n")
            f.write(f"magic-pdf -p \"{pdf}\" -o \"{output_path}\"\n")
            f.write(f"if [ $? -eq 0 ]; then\n")
            f.write(f"  echo \"Successfully processed: {pdf.name}\"\n")
            f.write(f"else\n")
            f.write(f"  echo \"Error processing: {pdf.name}\"\n")
            f.write(f"fi\n\n")
            
        f.write("echo \"All processing complete!\"\n")
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Process PDF files using MinerU's magic-pdf tool (one at a time)"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing PDF files"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for processed markdown files"
    )
    parser.add_argument(
        "-c", "--conda-env",
        help="Conda environment name where magic-pdf is installed (e.g., 'MinerU')"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "-s", "--script",
        action="store_true",
        help="Generate a shell script instead of executing commands directly"
    )
    
    args = parser.parse_args()
    
    # Generate shell script if requested
    if args.script:
        if not args.conda_env:
            print("Error: --conda-env is required when using --script")
            exit(1)
            
        script_path = generate_shell_script(
            args.input,
            args.output,
            args.conda_env
        )
        
        if script_path:
            print(f"Shell script generated: {script_path}")
            print(f"Run it with: bash {script_path}")
        else:
            print("Failed to generate shell script")
        
        exit(0)
    
    # Find all PDF files
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {args.input}")
        exit(0)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file one by one
    successful = 0
    failed = 0
    
    for i, pdf in enumerate(pdf_files):
        print(f"Processing {i+1}/{len(pdf_files)}: {pdf.name}")
        
        result = process_pdf(
            pdf,
            output_path,
            args.conda_env,
            args.verbose
        )
        
        if result:
            successful += 1
        else:
            failed += 1
        
        # Brief pause between processing files
        time.sleep(1)
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    
    if failed > 0:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()