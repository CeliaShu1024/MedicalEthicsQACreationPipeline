#!/usr/bin/env python3
import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
from datetime import datetime

def find_coi_files(base_directory="src/book_files", model_name="transformer"):
    """
    Find all content of interest (COI) files in the extracted books directory.
    
    Args:
        base_directory: Base directory containing book folders
        model_name: Model name used in the COI file naming pattern
        
    Returns:
        Dictionary mapping book IDs to their COI file paths
    """
    if not os.path.isdir(base_directory):
        print(f"Warning: Base directory {base_directory} does not exist")
        return {}
    
    coi_files = {}
    
    # Search for all coi files with the given model name
    pattern = os.path.join(base_directory, "*", f"*_coi_{model_name}.json")
    for coi_path in glob.glob(pattern):
        # Extract book ID from the file path
        dir_name = os.path.dirname(coi_path)
        book_id = os.path.basename(dir_name)
        coi_files[book_id] = coi_path
    
    return coi_files

def load_metadata_file(base_directory, book_id):
    """
    Load the original metadata file for a book to compare with COI content.
    
    Args:
        base_directory: Base directory containing book folders
        book_id: Book ID
        
    Returns:
        Metadata JSON or None if not found
    """
    metadata_path = os.path.join(base_directory, book_id, f"{book_id}_data.json")
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading data for {book_id}: {e}")
    return None

def count_tokens(text, tokenizer):
    """Count the number of tokens in a text using the specified tokenizer."""
    return len(tokenizer.encode(text))

def analyze_coi_file(coi_path, metadata=None, tokenizer=None):
    """
    Analyze a single COI file and compute statistics.
    
    Args:
        coi_path: Path to the COI JSON file
        metadata: Original metadata JSON for comparison (optional)
        tokenizer: Tokenizer for token counting (optional)
        
    Returns:
        Dictionary containing statistics for the book
    """
    try:
        with open(coi_path, 'r', encoding='utf-8') as f:
            coi_data = json.load(f)
        
        # Basic statistics
        stats = {
            "coi_items": len(coi_data),
            "total_items": len(metadata) if metadata else "N/A",
            "retention_rate": (len(coi_data) / len(metadata) * 100) if metadata else "N/A",
            "avg_heading_len": 0,
            "avg_text_len": 0,
            "min_text_len": float('inf'),
            "max_text_len": 0,
            "total_headings_chars": 0,
            "total_text_chars": 0,
            "chapter_distribution": defaultdict(int),
            "total_tokens": 0,
            "avg_tokens_per_item": 0,
            "min_tokens": float('inf'),
            "max_tokens": 0
        }
        
        # Process each item
        all_tokens = []
        for item in coi_data:
            # Get heading and text lengths
            heading_len = len(item["heading"])
            text_content = "\n\n".join(item["text"])
            text_len = len(text_content)
            
            # Update statistics
            stats["total_headings_chars"] += heading_len
            stats["total_text_chars"] += text_len
            stats["min_text_len"] = min(stats["min_text_len"], text_len)
            stats["max_text_len"] = max(stats["max_text_len"], text_len)
            
            # Extract chapter number if possible (assuming format like "Chapter X")
            if "chapter" in item["heading"].lower():
                match = re.search(r'chapter\s+(\d+)', item["heading"].lower())
                if match:
                    chapter = int(match.group(1))
                    stats["chapter_distribution"][chapter] += 1
            
            # Count tokens if tokenizer is provided
            if tokenizer:
                # Combine heading and text as would be used in an exam question
                combined_text = f"{item['heading']}\n\n{text_content}"
                token_count = count_tokens(combined_text, tokenizer)
                all_tokens.append(token_count)
                stats["total_tokens"] += token_count
                stats["min_tokens"] = min(stats["min_tokens"], token_count)
                stats["max_tokens"] = max(stats["max_tokens"], token_count)
        
        # Calculate averages
        if stats["coi_items"] > 0:
            stats["avg_heading_len"] = stats["total_headings_chars"] / stats["coi_items"]
            stats["avg_text_len"] = stats["total_text_chars"] / stats["coi_items"]
            if tokenizer:
                stats["avg_tokens_per_item"] = stats["total_tokens"] / stats["coi_items"]
                stats["token_percentiles"] = {
                    "25": np.percentile(all_tokens, 25),
                    "50": np.percentile(all_tokens, 50),
                    "75": np.percentile(all_tokens, 75),
                    "90": np.percentile(all_tokens, 90),
                    "95": np.percentile(all_tokens, 95)
                }
                stats["token_distribution"] = all_tokens
        
        # Reset min text length if no items were found
        if stats["min_text_len"] == float('inf'):
            stats["min_text_len"] = 0
            
        # Reset min tokens if no items were tokenized
        if stats["min_tokens"] == float('inf'):
            stats["min_tokens"] = 0
            
        return stats
        
    except Exception as e:
        print(f"Error analyzing COI file {coi_path}: {e}")
        return {"error": str(e)}

def create_summary_report(all_stats, output_dir, model_name):
    """
    Create a detailed summary report of the COI statistics.
    
    Args:
        all_stats: Dictionary mapping book IDs to their statistics
        output_dir: Directory to save the report
        model_name: Model name for report filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for the report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"coi_stats_{model_name}_{timestamp}.txt")
    
    # Calculate aggregate statistics
    total_books = len(all_stats)
    total_coi_items = sum(s["coi_items"] for s in all_stats.values())
    total_orig_items = sum(s["total_items"] for s in all_stats.values() if s["total_items"] != "N/A")
    total_tokens = sum(s["total_tokens"] for s in all_stats.values())
    
    all_retention_rates = [s["retention_rate"] for s in all_stats.values() if s["retention_rate"] != "N/A"]
    avg_retention = sum(all_retention_rates) / len(all_retention_rates) if all_retention_rates else 0
    
    all_token_counts = []
    for s in all_stats.values():
        if "token_distribution" in s:
            all_token_counts.extend(s["token_distribution"])
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Content of Interest Statistics Report\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        
        # Overall summary
        f.write(f"OVERALL SUMMARY:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Total books analyzed: {total_books}\n")
        f.write(f"Total COI items found: {total_coi_items}\n")
        f.write(f"Original total items: {total_orig_items}\n")
        f.write(f"Average retention rate: {avg_retention:.2f}%\n")
        f.write(f"Total tokens: {total_tokens:,}\n")
        
        # Token statistics if available
        if all_token_counts:
            avg_tokens = sum(all_token_counts) / len(all_token_counts)
            f.write(f"Average tokens per item: {avg_tokens:.1f}\n")
            f.write(f"Min tokens per item: {min(all_token_counts)}\n")
            f.write(f"Max tokens per item: {max(all_token_counts)}\n")
            
            # Calculate percentiles for all token counts
            f.write(f"\nToken count percentiles:\n")
            percentiles = [25, 50, 75, 90, 95, 99]
            for p in percentiles:
                f.write(f"  {p}th percentile: {np.percentile(all_token_counts, p):.0f} tokens\n")
        
        # Cost estimation (rough estimates based on current model pricing)
        f.write(f"\nToken Consumption Estimates:\n")
        f.write(f"Estimated input tokens for all COI items: {total_tokens:,}\n")
        # Assuming a standard prompt template and output generation
        template_tokens = 500  # Rough estimate for instructions, etc.
        estimated_prompt_tokens = total_coi_items * (template_tokens + np.percentile(all_token_counts, 50) if all_token_counts else 0)
        estimated_output_tokens = total_coi_items * 150  # Assuming ~150 tokens per question output
        f.write(f"Estimated total tokens for exam generation with all items:\n")
        f.write(f"  Prompt tokens: {estimated_prompt_tokens:,.0f}\n")
        f.write(f"  Output tokens: {estimated_output_tokens:,.0f}\n")
        f.write(f"  Total estimate: {estimated_prompt_tokens + estimated_output_tokens:,.0f}\n")
        
        # Per-book details
        f.write(f"\n{'='*80}\n")
        f.write(f"INDIVIDUAL BOOK STATISTICS:\n")
        f.write(f"{'='*80}\n\n")
        
        for book_id, stats in sorted(all_stats.items()):
            f.write(f"Book: {book_id}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"COI items: {stats['coi_items']}\n")
            f.write(f"Total items: {stats['total_items']}\n")
            f.write(f"Retention rate: {stats['retention_rate'] if stats['retention_rate'] != 'N/A' else 'N/A'}%\n")
            f.write(f"Average heading length: {stats['avg_heading_len']:.1f} chars\n")
            f.write(f"Average text length: {stats['avg_text_len']:.1f} chars\n")
            f.write(f"Text length range: {stats['min_text_len']} - {stats['max_text_len']} chars\n")
            
            if "total_tokens" in stats and stats["total_tokens"] > 0:
                f.write(f"Total tokens: {stats['total_tokens']:,}\n")
                f.write(f"Average tokens per item: {stats['avg_tokens_per_item']:.1f}\n")
                f.write(f"Token range: {stats['min_tokens']} - {stats['max_tokens']}\n")
                
                if "token_percentiles" in stats:
                    f.write(f"Token percentiles:\n")
                    for p, val in stats["token_percentiles"].items():
                        f.write(f"  {p}th: {val:.0f}\n")
            
            # Print chapter distribution if available
            if stats["chapter_distribution"]:
                f.write(f"\nChapter distribution:\n")
                for chapter, count in sorted(stats["chapter_distribution"].items()):
                    f.write(f"  Chapter {chapter}: {count} items\n")
            
            f.write(f"\n")
    
    print(f"Summary report generated at: {report_path}")
    return report_path

def create_visualizations(all_stats, output_dir, model_name):
    """
    Create visualizations of the COI statistics.
    
    Args:
        all_stats: Dictionary mapping book IDs to their statistics
        output_dir: Directory to save the visualizations
        model_name: Model name for visualization filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for the visualization filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Retention rates across books
    retention_data = {bid: stats["retention_rate"] for bid, stats in all_stats.items() 
                     if stats["retention_rate"] != "N/A"}
    
    if retention_data:
        plt.figure(figsize=(12, 6))
        plt.bar(retention_data.keys(), retention_data.values())
        plt.axhline(y=sum(retention_data.values())/len(retention_data), color='r', linestyle='-', 
                   label=f'Average: {sum(retention_data.values())/len(retention_data):.1f}%')
        plt.xlabel('Book ID')
        plt.ylabel('Retention Rate (%)')
        plt.title('Content Retention Rates by Book')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        retention_path = os.path.join(output_dir, f"retention_rates_{model_name}_{timestamp}.png")
        plt.savefig(retention_path)
        plt.close()
        print(f"Retention rate visualization saved to: {retention_path}")
    
    # 2. Token distribution histogram (combined across all books)
    all_token_counts = []
    for stats in all_stats.values():
        if "token_distribution" in stats:
            all_token_counts.extend(stats["token_distribution"])
    
    if all_token_counts:
        plt.figure(figsize=(10, 6))
        plt.hist(all_token_counts, bins=50, alpha=0.7, color='blue')
        plt.axvline(x=np.mean(all_token_counts), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(all_token_counts):.1f}')
        plt.axvline(x=np.median(all_token_counts), color='g', linestyle='--', 
                   label=f'Median: {np.median(all_token_counts):.1f}')
        
        # Add percentile lines
        percentiles = [90, 95]
        colors = ['orange', 'purple']
        for p, c in zip(percentiles, colors):
            p_val = np.percentile(all_token_counts, p)
            plt.axvline(x=p_val, color=c, linestyle=':', 
                       label=f'{p}th percentile: {p_val:.1f}')
        
        plt.xlabel('Tokens per Item')
        plt.ylabel('Frequency')
        plt.title('Distribution of Token Counts per Content Item')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        token_dist_path = os.path.join(output_dir, f"token_distribution_{model_name}_{timestamp}.png")
        plt.savefig(token_dist_path)
        plt.close()
        print(f"Token distribution visualization saved to: {token_dist_path}")
    
    # 3. Items per book comparison
    book_ids = list(all_stats.keys())
    coi_counts = [stats["coi_items"] for stats in all_stats.values()]
    total_counts = [stats["total_items"] if stats["total_items"] != "N/A" else 0 for stats in all_stats.values()]
    
    if book_ids:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(book_ids))
        width = 0.35
        
        plt.bar(x - width/2, total_counts, width, label='Original Items')
        plt.bar(x + width/2, coi_counts, width, label='COI Items')
        
        plt.xlabel('Book ID')
        plt.ylabel('Number of Items')
        plt.title('Original vs. COI Items by Book')
        plt.xticks(x, book_ids, rotation=90)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        items_path = os.path.join(output_dir, f"items_comparison_{model_name}_{timestamp}.png")
        plt.savefig(items_path)
        plt.close()
        print(f"Items comparison visualization saved to: {items_path}")
    
    return True

def export_to_csv(all_stats, output_dir, model_name):
    """
    Export statistics to CSV files for further analysis.
    
    Args:
        all_stats: Dictionary mapping book IDs to their statistics
        output_dir: Directory to save the CSV files
        model_name: Model name for CSV filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for the CSV filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Book summary statistics
    book_summary = []
    for book_id, stats in all_stats.items():
        summary = {
            "book_id": book_id,
            "coi_items": stats["coi_items"],
            "total_items": stats["total_items"] if stats["total_items"] != "N/A" else None,
            "retention_rate": stats["retention_rate"] if stats["retention_rate"] != "N/A" else None,
            "avg_heading_len": stats["avg_heading_len"],
            "avg_text_len": stats["avg_text_len"],
            "min_text_len": stats["min_text_len"],
            "max_text_len": stats["max_text_len"],
            "total_tokens": stats.get("total_tokens", None),
            "avg_tokens_per_item": stats.get("avg_tokens_per_item", None),
            "min_tokens": stats.get("min_tokens", None) if stats.get("min_tokens", float('inf')) != float('inf') else None,
            "max_tokens": stats.get("max_tokens", None)
        }
        
        # Add percentiles if available
        if "token_percentiles" in stats:
            for p, val in stats["token_percentiles"].items():
                summary[f"token_p{p}"] = val
                
        book_summary.append(summary)
    
    if book_summary:
        df_summary = pd.DataFrame(book_summary)
        summary_path = os.path.join(output_dir, f"book_summary_{model_name}_{timestamp}.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"Book summary CSV exported to: {summary_path}")
    
    # 2. Token distribution data (if available)
    token_data = []
    for book_id, stats in all_stats.items():
        if "token_distribution" in stats:
            for token_count in stats["token_distribution"]:
                token_data.append({
                    "book_id": book_id,
                    "token_count": token_count
                })
    
    if token_data:
        df_tokens = pd.DataFrame(token_data)
        tokens_path = os.path.join(output_dir, f"token_distribution_{model_name}_{timestamp}.csv")
        df_tokens.to_csv(tokens_path, index=False)
        print(f"Token distribution CSV exported to: {tokens_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate statistics for Content of Interest (COI) files')
    parser.add_argument('--dir', type=str, default='src/book_files', help='Base directory containing book folders')
    parser.add_argument('--model-name', type=str, default='transformer', help='Model name used in COI file naming')
    parser.add_argument('--output-dir', type=str, default='coi_stats', help='Directory to save output statistics')
    parser.add_argument('--tokenizer', type=str, help='Path to the tokenizer or HuggingFace model ID for token counting')
    parser.add_argument('--no-visualizations', action='store_true', help='Skip creating visualizations')
    parser.add_argument('--no-csv', action='store_true', help='Skip exporting to CSV')
    parser.add_argument('--book', type=str, help='Process a single book by its ID')
    args = parser.parse_args()
    
    # Load tokenizer if specified
    tokenizer = None
    if args.tokenizer:
        try:
            print(f"Loading tokenizer from {args.tokenizer}...")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f"Tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Continuing without token counting capability")
    
    # Find and process COI files
    if args.book:
        # Process a single book
        coi_path = os.path.join(args.dir, args.book, f"{args.book}_coi_{args.model_name}.json")
        if os.path.isfile(coi_path):
            coi_files = {args.book: coi_path}
        else:
            print(f"Error: Could not find COI file for book {args.book} at {coi_path}")
            return
    else:
        # Process all books
        coi_files = find_coi_files(args.dir, args.model_name)
    
    print(f"Found {len(coi_files)} COI files to analyze")
    
    # Analyze each COI file
    all_stats = {}
    for book_id, coi_path in coi_files.items():
        print(f"Analyzing {book_id}...")
        
        # Load the original metadata for comparison if available
        metadata = load_metadata_file(args.dir, book_id)
        
        # Analyze the COI file
        stats = analyze_coi_file(coi_path, metadata, tokenizer)
        all_stats[book_id] = stats
    
    # Create summary report
    if all_stats:
        report_path = create_summary_report(all_stats, args.output_dir, args.model_name)
        
        # Create visualizations if requested
        if not args.no_visualizations:
            create_visualizations(all_stats, args.output_dir, args.model_name)
        
        # Export to CSV if requested
        if not args.no_csv:
            export_to_csv(all_stats, args.output_dir, args.model_name)
    else:
        print("No statistics generated. Check that COI files exist and are accessible.")

if __name__ == "__main__":
    import re  # Import here as needed for chapter extraction
    main()