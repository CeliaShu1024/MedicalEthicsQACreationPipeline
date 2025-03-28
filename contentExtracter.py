import os
import re
import json
import glob
from pathlib import Path

def preprocess_markdown_to_json(book_directory):
    """
    Process a markdown file from MinerU extraction and convert it to the specified JSON format.
    
    Args:
        book_directory (str): Path to the book directory containing the 'auto' folder with markdown files
    
    Returns:
        dict: The processed content in the required JSON format
    """
    # Find the markdown file
    book_id = os.path.basename(os.path.normpath(book_directory))
    markdown_file = os.path.join(book_directory, "auto", f"{book_id}.md")
    if not os.path.exists(markdown_file):
        raise FileNotFoundError(f"Markdown file not found: {markdown_file}")
    
    # Read the markdown content
    with open(markdown_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Process the content
    sections = []
    current_heading = None
    current_paragraphs = []
    section_counter = 0
    
    # Split content by lines
    lines = content.split('\n')
    i = 0
    chapter_number = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if line is a heading (starts with #)
        if line.startswith('#'):
            if current_heading is not None: # If we already have a heading and paragraphs, add them to sections
                section_counter += 1
                sections.append({
                    "id": section_counter,
                    "heading": current_heading,
                    "text": current_paragraphs
                })
                current_paragraphs = []
            
            
            heading_text = re.sub(r'^#+\s*', '', line).strip() # Extract heading text without # symbols
            
            # Try to identify if this is a chapter heading (usually just a number or "Chapter X")
            if re.match(r'^(\d+|chapter\s+\d+)$', heading_text.lower()):
                chapter_number = heading_text
            if i + 1 < len(lines) and lines[i + 1].strip().startswith('#'): # Check if the next line is also a heading
                next_heading = re.sub(r'^#+\s*', '', lines[i + 1].strip()).strip()
                current_heading = f"{heading_text} - {next_heading}" # Merge headings
                i += 1  # Skip the next line since we've processed it
            else:
                current_heading = heading_text
                
            # Add chapter context to common section names if chapter number is available
            if chapter_number and re.match(r'^(abstract|introduction|conclusion|references|bibliography)$', current_heading.lower()):
                current_heading = f"Chapter {chapter_number} - {current_heading}"
        
        elif line and current_heading is not None: # Handle non-heading content (paragraphs)
            # Start a new paragraph
            paragraph = line
            
            # Collect continuous non-empty lines for this paragraph
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith('#'):
                paragraph += " " + lines[j].strip()
                j += 1
            
            i = j - 1  # Update the index to skip the lines we've processed
            current_paragraphs.append(paragraph)
        
        i += 1
    
    # Add the last section if there's one
    if current_heading is not None:
        section_counter += 1
        sections.append({
            "id": section_counter,
            "heading": current_heading,
            "text": current_paragraphs
        })
    
    return sections

def process_all_books(root_directory):
    """
    Process all book directories in the given root directory.
    
    Args:
        root_directory (str): Path to the directory containing all book directories
    """
    book_dirs = [d for d in os.listdir(root_directory) 
                if os.path.isdir(os.path.join(root_directory, d)) and 
                os.path.exists(os.path.join(root_directory, d, "auto"))]
    
    for book_id in book_dirs:
        book_path = os.path.join(root_directory, book_id)
        try:
            print(f"Processing book: {book_id}")
            # Get the JSON output
            json_content = preprocess_markdown_to_json(book_path)
            output_path = os.path.join(book_path, f"{book_id}_data.json")
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_content, json_file, indent=2, ensure_ascii=False)
            
            print(f"Successfully created: {output_path}")
        except Exception as e:
            print(f"Error processing book {book_id}: {str(e)}")

def process_single_book(book_directory):
    """
    Process a single book directory.
    
    Args:
        book_directory (str): Path to the book directory
    """
    try:
        book_id = os.path.basename(os.path.normpath(book_directory))
        print(f"Processing book: {book_id}")
        json_content = preprocess_markdown_to_json(book_directory)

        output_path = os.path.join(book_directory, f"{book_id}_data.json")
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_content, json_file, indent=2, ensure_ascii=False)
        
        print(f"Successfully created: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing book: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process MinerU markdown files to JSON data')
    parser.add_argument('--dir', type=str, help='Root directory containing all book directories')
    parser.add_argument('--book', type=str, help='Process a single book directory')
    
    args = parser.parse_args()
    
    if args.dir:
        process_all_books(args.dir)
    elif args.book:
        process_single_book(args.book)
    else:
        print("Please provide either --dir or --book argument")