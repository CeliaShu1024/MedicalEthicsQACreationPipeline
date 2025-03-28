import os
import json
import argparse
import time
import random
import datetime
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

def find_coi_files(base_directory="src/book_files", model_name="deepseek-r1", bookid=None):
    """
    Find all content of interest (COI) files in the extracted directory.
    
    Args:
        base_directory: Base directory containing book folders
        model_name: Model name used in COI filenames
        bookid: Optional book ID to filter by
        
    Returns:
        List of COI file paths
    """
    if not os.path.isdir(base_directory):
        print(f"Warning: Base directory {base_directory} does not exist")
        return []
    
    coi_files = []
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith(f"_coi_{model_name}.json"):
                # If bookid is specified, only include files for that book
                if bookid and not file.startswith(bookid):
                    continue
                coi_files.append(os.path.join(root, file))
    
    return coi_files

def load_coi_file(file_path):
    """
    Load a COI file.
    
    Args:
        file_path: Path to the COI file
        
    Returns:
        List of content items
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        return content
    except Exception as e:
        print(f"Error loading COI file {file_path}: {e}")
        return []

def generate_questions_with_openai(content_item, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen2.5-7B", num_questions=1, max_retries=3):
    """
    Generate exam questions using API.
    
    Args:
        content_item: Dictionary containing heading and text content
        api_key: API key
        base_url: Base URL for the API
        model: Model to use
        num_questions: Number of questions to generate
        max_retries: Maximum number of retries for API calls
        
    Returns:
        List of generated questions (each with question, options, and correct answer)
    """
    # Create OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Combine heading and text
    all_text = content_item["heading"] + "\n\n" + "\n\n".join(content_item["text"])
    
    # Prepare system prompt
    system_prompt = """
    You are an expert in healthcare ethics and medical education. Your task is to create challenging multiple-choice exam questions for practicing physicians based on the provided passage. 
    
    Guidelines for question creation:
    1. Focus on healthcare ethics concepts, principles, and applications
    2. Create questions that require critical thinking and application of ethical principles
    3. Questions should be at an appropriate difficulty level for board certification exams
    4. Include 4 answer options (A, B, C, D) with only one correct answer
    5. Make distractors plausible but clearly incorrect upon careful analysis
    6. Ensure the question is directly related to the provided content
    
    Return each question in the following JSON format:
    {
        "question": "The full question text",
        "options": {
            "A": "First option text",
            "B": "Second option text",
            "C": "Third option text",
            "D": "Fourth option text"
        },
        "correct_answer": "The letter of the correct option (A, B, C, or D)",
        "explanation": "A detailed explanation of why the correct answer is right and why the other options are wrong"
    }
    """
    
    # Prepare user message
    user_message = f"""
    Please create {num_questions} multiple-choice question(s) based on the following passage about healthcare ethics:
    
    {all_text}
    
    Return the question(s) in the specified JSON format.
    """
    
    # Implement retries
    for retry in range(max_retries):
        try:
            # Make API call
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Extract and parse response
            response_content = response.choices[0].message.content
            questions_data = json.loads(response_content)
            
            # Handle both single question and multiple questions formats
            if "question" in questions_data:
                # Single question format
                return [questions_data]
            elif "questions" in questions_data:
                # Multiple questions format
                return questions_data["questions"]
            else:
                # Try to determine if it's a list or dictionary with numeric keys
                if isinstance(questions_data, list):
                    return questions_data
                elif all(key.isdigit() for key in questions_data.keys()):
                    return list(questions_data.values())
                else:
                    print(f"Unexpected response format: {response_content[:100]}...")
                    # Continue to retry since format was unexpected
                    if retry < max_retries - 1:
                        time.sleep(2)  # Add delay before retry
                        continue
                    return []
                
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error generating questions (attempt {retry+1}/{max_retries}): {e}")
                time.sleep(2)  # Add delay before retry
            else:
                print(f"Failed to generate questions after {max_retries} attempts: {e}")
                return []
    
    return []  # Return empty list if all retries failed

def sample_content_items(content_items, num_items):
    """
    Sample a specified number of content items.
    
    Args:
        content_items: List of content items
        num_items: Number of items to sample (0 for all items)
        
    Returns:
        List of sampled content items
    """
    # If num_items is 0 or greater than length, return all items
    if num_items <= 0 or num_items >= len(content_items):
        return content_items
    
    return random.sample(content_items, num_items)

def process_content_item(item, bookid, api_key, base_url, model, questions_per_item, max_retries):
    """
    Process a single content item to generate questions.
    This function is designed to be run in a thread pool.
    
    Args:
        item: Content item dictionary
        bookid: Book ID
        api_key: API key
        base_url: Base URL for the API
        model: Model to use
        questions_per_item: Number of questions to generate per content item
        max_retries: Maximum number of retries for API calls
        
    Returns:
        List of questions with source information
    """
    questions = generate_questions_with_openai(
        item, 
        api_key,
        base_url,
        model, 
        questions_per_item,
        max_retries
    )
    
    # Add source information to each question
    for question in questions:
        question["source"] = {
            "bookid": bookid,
            "heading": item["heading"]
        }
    
    return questions

def process_coi_file(coi_file, api_key, base_url, model, items_per_file, questions_per_item, max_retries, num_threads=1):
    """
    Process a single COI file to generate questions.
    Handles multithreading for content items within a file.
    
    Args:
        coi_file: Path to the COI file
        api_key: API key
        base_url: Base URL for the API
        model: Model to use
        items_per_file: Number of content items to sample from each file
        questions_per_item: Number of questions to generate per content item
        max_retries: Maximum number of retries for API calls
        num_threads: Number of threads to use for processing content items
        
    Returns:
        Tuple of (bookid, list of questions)
    """
    # Extract book ID from filename
    bookid = os.path.basename(coi_file).split("_coi_")[0]
    
    # Load content items
    content_items = load_coi_file(coi_file)
    if not content_items:
        return bookid, []
    
    # Sample content items
    sampled_items = sample_content_items(content_items, items_per_file)
    
    all_questions = []
    
    # Use ThreadPoolExecutor to process content items in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create a list to hold future objects
        future_to_item = {
            executor.submit(
                process_content_item, 
                item, 
                bookid, 
                api_key, 
                base_url, 
                model, 
                questions_per_item, 
                max_retries
            ): item for item in sampled_items
        }
        
        # Process completed futures with tqdm for progress tracking
        for future in tqdm(
                concurrent.futures.as_completed(future_to_item), 
                total=len(sampled_items),
                desc=f"Generating questions for {bookid}",
                leave=False
            ):
            try:
                questions = future.result()
                all_questions.extend(questions)
            except Exception as e:
                print(f"Error processing content item: {e}")
    
    return bookid, all_questions

def create_exam(coi_files, output_dir, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen2.5-7B", 
               items_per_file=5, questions_per_item=1, max_retries=3, num_threads=1, parallel_books=False):
    """
    Create exams from COI files, one exam per book.
    
    Args:
        coi_files: List of COI file paths
        output_dir: Directory to save exam files
        api_key: API key
        base_url: Base URL for the API
        model: Model to use
        items_per_file: Number of content items to sample from each file
        questions_per_item: Number of questions to generate per content item
        max_retries: Maximum number of retries for API calls
        num_threads: Number of threads to use for processing content items
        parallel_books: If True, process books in parallel; otherwise process content items in parallel
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for logging purposes
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting exam creation at {timestamp}")
    
    # Create a dict to store the processed books info
    processed_books = {}
    
    if parallel_books and len(coi_files) > 1:
        # Process books in parallel
        print(f"Processing {len(coi_files)} books in parallel using {num_threads} threads")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Process each book in its own thread
            future_to_book = {
                executor.submit(
                    process_coi_file, 
                    coi_file, 
                    api_key, 
                    base_url, 
                    model, 
                    items_per_file, 
                    questions_per_item, 
                    max_retries, 
                    1  # Use 1 thread per content item when parallelizing books
                ): coi_file for coi_file in coi_files
            }
            
            # Process completed futures with tqdm for progress tracking
            for future in tqdm(
                    concurrent.futures.as_completed(future_to_book), 
                    total=len(coi_files),
                    desc="Processing books"
                ):
                try:
                    bookid, questions = future.result()
                    
                    if questions:
                        # Save questions to a book-specific file in the output directory
                        questions_file = os.path.join(output_dir, f"{bookid}_questions.json")
                        with open(questions_file, 'w', encoding='utf-8') as f:
                            json.dump(questions, f, ensure_ascii=False, indent=2)
                        
                        # Create book-specific metadata
                        metadata = {
                            "bookid": bookid,
                            "timestamp": timestamp,
                            "model": model,
                            "total_questions": len(questions),
                            "items_sampled": items_per_file,
                            "questions_per_item": questions_per_item
                        }
                        
                        # Save metadata to a book-specific file
                        metadata_file = os.path.join(output_dir, f"{bookid}_data.json")
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                        
                        # Add to processed books
                        processed_books[bookid] = {
                            "questions_file": questions_file,
                            "metadata_file": metadata_file,
                            "question_count": len(questions)
                        }
                        
                        print(f"Processed {bookid}: {len(questions)} questions")
                    else:
                        print(f"No questions generated for {bookid}")
                        
                except Exception as e:
                    print(f"Error processing book: {e}")
    else:
        # Process books sequentially, but content items in parallel
        for coi_file in tqdm(coi_files, desc="Processing books"):
            try:
                bookid, questions = process_coi_file(
                    coi_file, 
                    api_key, 
                    base_url, 
                    model, 
                    items_per_file, 
                    questions_per_item, 
                    max_retries, 
                    num_threads
                )
                
                if questions:
                    # Save questions to a book-specific file in the output directory
                    questions_file = os.path.join(output_dir, f"{bookid}_questions.json")
                    with open(questions_file, 'w', encoding='utf-8') as f:
                        json.dump(questions, f, ensure_ascii=False, indent=2)
                    
                    # Create book-specific metadata
                    metadata = {
                        "bookid": bookid,
                        "timestamp": timestamp,
                        "model": model,
                        "total_questions": len(questions),
                        "items_sampled": items_per_file,
                        "questions_per_item": questions_per_item
                    }
                    
                    # Save metadata to a book-specific file
                    metadata_file = os.path.join(output_dir, f"{bookid}_data.json")
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    # Add to processed books
                    processed_books[bookid] = {
                        "questions_file": questions_file,
                        "metadata_file": metadata_file,
                        "question_count": len(questions)
                    }
                    
                    print(f"Processed {bookid}: {len(questions)} questions")
                else:
                    print(f"No questions generated for {bookid}")
            
            except Exception as e:
                print(f"Error processing book: {e}")
    
    # Save a summary of all processed books
    if processed_books:
        summary = {
            "timestamp": timestamp,
            "total_books": len(processed_books),
            "total_questions": sum(book_info["question_count"] for book_info in processed_books.values()),
            "books": {bookid: book_info["question_count"] for bookid, book_info in processed_books.items()}
        }
        
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Exam creation complete. Processed {len(processed_books)} books with {summary['total_questions']} total questions")
        print(f"Summary saved to: {summary_file}")
    else:
        print("No exams were created successfully")
    
    return processed_books

def get_previously_used_coi_files(history_file):
    """
    Get a set of previously used COI files from the history file.
    
    Args:
        history_file: Path to the history file
        
    Returns:
        Set of previously used COI file paths
    """
    if not os.path.exists(history_file):
        return set()
        
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        return set(history.get('used_coi_files', []))
    except Exception as e:
        print(f"Error loading history file: {e}")
        return set()

def update_coi_history(history_file, used_files):
    """
    Update the COI file history with newly used files.
    
    Args:
        history_file: Path to the history file
        used_files: List of COI file paths that were used
    """
    # Get existing history
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = {'used_coi_files': []}
    else:
        history = {'used_coi_files': []}
    
    # Add new files to history
    history['used_coi_files'] = list(set(history['used_coi_files'] + used_files))
    
    # Write updated history
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error updating history file: {e}")

def main():
    """Main function to parse arguments and run the program."""
    parser = argparse.ArgumentParser(description='Create exam questions from content of interest files')
    parser.add_argument('--dir', type=str, default='src/book_files', help='Base directory containing book folders')
    parser.add_argument('--output', type=str, default='src/exams', help='Directory to save exam files')
    parser.add_argument('--api-key', type=str, help='API key')
    parser.add_argument('--base-url', type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1', help='API base URL')
    parser.add_argument('--model', type=str, default='qwen2.5-7B', help='Model to use')
    parser.add_argument('--model-name', type=str, default='deepseek-r1', help='Name used in COI filenames')
    parser.add_argument('--items-per-file', type=int, default=5, help='Number of content items to sample from each file (0 for all items)')
    parser.add_argument('--questions-per-item', type=int, default=1, help='Number of questions to generate per content item')
    parser.add_argument('--pdf', action='store_true', help='Generate PDF of the exam')
    parser.add_argument('--book', type=str, help='Process a single book by its ID')
    parser.add_argument('--books', type=str, help='Comma-separated list of book IDs to process')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use for parallel processing')
    parser.add_argument('--parallel-books', action='store_true', help='Process books in parallel instead of content items')
    parser.add_argument('--skip-used', action='store_true', help='Skip COI files that have been used in previous exams')
    parser.add_argument('--force-reuse', action='store_true', help='Force reuse of COI files even if they have been used before')
    parser.add_argument('--history-file', type=str, default='coi_history.json', help='File to track COI usage history')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        print("Error: API key not provided. Use --api-key or set OPENAI_API_KEY/API_KEY environment variable.")
        return
        
    # Set up base URL from argument
    base_url = args.base_url
    print(f"Using API base URL: {base_url}")
    
    # Process specific books
    if args.book or args.books:
        book_ids = []
        if args.book:
            book_ids.append(args.book)
        if args.books:
            book_ids.extend([b.strip() for b in args.books.split(',')])
        
        # Deduplicate book IDs
        book_ids = list(set(book_ids))
        
        coi_files = []
        for book_id in book_ids:
            book_coi_path = None
            # Look for the COI file for the specified book
            for root, _, files in os.walk(args.dir):
                for file in files:
                    if file.startswith(book_id) and file.endswith(f"_coi_{args.model_name}.json"):
                        book_coi_path = os.path.join(root, file)
                        break
                if book_coi_path:
                    break
            
            if book_coi_path:
                coi_files.append(book_coi_path)
                print(f"Found COI file for book: {book_id}")
            else:
                print(f"Warning: No COI file found for book {book_id}")
        
        if not coi_files:
            print(f"No COI files found for the specified books")
            return
        
        print(f"Processing {len(coi_files)} books")
    else:
        # Find all COI files
        coi_files = find_coi_files(args.dir, args.model_name)
        if not coi_files:
            print(f"No COI files found in {args.dir}")
            return
        
        print(f"Found {len(coi_files)} COI files")
    
    # Check for previously used COI files if --skip-used is specified
    if args.skip_used and not args.force_reuse:
        previously_used = get_previously_used_coi_files(args.history_file)
        original_count = len(coi_files)
        
        # Filter out previously used files
        coi_files = [file for file in coi_files if file not in previously_used]
        
        skipped_count = original_count - len(coi_files)
        if skipped_count > 0:
            print(f"Skipped {skipped_count} previously used COI files")
            
        if not coi_files:
            print("All COI files have been used previously. Use --force-reuse to reuse files.")
            return
    
    # Display threading configuration
    if args.threads > 1:
        if args.parallel_books:
            print(f"Using {args.threads} threads to process books in parallel")
        else:
            print(f"Using {args.threads} threads to process content items in parallel")
    else:
        print("Running in single-threaded mode")
    
    # Create exams
    processed_books = create_exam(
        coi_files,
        args.output,
        api_key,
        args.base_url,
        args.model,
        args.items_per_file,
        args.questions_per_item,
        max_retries=3,
        num_threads=args.threads,
        parallel_books=args.parallel_books
    )
    
    # Generate PDFs if requested
    if args.pdf and processed_books:
        for bookid in processed_books:
            pdf_path = generate_exam_pdf(args.output, bookid)
            if pdf_path:
                print(f"Generated PDF for {bookid}: {pdf_path}")
    
    # Update COI history file with newly used files if skip-used option is enabled
    if (args.skip_used or args.force_reuse) and processed_books:
        update_coi_history(args.history_file, coi_files)
        print(f"Updated COI history file: {args.history_file}")

if __name__ == "__main__":
    main()