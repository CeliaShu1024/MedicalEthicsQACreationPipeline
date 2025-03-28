import json
import os
import time
import argparse
from openai import OpenAI

def answer_questions_with_model(questions, api_key, base_url="https://api.deepseek.com", model="deepseek-chat", max_retries=3):
    """
    Have the model answer exam questions via OpenAI-compatible API.
    
    Args:
        questions: List of question dictionaries to answer
        api_key: API key for the model
        base_url: Base URL for the API
        model: Model to use
        max_retries: Maximum number of retries for API calls
        
    Returns:
        List of results with original questions and model answers
    """
    # Create OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    results = []
    
    # System prompt to guide the model
    system_prompt = """
    You are presented with a multiple-choice question in the healthcare ethics domain.
    Your task is to carefully analyze the question and options, then select the option you believe is correct.
    Showing your reasoning process will be considered a false answer. Please provide ONLY the letter of your answer (A, B, C, or D). DO NOT provide any additional information. 
    """
    
    # Process each question
    for i, question_data in enumerate(questions):
        # Extract just the question and options (without the correct answer or explanation)
        question_text = question_data["question"]
        options = question_data["options"]
        
        # Format the options for presentation
        formatted_options = "\n".join([f"{key}: {value}" for key, value in options.items()])
        
        user_message = f"""
        Please answer the following multiple-choice question:
        
        Question: {question_text}
        
        Options:
        {formatted_options}
        
        Please respond with just the letter of your answer (A, B, C, or D).
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
                    temperature=0.1  # Low temperature for more deterministic answers
                )
                
                # Extract response
                response_content = response.choices[0].message.content.strip()
                
                # Try to extract just the letter answer if there's additional text
                # This handles cases where the model outputs "The answer is B" instead of just "B"
                answer_letter = None
                for letter in ["A", "B", "C", "D"]:
                    if response_content == letter or response_content.endswith(f"is {letter}.") or response_content.endswith(f"is {letter}"):
                        answer_letter = letter
                        break
                
                # If we couldn't extract a clean letter, use the whole response
                if answer_letter is None:
                    answer_letter = response_content
                
                # Create the result
                result = {
                    "original_question": question_data,
                    "model_answer": answer_letter,
                    "correct_answer": question_data["correct_answer"],
                    "is_correct": answer_letter == question_data["correct_answer"]
                }
                
                results.append(result)
                
                # Log progress
                print(f"Question {i+1}/{len(questions)}: Model answered {answer_letter}, Correct: {question_data['correct_answer']}, Match: {answer_letter == question_data['correct_answer']}")
                
                # Break out of retry loop on success
                break
                
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error answering question {i+1} (attempt {retry+1}/{max_retries}): {e}")
                    time.sleep(2)  # Add delay before retry
                else:
                    print(f"Failed to answer question {i+1} after {max_retries} attempts: {e}")
                    # Add failure record to results
                    results.append({
                        "original_question": question_data,
                        "model_answer": "ERROR",
                        "correct_answer": question_data["correct_answer"],
                        "is_correct": False,
                        "error": str(e)
                    })
        
        # Add a small delay between questions to avoid rate limiting
        time.sleep(1)
    
    return results

def load_exam_questions(directory_path):
    """
    Load all JSON exam files from the specified directory.
    
    Args:
        directory_path: Path to directory containing exam JSON files
        
    Returns:
        List of all questions from all files
    """
    all_questions = []
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} not found")
        return all_questions
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    questions = json.load(f)
                    print(f"Loaded {len(questions)} questions from {filename}")
                    all_questions.extend(questions)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return all_questions

def save_results(results, output_file):
    """
    Save results to a JSON file.
    
    Args:
        results: List of results with model answers
        output_file: Path to output file
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {output_file}")
    
    # Calculate and display accuracy
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\nAccuracy Summary:")
    print(f"Total questions: {total}")
    print(f"Correctly answered: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test LLM performance on healthcare ethics exam questions')
    parser.add_argument('--api_key', '--api-key', required=True, help='API key for the model')
    parser.add_argument('--base_url', '--base-url', default='https://api.deepseek.com', help='Base URL for the API')
    parser.add_argument('--model', default='deepseek-chat', help='Model to use')
    parser.add_argument('--input_dir', '--dir', default='src/exams', help='Directory containing exam JSON files')
    parser.add_argument('--output_file', '--out', default='model_answers.json', help='Output file for results')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions to process (for testing)')
    
    args = parser.parse_args()
    
    # Load questions from files
    all_questions = load_exam_questions(args.input_dir)
    print(f"Loaded a total of {len(all_questions)} questions from all files")
    
    # Apply limit if specified
    if args.limit and args.limit > 0 and args.limit < len(all_questions):
        all_questions = all_questions[:args.limit]
        print(f"Limited to {args.limit} questions for processing")
    
    # Answer questions
    if all_questions:
        results = answer_questions_with_model(
            questions=all_questions,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model
        )
        
        # Save results
        save_results(results, args.output_file)
    else:
        print("No questions found to process")

if __name__ == "__main__":
    main()