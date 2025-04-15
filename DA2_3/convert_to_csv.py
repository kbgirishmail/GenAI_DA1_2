import json
import csv

# Input and output file paths
input_file = 'train_expanded.json'  # Change this to your input file name
output_file = 'Maktek_faq.csv'   # Change this to your desired output file name

# Convert JSONL to CSV
def convert_jsonl_to_csv(input_path, output_path):
    # Open the output CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Create CSV writer
        csv_writer = csv.writer(csvfile)
        
        # Write the header row
        csv_writer.writerow(['Question', 'Answer'])
        
        # Read the JSONL file line by line
        with open(input_path, 'r', encoding='utf-8') as jsonlfile:
            for line in jsonlfile:
                # Parse the JSON object from each line
                if line.strip():  # Skip empty lines
                    faq_item = json.loads(line)
                    
                    # Write the question and answer to the CSV
                    csv_writer.writerow([faq_item['question'], faq_item['answer']])
    
    print(f"Conversion completed! CSV file saved as '{output_path}'")

# Execute the conversion
try:
    convert_jsonl_to_csv(input_file, output_file)
except FileNotFoundError:
    print(f"Input file '{input_file}' not found. Please check the file path.")
except json.JSONDecodeError as e:
    print(f"Error parsing JSON data: {e}")
except Exception as e:
    print(f"An error occurred: {e}")