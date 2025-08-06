import json
import re
import os
import glob


def convert_jsonl_to_json_array(input_file, output_file):
    json_objects = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    json_obj = json.loads(line)
                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {input_file}: {e}")
                    continue
    
    # Write as JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_objects, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(json_objects)} objects from {os.path.basename(input_file)} to {os.path.basename(output_file)}")
    return len(json_objects)

def process_all_year_files():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use relative paths based on script location
    input_dir = os.path.join(script_dir, 'year_wise_data')
    output_dir = os.path.join(script_dir, 'year_wise_data_arrays')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files in the year_wise_data directory
    json_files = glob.glob(os.path.join(input_dir, 'dblp_*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    total_files = len(json_files)
    total_records = 0
    processed_files = 0
    
    print(f"Found {total_files} JSON files to process...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    for input_file in sorted(json_files):
        try:
            # Extract filename without path and extension
            filename = os.path.basename(input_file)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Create output filename with _array suffix
            output_filename = f"{name_without_ext}_array.json"
            output_file = os.path.join(output_dir, output_filename)
            
            # Convert the file
            records_count = convert_jsonl_to_json_array(input_file, output_file)
            total_records += records_count
            processed_files += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Files processed: {processed_files}/{total_files}")
    print(f"Total records converted: {total_records}")
    print(f"Output files saved in: {output_dir}")

# Run the batch processing
if __name__ == "__main__":
    process_all_year_files()