import json
import re
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Use relative paths based on script location
input_file = os.path.join(script_dir, 'dblp_v14.json')
output_dir = os.path.join(script_dir, 'year_wise_data')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

count = 0
error_count = 0
year_counts = {}
open_files = {}

print(f"Extracting records year-wise...")
print(f"Input file: {input_file}")
print(f"Output directory: {output_dir}")

def extract_json_objects(line):
    objects = []
    brace_count = 0
    start_pos = 0
    
    for i, char in enumerate(line):
        if char == '{':
            if brace_count == 0:
                start_pos = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = line[start_pos:i+1]
                try:
                    obj = json.loads(json_str)
                    objects.append(obj)
                except json.JSONDecodeError:
                    continue
    
    return objects

def get_year_file(year):
    """Get or create file handle for a specific year"""
    if year not in open_files:
        filename = f"dblp_{year}.json"
        filepath = os.path.join(output_dir, filename)
        open_files[year] = open(filepath, 'w', encoding='utf-8')
        print(f"Created new file: {filename}")
    return open_files[year]

try:
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            count += 1
            try:
                # Try to parse as single JSON object first
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    try:
                        record = json.loads(line.strip())
                        records = [record]
                    except json.JSONDecodeError:
                        # If single JSON fails, try to extract multiple objects
                        records = extract_json_objects(line.strip())
                else:
                    records = extract_json_objects(line.strip())
                
                for record in records:
                    year = record.get('year')
                    if year:
                        try:
                            year_int = int(year)
                            
                            # Get the appropriate file for this year
                            year_file = get_year_file(year_int)
                            
                            # Write the record to the year-specific file
                            year_file.write(json.dumps(record) + '\n')
                            
                            # Update year count
                            year_counts[year_int] = year_counts.get(year_int, 0) + 1
                            
                            # Print sample records for first few years
                            if year_counts[year_int] <= 3:
                                title = record.get('title', 'N/A')
                                print(f"Year {year_int}, Record {year_counts[year_int]}: {title[:50]}...")
                        
                        except ValueError:
                            # Skip records with invalid year format
                            continue
                    
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"Error on line {count}: {str(e)[:100]}")
            
            if count % 100000 == 0:
                total_extracted = sum(year_counts.values())
                print(f"Processed {count} lines, extracted {total_extracted} records, errors: {error_count}")
                print(f"Years found so far: {sorted(year_counts.keys())}")

finally:
    # Close all open files
    for year, file_handle in open_files.items():
        file_handle.close()

# Print summary
total_extracted = sum(year_counts.values())
print(f"\nDone! Processed {count} lines, extracted {total_extracted} records, errors: {error_count}")
print(f"Files created for years: {sorted(year_counts.keys())}")
print(f"\nRecords per year:")
for year in sorted(year_counts.keys()):
    print(f"  {year}: {year_counts[year]} records")

# Verify output files
print(f"\nOutput files in {output_dir}:")
for filename in sorted(os.listdir(output_dir)):
    if filename.endswith('.json'):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"  {filename}: {line_count} records")
