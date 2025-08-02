import json
import os
from pathlib import Path

def load_nobel_data(nobel_file_path):
    """Load Nobel prize data and extract author names"""
    with open(nobel_file_path, 'r', encoding='utf-8') as f:
        nobel_data = json.load(f)
    
    # Extract all possible name variations for Nobel laureates
    nobel_names = set()
    nobel_authors = []
    
    for entry in nobel_data:
        if entry.get('name'):
            nobel_authors.append({
                'name': entry['name'],
                'fullName': entry.get('fullName', ''),
                'knownName': entry.get('knownName', ''),
                'givenName': entry.get('givenName', ''),
                'familyName': entry.get('familyName', ''),
                'awardYear': entry.get('awardYear', ''),
                'category': entry.get('category', '')
            })
            
            # Add various name formats to the set for matching
            nobel_names.add(entry['name'].strip().lower())
            if entry.get('fullName'):
                nobel_names.add(entry['fullName'].strip().lower())
            if entry.get('knownName'):
                nobel_names.add(entry['knownName'].strip().lower())
    
    return nobel_names, nobel_authors

def search_authors_in_dblp_arrays(array_folder_path, nobel_names):
    """Search for Nobel laureates in DBLP year-wise data arrays"""
    found_authors = {}
    total_files_processed = 0
    
    # Get all JSON files in the array folder
    array_files = []
    if os.path.exists(array_folder_path):
        array_files = [f for f in os.listdir(array_folder_path) if f.endswith('_array.json')]
    
    print(f"Found {len(array_files)} array files to process")
    
    for filename in sorted(array_files):
        file_path = os.path.join(array_folder_path, filename)
        year = filename.replace('dblp_', '').replace('_array.json', '')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_files_processed += 1
            print(f"Processing {filename} ({len(data)} papers)...")
            
            for paper in data:
                if 'authors' in paper:
                    for author in paper['authors']:
                        author_name = author.get('name', '').strip().lower()
                        if author_name in nobel_names:
                            if author_name not in found_authors:
                                found_authors[author_name] = {
                                    'original_name': author.get('name', ''),
                                    'papers': [],
                                    'years': set()
                                }
                            
                            found_authors[author_name]['papers'].append({
                                'title': paper.get('title', ''),
                                'year': paper.get('year', year),
                                'file': filename
                            })
                            found_authors[author_name]['years'].add(str(paper.get('year', year)))
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return found_authors, total_files_processed

def main():
    # File paths
    nobel_file = "nobel.json"
    array_folder = "year_wise_data_arrays"
    
    print("Loading Nobel prize data...")
    nobel_names, nobel_authors = load_nobel_data(nobel_file)
    print(f"Loaded {len(nobel_authors)} Nobel laureates with {len(nobel_names)} name variations")
    
    print("\nSearching for Nobel laureates in DBLP data arrays...")
    found_authors, files_processed = search_authors_in_dblp_arrays(array_folder, nobel_names)
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Total Nobel laureates in database: {len(nobel_authors)}")
    print(f"Total DBLP array files processed: {files_processed}")
    print(f"Nobel laureates found in DBLP data: {len(found_authors)}")
    
    if found_authors:
        print(f"\nFound the following Nobel laureates in DBLP data:")
        print("-" * 60)
        
        for i, (author_key, details) in enumerate(found_authors.items(), 1):
            print(f"{i}. {details['original_name']}")
            print(f"   Papers found: {len(details['papers'])}")
            print(f"   Years active: {', '.join(sorted(details['years']))}")
            
            # Show first few papers as examples
            for j, paper in enumerate(details['papers'][:3]):
                print(f"   Paper {j+1}: {paper['title'][:80]}... ({paper['year']})")
            
            if len(details['papers']) > 3:
                print(f"   ... and {len(details['papers']) - 3} more papers")
            print()
    else:
        print("\nNo Nobel laureates found in the DBLP data arrays.")
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print(f"- Total unique Nobel laureates found: {len(found_authors)}")
    if found_authors:
        total_papers = sum(len(details['papers']) for details in found_authors.values())
        print(f"- Total papers by Nobel laureates: {total_papers}")
        all_years = set()
        for details in found_authors.values():
            all_years.update(details['years'])
        print(f"- Years spanning: {min(all_years)} - {max(all_years)}")

if __name__ == "__main__":
    main()
