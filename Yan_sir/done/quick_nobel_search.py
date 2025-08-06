import json
import os
from datetime import datetime

def load_nobel_names():
    """Load Nobel prize data and create a lookup set"""
    with open("nobel.json", 'r', encoding='utf-8') as f:
        nobel_data = json.load(f)
    
    nobel_lookup = {}
    for entry in nobel_data:
        if entry.get('name'):
            # Create multiple lookup keys for name matching
            names_to_check = [
                entry['name'].strip(),
                entry.get('fullName', '').strip(),
                entry.get('knownName', '').strip()
            ]
            
            for name in names_to_check:
                if name:
                    nobel_lookup[name.lower()] = {
                        'name': entry['name'],
                        'fullName': entry.get('fullName', ''),
                        'awardYear': entry.get('awardYear', ''),
                        'category': entry.get('category', ''),
                        'id': entry.get('id', '')
                    }
    
    print(f"Loaded {len(nobel_data)} Nobel laureates, created {len(nobel_lookup)} name lookups")
    return nobel_lookup

def quick_search():
    """Quick search through a few sample files to test"""
    nobel_lookup = load_nobel_names()
    
    # Sample a few files to process quickly
    sample_files = [
        "dblp_1950_array.json", "dblp_1960_array.json", "dblp_1970_array.json", 
        "dblp_1980_array.json", "dblp_1990_array.json", "dblp_2000_array.json",
        "dblp_2010_array.json", "dblp_2020_array.json"
    ]
    
    found_authors = {}
    array_folder = "year_wise_data_arrays"
    
    for filename in sample_files:
        file_path = os.path.join(array_folder, filename)
        if not os.path.exists(file_path):
            continue
            
        print(f"Processing {filename}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for paper in data:
                if 'authors' in paper:
                    for author in paper['authors']:
                        author_name = author.get('name', '').strip()
                        if author_name.lower() in nobel_lookup:
                            nobel_info = nobel_lookup[author_name.lower()]
                            
                            if author_name not in found_authors:
                                found_authors[author_name] = {
                                    'nobel_info': nobel_info,
                                    'papers': [],
                                    'files_found_in': []
                                }
                            
                            found_authors[author_name]['papers'].append({
                                'title': paper.get('title', ''),
                                'year': paper.get('year', '')
                            })
                            
                            if filename not in found_authors[author_name]['files_found_in']:
                                found_authors[author_name]['files_found_in'].append(filename)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create output
    output = {
        'metadata': {
            'search_date': datetime.now().isoformat(),
            'search_type': 'quick_sample',
            'files_searched': sample_files,
            'total_nobel_authors_found': len(found_authors)
        },
        'found_authors': found_authors
    }
    
    with open('nobel_authors_quick_search.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nQuick search completed!")
    print(f"Found {len(found_authors)} Nobel laureates in sample files")
    print("Results saved to: nobel_authors_quick_search.json")
    
    for name, details in found_authors.items():
        print(f"- {name}: {len(details['papers'])} papers ({details['nobel_info']['category']} {details['nobel_info']['awardYear']})")

if __name__ == "__main__":
    quick_search()
