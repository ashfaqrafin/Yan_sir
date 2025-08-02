import json
import os
from datetime import datetime

def main():
    print("Starting Nobel laureate search...")
    
    # Load Nobel data
    print("Loading Nobel prize data...")
    with open("nobel.json", 'r', encoding='utf-8') as f:
        nobel_data = json.load(f)
    
    # Create name lookup
    nobel_names = set()
    nobel_info = {}
    
    for entry in nobel_data:
        if entry.get('name'):
            name = entry['name'].strip().lower()
            nobel_names.add(name)
            nobel_info[name] = {
                'original_name': entry['name'],
                'fullName': entry.get('fullName', ''),
                'awardYear': entry.get('awardYear', ''),
                'category': entry.get('category', ''),
                'id': entry.get('id', ''),
                'motivation': entry.get('motivation', '')[:100] + '...' if entry.get('motivation') else ''
            }
            
            # Also add fullName if different
            if entry.get('fullName') and entry['fullName'].strip().lower() != name:
                full_name = entry['fullName'].strip().lower()
                nobel_names.add(full_name)
                nobel_info[full_name] = nobel_info[name]
    
    print(f"Loaded {len(nobel_data)} Nobel entries, {len(nobel_names)} unique names")
    
    # Search in year-wise data
    found_authors = {}
    array_folder = "year_wise_data_arrays"
    
    # Get list of files
    array_files = [f for f in os.listdir(array_folder) if f.endswith('_array.json')]
    
    # Process every 10th file to get a representative sample
    sample_files = sorted(array_files)[::10]  # Every 10th file
    
    print(f"Processing {len(sample_files)} sample files out of {len(array_files)} total files...")
    
    for filename in sample_files:
        file_path = os.path.join(array_folder, filename)
        year = filename.replace('dblp_', '').replace('_array.json', '')
        
        print(f"Processing {filename}...", end=' ')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"({len(data)} papers)")
            
            for paper in data:
                if 'authors' in paper:
                    for author in paper['authors']:
                        author_name = author.get('name', '').strip().lower()
                        if author_name in nobel_names:
                            if author_name not in found_authors:
                                found_authors[author_name] = {
                                    'nobel_info': nobel_info[author_name],
                                    'papers_found': 0,
                                    'years_found': set(),
                                    'sample_papers': []
                                }
                            
                            found_authors[author_name]['papers_found'] += 1
                            found_authors[author_name]['years_found'].add(year)
                            
                            # Keep only first 3 papers as samples
                            if len(found_authors[author_name]['sample_papers']) < 3:
                                found_authors[author_name]['sample_papers'].append({
                                    'title': paper.get('title', ''),
                                    'year': paper.get('year', year),
                                    'file': filename
                                })
        
        except Exception as e:
            print(f"Error: {e}")
    
    # Convert sets to lists for JSON
    for author_data in found_authors.values():
        author_data['years_found'] = sorted(list(author_data['years_found']))
    
    # Create output
    output = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'total_nobel_laureates': len(nobel_data),
            'files_processed': len(sample_files),
            'total_files_available': len(array_files),
            'search_method': 'sample_every_10th_file',
            'nobel_laureates_found': len(found_authors)
        },
        'summary': {
            'total_found': len(found_authors),
            'total_papers_by_nobel_laureates': sum(a['papers_found'] for a in found_authors.values())
        },
        'found_nobel_laureates': found_authors
    }
    
    # Save results
    output_filename = 'nobel_laureates_in_dblp_sample.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY:")
    print(f"{'='*60}")
    print(f"Nobel laureates found: {len(found_authors)}")
    print(f"Total papers found: {sum(a['papers_found'] for a in found_authors.values())}")
    print(f"Results saved to: {output_filename}")
    
    if found_authors:
        print(f"\nFound Nobel laureates:")
        for i, (key, data) in enumerate(found_authors.items(), 1):
            info = data['nobel_info']
            print(f"{i:2d}. {info['original_name']} - {data['papers_found']} papers")
            print(f"     {info['category']} {info['awardYear']}")
            print(f"     Years: {', '.join(data['years_found'])}")
            print()
    
    print(f"Detailed results saved in '{output_filename}'")

if __name__ == "__main__":
    main()
