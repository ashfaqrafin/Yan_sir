import json
import os
from pathlib import Path
from datetime import datetime

def load_nobel_data(nobel_file_path):
    """Load Nobel prize data and extract author names"""
    with open(nobel_file_path, 'r', encoding='utf-8') as f:
        nobel_data = json.load(f)
    
    # Extract all possible name variations for Nobel laureates
    nobel_names = {}  # Changed to dict to keep track of original entries
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
                'category': entry.get('category', ''),
                'id': entry.get('id', ''),
                'affiliation_1': entry.get('affiliation_1', ''),
                'birth_country': entry.get('birth_country', ''),
                'motivation': entry.get('motivation', '')
            })
            
            # Add various name formats to the dict for matching
            key_name = entry['name'].strip().lower()
            nobel_names[key_name] = entry
            if entry.get('fullName'):
                nobel_names[entry['fullName'].strip().lower()] = entry
            if entry.get('knownName'):
                nobel_names[entry['knownName'].strip().lower()] = entry
    
    return nobel_names, nobel_authors

def search_authors_in_dblp_arrays(array_folder_path, nobel_names):
    """Search for Nobel laureates in DBLP year-wise data arrays"""
    found_authors = {}
    total_files_processed = 0
    processing_log = []
    
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
            papers_count = len(data)
            print(f"Processing {filename} ({papers_count} papers)...")
            
            file_matches = 0
            for paper in data:
                if 'authors' in paper:
                    for author in paper['authors']:
                        author_name = author.get('name', '').strip().lower()
                        if author_name in nobel_names:
                            file_matches += 1
                            nobel_info = nobel_names[author_name]
                            
                            if author_name not in found_authors:
                                found_authors[author_name] = {
                                    'original_name': author.get('name', ''),
                                    'nobel_info': {
                                        'name': nobel_info.get('name', ''),
                                        'fullName': nobel_info.get('fullName', ''),
                                        'awardYear': nobel_info.get('awardYear', ''),
                                        'category': nobel_info.get('category', ''),
                                        'id': nobel_info.get('id', ''),
                                        'affiliation_1': nobel_info.get('affiliation_1', ''),
                                        'birth_country': nobel_info.get('birth_country', ''),
                                        'motivation': nobel_info.get('motivation', '')
                                    },
                                    'papers': [],
                                    'years': set(),
                                    'total_papers': 0
                                }
                            
                            found_authors[author_name]['papers'].append({
                                'title': paper.get('title', ''),
                                'year': paper.get('year', year),
                                'file': filename,
                                'doc_type': paper.get('doc_type', ''),
                                'publisher': paper.get('publisher', ''),
                                'n_citation': paper.get('n_citation', 0),
                                'doi': paper.get('doi', ''),
                                'paper_id': paper.get('id', ''),
                                'co_authors': [a.get('name', '') for a in paper.get('authors', []) if a.get('name', '').strip().lower() != author_name]
                            })
                            found_authors[author_name]['years'].add(str(paper.get('year', year)))
                            found_authors[author_name]['total_papers'] += 1
            
            processing_log.append({
                'filename': filename,
                'year': year,
                'papers_in_file': papers_count,
                'matches_found': file_matches
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            processing_log.append({
                'filename': filename,
                'year': year,
                'error': str(e)
            })
            continue
    
    # Convert sets to lists for JSON serialization
    for author_data in found_authors.values():
        author_data['years'] = sorted(list(author_data['years']))
    
    return found_authors, total_files_processed, processing_log

def create_detailed_json_output(found_authors, nobel_authors, files_processed, processing_log):
    """Create detailed JSON output with all findings"""
    
    # Summary statistics
    total_papers = sum(details['total_papers'] for details in found_authors.values())
    all_years = set()
    for details in found_authors.values():
        all_years.update(details['years'])
    
    # Create the detailed output structure
    output = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'total_nobel_laureates_in_database': len(nobel_authors),
            'total_dblp_files_processed': files_processed,
            'nobel_laureates_found_in_dblp': len(found_authors),
            'total_papers_by_nobel_laureates': total_papers,
            'year_range': {
                'earliest': min(all_years) if all_years else None,
                'latest': max(all_years) if all_years else None
            }
        },
        'processing_log': processing_log,
        'found_nobel_laureates': {},
        'summary_by_category': {},
        'summary_by_year': {}
    }
    
    # Detailed information for each found author
    for author_key, details in found_authors.items():
        output['found_nobel_laureates'][details['original_name']] = {
            'nobel_prize_info': details['nobel_info'],
            'research_activity': {
                'total_papers_found': details['total_papers'],
                'years_active': details['years'],
                'papers': details['papers']
            }
        }
    
    # Summary by Nobel category
    categories = {}
    for details in found_authors.values():
        category = details['nobel_info'].get('category', 'Unknown')
        if category not in categories:
            categories[category] = {'count': 0, 'authors': []}
        categories[category]['count'] += 1
        categories[category]['authors'].append(details['original_name'])
    output['summary_by_category'] = categories
    
    # Summary by award year
    award_years = {}
    for details in found_authors.values():
        award_year = details['nobel_info'].get('awardYear', 'Unknown')
        if award_year not in award_years:
            award_years[award_year] = {'count': 0, 'authors': []}
        award_years[award_year]['count'] += 1
        award_years[award_year]['authors'].append(details['original_name'])
    output['summary_by_year'] = award_years
    
    return output

def main():
    # File paths
    nobel_file = "nobel.json"
    array_folder = "year_wise_data_arrays"
    output_file = "nobel_authors_found_in_dblp.json"
    
    print("Loading Nobel prize data...")
    nobel_names, nobel_authors = load_nobel_data(nobel_file)
    print(f"Loaded {len(nobel_authors)} Nobel laureates with {len(nobel_names)} name variations")
    
    print("\nSearching for Nobel laureates in DBLP data arrays...")
    found_authors, files_processed, processing_log = search_authors_in_dblp_arrays(array_folder, nobel_names)
    
    print(f"\nCreating detailed JSON output...")
    detailed_output = create_detailed_json_output(found_authors, nobel_authors, files_processed, processing_log)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    print(f"Total Nobel laureates in database: {len(nobel_authors)}")
    print(f"Total DBLP array files processed: {files_processed}")
    print(f"Nobel laureates found in DBLP data: {len(found_authors)}")
    
    if found_authors:
        total_papers = sum(details['total_papers'] for details in found_authors.values())
        print(f"Total papers by Nobel laureates: {total_papers}")
        
        print(f"\nTop 10 Nobel laureates by number of papers:")
        sorted_authors = sorted(found_authors.items(), key=lambda x: x[1]['total_papers'], reverse=True)
        for i, (author_key, details) in enumerate(sorted_authors[:10], 1):
            print(f"{i:2d}. {details['original_name']}: {details['total_papers']} papers "
                  f"({details['nobel_info']['category']} {details['nobel_info']['awardYear']})")
    
    print(f"\nDetailed results have been saved to '{output_file}'")

if __name__ == "__main__":
    main()
