import json
import os

# Quick test script
print("Testing Nobel laureate search...")

# Load a few Nobel names
with open("nobel.json", 'r', encoding='utf-8') as f:
    nobel_data = json.load(f)

print(f"Loaded {len(nobel_data)} Nobel laureates")

# Get first 10 Nobel names as test
test_names = []
for i, entry in enumerate(nobel_data[:10]):
    if entry.get('name'):
        test_names.append(entry['name'].lower())
        print(f"{i+1}. {entry['name']} ({entry.get('category')} {entry.get('awardYear')})")

print(f"\nTesting with {len(test_names)} Nobel names...")

# Check one small file
test_file = "year_wise_data_arrays/dblp_1950_array.json"
if os.path.exists(test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} papers from {test_file}")
    
    matches = 0
    for paper in data:
        if 'authors' in paper:
            for author in paper['authors']:
                author_name = author.get('name', '').lower()
                if author_name in test_names:
                    matches += 1
                    print(f"MATCH FOUND: {author.get('name')} in paper: {paper.get('title', '')[:50]}...")
    
    print(f"\nTotal matches found: {matches}")
else:
    print(f"Test file {test_file} not found")

# Save a simple result
result = {
    "test_completed": True,
    "nobel_laureates_tested": len(test_names),
    "test_file": test_file,
    "matches_found": matches if 'matches' in locals() else 0
}

with open('test_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("Test completed, results saved to test_result.json")
