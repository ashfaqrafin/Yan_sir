"""
Citation Graph Matrix Visualizer

This script provides a clear visualization of the citation graph matrix
with researcher names and Nobel Prize winner indicators.
"""

import json

def visualize_citation_matrix(graph_file):
    """
    Display the citation matrix with researcher names and Nobel indicators
    """
    # Load the graph data
    with open(graph_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    adjacency_matrix = data['adjacency_matrix']
    researchers = data['researchers']
    nobel_winners = set(data['nobel_winners_ids'])
    n = len(adjacency_matrix)
    
    print("=" * 120)
    print("CITATION GRAPH ADJACENCY MATRIX VISUALIZATION")
    print("=" * 120)
    
    print("\nResearcher List:")
    print("-" * 60)
    for i, researcher in enumerate(researchers):
        nobel_status = "🏆 NOBEL" if i in nobel_winners else "      "
        print(f"ID {i:2d}: {researcher['name']:<25} {nobel_status}")
        if i in nobel_winners:
            print(f"      ({researcher['nobel_category']} {researcher['award_year']})")
    
    print("\n" + "=" * 120)
    print("ADJACENCY MATRIX")
    print("Matrix[i][j] = 1 means researcher i cites researcher j")
    print("=" * 120)
    
    # Print column headers (researcher IDs)
    print("     ", end="")
    for j in range(n):
        print(f"{j:2d}", end=" ")
    print()
    
    print("     ", end="")
    for j in range(n):
        nobel_indicator = "🏆" if j in nobel_winners else " "
        print(f" {nobel_indicator}", end="")
    print()
    
    # Print separator line
    print("   " + "-" * (n * 3 + 2))
    
    # Print matrix rows with researcher info
    for i in range(n):
        nobel_indicator = "🏆" if i in nobel_winners else " "
        print(f"{i:2d}{nobel_indicator}|", end=" ")
        
        for j in range(n):
            if adjacency_matrix[i][j] == 1:
                print(" ●", end=" ")  # Filled circle for citation
            else:
                print(" ·", end=" ")  # Small dot for no citation
        
        # Add researcher name at the end of row
        researcher_name = researchers[i]['name']
        if len(researcher_name) > 25:
            researcher_name = researcher_name[:22] + "..."
        print(f"  {researcher_name}")
    
    print("\n" + "=" * 120)
    
    # Summary statistics
    total_citations = sum(sum(row) for row in adjacency_matrix)
    nobel_citations_received = sum(
        sum(adjacency_matrix[i][j] for i in range(n))
        for j in nobel_winners
    )
    nobel_citations_made = sum(
        sum(adjacency_matrix[i])
        for i in nobel_winners
    )
    
    print("SUMMARY STATISTICS:")
    print("-" * 60)
    print(f"Total Researchers: {n}")
    print(f"Nobel Prize Winners: {len(nobel_winners)}")
    print(f"Total Citations: {total_citations}")
    print(f"Citations to Nobel Winners: {nobel_citations_received}")
    print(f"Citations by Nobel Winners: {nobel_citations_made}")
    print(f"Graph Density: {total_citations / (n * (n - 1)):.3f}")
    
    # Most cited researchers
    citations_received = [(i, sum(adjacency_matrix[j][i] for j in range(n))) 
                         for i in range(n)]
    citations_received.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost Cited Researchers:")
    print("-" * 60)
    for i, (researcher_id, citations) in enumerate(citations_received[:5]):
        researcher = researchers[researcher_id]
        nobel_status = "🏆" if researcher_id in nobel_winners else ""
        print(f"{i+1}. {researcher['name']} {nobel_status} - {citations} citations")
    
    # Most citing researchers
    citations_made = [(i, sum(adjacency_matrix[i])) for i in range(n)]
    citations_made.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost Citing Researchers:")
    print("-" * 60)
    for i, (researcher_id, citations) in enumerate(citations_made[:5]):
        researcher = researchers[researcher_id]
        nobel_status = "🏆" if researcher_id in nobel_winners else ""
        print(f"{i+1}. {researcher['name']} {nobel_status} - {citations} citations made")
    
    print("\n" + "=" * 120)
    
    # Citation patterns
    print("CITATION PATTERNS:")
    print("-" * 60)
    
    # Find all citations involving Nobel winners
    print("Citations involving Nobel Prize winners:")
    print("(Format: Citing Researcher → Cited Researcher)")
    print()
    
    nobel_citations = []
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i][j] == 1 and (i in nobel_winners or j in nobel_winners):
                citing = researchers[i]['name']
                cited = researchers[j]['name']
                citing_nobel = "🏆" if i in nobel_winners else ""
                cited_nobel = "🏆" if j in nobel_winners else ""
                nobel_citations.append((i, j, citing, cited, citing_nobel, cited_nobel))
    
    for i, j, citing, cited, citing_nobel, cited_nobel in nobel_citations:
        print(f"  {citing} {citing_nobel} → {cited} {cited_nobel}")
    
    print(f"\nTotal citations involving Nobel winners: {len(nobel_citations)}")
    print("=" * 120)


def main():
    # Find the most recent graph file
    import glob
    import os
    
    pattern = "citation_graph_matrix_*.json"
    files = glob.glob(pattern)
    
    if not files:
        print("No citation graph files found. Please run simple_citation_graph_generator.py first.")
        return
    
    # Use the most recent file
    graph_file = max(files, key=os.path.getctime)
    print(f"Visualizing graph from: {graph_file}\n")
    
    # Visualize the matrix
    visualize_citation_matrix(graph_file)


if __name__ == "__main__":
    main()
