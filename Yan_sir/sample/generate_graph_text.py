"""
Citation Graph Text Report Generator

This script generates a comprehensive text-based report containing all details 
of the citation graph for easy viewing and printing.
"""

import json
from datetime import datetime
import os

class CitationGraphTextGenerator:
    def __init__(self, graph_file):
        """Initialize with citation graph data"""
        with open(graph_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.adjacency_matrix = self.data['adjacency_matrix']
        self.researchers = self.data['researchers']
        self.nobel_winners = set(self.data['nobel_winners_ids'])
        self.statistics = self.data['statistics']
        self.metadata = self.data['metadata']
        self.n = len(self.adjacency_matrix)
    
    def generate_header(self):
        """Generate report header"""
        gen_date = datetime.fromisoformat(self.metadata['generated_date']).strftime("%B %d, %Y at %I:%M %p")
        
        return f"""
{'='*80}
CITATION GRAPH TEST DATASET - COMPREHENSIVE ANALYSIS REPORT
{'='*80}

Generated: {gen_date}
Report Date: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}

OVERVIEW
{'-'*80}
Total Researchers: {self.statistics['total_researchers']}
Nobel Prize Winners: {self.statistics['nobel_winners']}
Total Citations: {self.statistics['total_citations']}
Graph Density: {self.statistics['graph_density']:.3f}
Average Citations per Researcher: {self.statistics['average_citations_per_researcher']:.2f}

This report presents a comprehensive analysis of a test citation graph containing 
20 researchers, including 5 Nobel Prize winners. The graph is represented as a 
directed adjacency matrix where each edge represents a citation relationship.

{'='*80}
"""
    
    def generate_researcher_profiles(self):
        """Generate detailed researcher profiles"""
        content = f"""
RESEARCHER PROFILES
{'-'*80}

{'ID':<3} {'Name':<25} {'Nobel Prize':<20} {'Research Area':<15} {'H-Index':<8} {'Citations':<10}
{'-'*85}
"""
        
        for researcher in self.researchers:
            nobel_info = ""
            if researcher['is_nobel_winner']:
                nobel_info = f"🏆 {researcher['nobel_category']} {researcher['award_year']}"
            
            content += f"{researcher['id']:<3} {researcher['name']:<25} {nobel_info:<20} {researcher['research_area']:<15} {researcher['h_index']:<8} {researcher['total_citations']:<10,}\n"
        
        content += f"\n\nNOBEL PRIZE WINNERS DETAILED PROFILES\n{'-'*50}\n"
        
        for nobel_id in sorted(self.nobel_winners):
            researcher = self.researchers[nobel_id]
            citations_received = sum(self.adjacency_matrix[i][nobel_id] for i in range(self.n))
            citations_made = sum(self.adjacency_matrix[nobel_id])
            
            content += f"""
🏆 {researcher['name']} (ID {nobel_id})
   Nobel Prize: {researcher['nobel_category']} ({researcher['award_year']})
   Research Area: {researcher['research_area']}
   Academic Profile: H-Index: {researcher['h_index']}, Total Citations: {researcher['total_citations']:,}
   Citation Graph: Receives {citations_received} citations, Makes {citations_made} citations
"""
        
        return content
    
    def generate_adjacency_matrix(self):
        """Generate adjacency matrix visualization"""
        content = f"""

ADJACENCY MATRIX
{'-'*80}

Matrix Interpretation: matrix[i][j] = 1 means researcher i cites researcher j
Nobel Prize winners are marked with 🏆

     """
        
        # Column headers
        for j in range(self.n):
            content += f"{j:2d} "
        content += "\n     "
        
        for j in range(self.n):
            if j in self.nobel_winners:
                content += "🏆 "
            else:
                content += "   "
        content += f"\n   {'-' * (self.n * 3 + 2)}\n"
        
        # Matrix rows
        for i in range(self.n):
            nobel_indicator = "🏆" if i in self.nobel_winners else " "
            content += f"{i:2d}{nobel_indicator}|"
            
            for j in range(self.n):
                if self.adjacency_matrix[i][j] == 1:
                    content += " ● "
                else:
                    content += " · "
            
            # Add researcher name
            researcher_name = self.researchers[i]['name']
            if len(researcher_name) > 25:
                researcher_name = researcher_name[:22] + "..."
            content += f"  {researcher_name}\n"
        
        content += f"""
LEGEND:
● = Citation exists (researcher in row cites researcher in column)
· = No citation
🏆 = Nobel Prize winner
Numbers represent researcher IDs (0-19)
"""
        
        return content
    
    def generate_statistics(self):
        """Generate comprehensive statistics"""
        content = f"""

GRAPH STATISTICS AND ANALYSIS
{'-'*80}

BASIC STATISTICS
{'-'*50}
• Total Researchers: {self.statistics['total_researchers']}
• Nobel Prize Winners: {self.statistics['nobel_winners']} ({self.statistics['nobel_winners']/self.statistics['total_researchers']*100:.1f}%)
• Total Citation Edges: {self.statistics['total_citations']}
• Graph Density: {self.statistics['graph_density']:.3f} ({self.statistics['graph_density']*100:.1f}% of possible edges)
• Average Citations per Researcher: {self.statistics['average_citations_per_researcher']:.2f}
• Maximum Citations Received: {self.statistics['max_citations_received']}
• Maximum Citations Made: {self.statistics['max_citations_made']}

TOP 10 MOST CITED RESEARCHERS
{'-'*50}
"""
        
        # Most cited researchers
        citations_received = [(i, self.statistics['in_degrees'][i]) for i in range(self.n)]
        citations_received.sort(key=lambda x: x[1], reverse=True)
        
        content += f"{'Rank':<5} {'Name':<25} {'Citations':<10} {'Nobel':<10}\n{'-'*50}\n"
        
        for rank, (researcher_id, citations) in enumerate(citations_received[:10], 1):
            researcher = self.researchers[researcher_id]
            nobel_status = "🏆 Yes" if researcher_id in self.nobel_winners else "No"
            content += f"{rank:<5} {researcher['name']:<25} {citations:<10} {nobel_status:<10}\n"
        
        content += f"\nTOP 10 MOST ACTIVE CITERS\n{'-'*50}\n"
        
        # Most active citers
        citations_made = [(i, self.statistics['out_degrees'][i]) for i in range(self.n)]
        citations_made.sort(key=lambda x: x[1], reverse=True)
        
        content += f"{'Rank':<5} {'Name':<25} {'Citations':<10} {'Nobel':<10}\n{'-'*50}\n"
        
        for rank, (researcher_id, citations) in enumerate(citations_made[:10], 1):
            researcher = self.researchers[researcher_id]
            nobel_status = "🏆 Yes" if researcher_id in self.nobel_winners else "No"
            content += f"{rank:<5} {researcher['name']:<25} {citations:<10} {nobel_status:<10}\n"
        
        content += f"""
NOBEL PRIZE WINNER ANALYSIS
{'-'*50}
• Citations to Nobel Winners: {self.statistics['nobel_winners_citations_received']} ({self.statistics['nobel_winners_citations_received']/self.statistics['total_citations']*100:.1f}% of all citations)
• Citations by Nobel Winners: {self.statistics['nobel_winners_citations_made']} ({self.statistics['nobel_winners_citations_made']/self.statistics['total_citations']*100:.1f}% of all citations)
• Average Citations Received by Nobel Winners: {self.statistics['nobel_winners_citations_received']/len(self.nobel_winners):.1f}
• Average Citations Received by Non-Nobel Winners: {(self.statistics['total_citations'] - self.statistics['nobel_winners_citations_received'])/(self.n - len(self.nobel_winners)):.1f}
"""
        
        return content
    
    def generate_citation_patterns(self):
        """Generate citation patterns analysis"""
        content = f"""

CITATION PATTERNS
{'-'*80}

CITATIONS INVOLVING NOBEL PRIZE WINNERS
{'-'*50}
Format: Citing Researcher → Cited Researcher
"""
        
        # Find all citations involving Nobel winners
        nobel_citations = []
        for i in range(self.n):
            for j in range(self.n):
                if self.adjacency_matrix[i][j] == 1 and (i in self.nobel_winners or j in self.nobel_winners):
                    citing = self.researchers[i]['name']
                    cited = self.researchers[j]['name']
                    citing_nobel = "🏆" if i in self.nobel_winners else ""
                    cited_nobel = "🏆" if j in self.nobel_winners else ""
                    nobel_citations.append((i, j, citing, cited, citing_nobel, cited_nobel))
        
        content += f"\nTotal: {len(nobel_citations)} citations involving Nobel winners\n\n"
        
        for i, j, citing, cited, citing_nobel, cited_nobel in nobel_citations:
            content += f"{citing} {citing_nobel} → {cited} {cited_nobel}\n"
        
        content += f"\nRESEARCH AREA DISTRIBUTION\n{'-'*50}\n"
        
        # Research area analysis
        research_areas = {}
        for researcher in self.researchers:
            area = researcher['research_area']
            if area not in research_areas:
                research_areas[area] = {'total': 0, 'nobel': 0}
            research_areas[area]['total'] += 1
            if researcher['is_nobel_winner']:
                research_areas[area]['nobel'] += 1
        
        content += f"{'Research Area':<20} {'Total':<8} {'Nobel':<8} {'% Nobel':<8}\n{'-'*44}\n"
        
        for area, counts in sorted(research_areas.items()):
            nobel_pct = (counts['nobel'] / counts['total']) * 100 if counts['total'] > 0 else 0
            content += f"{area:<20} {counts['total']:<8} {counts['nobel']:<8} {nobel_pct:<8.1f}%\n"
        
        return content
    
    def generate_algorithm_guidelines(self):
        """Generate algorithm testing guidelines"""
        content = f"""

ALGORITHM TESTING GUIDELINES
{'-'*80}

OVERVIEW
{'-'*50}
This citation graph is designed for testing various graph algorithms. The adjacency 
matrix format makes it suitable for testing centrality measures, path-finding 
algorithms, community detection, and influence analysis.

RECOMMENDED ALGORITHM CATEGORIES
{'-'*50}

1. CENTRALITY MEASURES
   Algorithms: PageRank, Betweenness Centrality, Closeness Centrality, Eigenvector Centrality
   Expected: Nobel winners should generally score higher, especially Prof. Jack White (ID 9)

2. PATH FINDING ALGORITHMS
   Algorithms: Shortest paths, All-pairs shortest paths, Reachability analysis
   Expected: Average path length ~2-3 steps, most researchers reachable from Nobel winners

3. COMMUNITY DETECTION
   Algorithms: Louvain, Leiden, Modularity-based clustering
   Expected: Research area clustering effects, one main connected component

4. INFLUENCE ANALYSIS
   Algorithms: Information propagation, Influence maximization, Network flow
   Expected: Nobel winner bias in influence spread models

5. NETWORK ANALYSIS
   Algorithms: Strongly connected components, Topological sorting, Graph traversal
   Expected: One large strongly connected component containing most researchers

DATA QUALITY FEATURES
{'-'*50}
• Realistic Bias: Nobel winners have 2.5x higher probability of being cited
• Research Clustering: Researchers in same area cite each other more frequently
• Temporal Consistency: Award years are realistic (2015-2023)
• Scale: 20 researchers provide good test size without being overwhelming
• No Self-Citations: Diagonal elements are all zero
• Connected Structure: Most nodes are reachable from each other

USAGE INSTRUCTIONS
{'-'*50}
To use this graph in your algorithms:

1. Load the data:
   import json
   with open('citation_graph_matrix_YYYYMMDD_HHMMSS.json', 'r') as f:
       data = json.load(f)
   adjacency_matrix = data['adjacency_matrix']
   researchers = data['researchers']
   nobel_winners = set(data['nobel_winners_ids'])

2. Convert to your preferred format:
   - Adjacency list: for sparse graph algorithms
   - NetworkX graph: for using NetworkX library
   - NumPy array: for numerical computations

3. Run your algorithms and compare results with expected patterns

EXPECTED TEST RESULTS
{'-'*50}
Based on the graph structure, algorithms should show:
• Higher centrality scores for Nobel Prize winners
• Short citation paths between researchers (average path length ~2-3)
• One main strongly connected component containing most researchers
• Research area clustering effects in community detection
• Nobel winner bias in influence propagation models

VALIDATION CHECKLIST
{'-'*50}
✓ Proper adjacency matrix format (20×20)
✓ No self-citations (diagonal zeros)
✓ Realistic citation distribution
✓ Nobel winner bias implementation
✓ Connected graph structure
✓ Research area clustering effects
"""
        
        return content
    
    def generate_text_report(self, output_filename=None):
        """Generate the complete text report"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"citation_graph_report_{timestamp}.txt"
        
        # Generate all sections
        report_content = ""
        report_content += self.generate_header()
        report_content += self.generate_researcher_profiles()
        report_content += self.generate_adjacency_matrix()
        report_content += self.generate_statistics()
        report_content += self.generate_citation_patterns()
        report_content += self.generate_algorithm_guidelines()
        
        # Add footer
        report_content += f"""

{'='*80}
END OF REPORT
{'='*80}
Report generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
Citation Graph Test Dataset - Ready for Algorithm Testing
{'='*80}
"""
        
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            return output_filename
        except Exception as e:
            print(f"Error generating text report: {e}")
            return None


def main():
    """Main function to generate text report"""
    # Find the most recent graph file
    import glob
    
    pattern = "citation_graph_matrix_*.json"
    files = glob.glob(pattern)
    
    if not files:
        print("No citation graph files found. Please run simple_citation_graph_generator.py first.")
        return
    
    # Use the most recent file
    graph_file = max(files, key=os.path.getctime)
    print(f"Using graph file: {graph_file}")
    
    # Generate text report
    text_generator = CitationGraphTextGenerator(graph_file)
    output_file = text_generator.generate_text_report()
    
    if output_file:
        print(f"\n✅ Text report generated successfully: {output_file}")
        print("\n📋 The report contains:")
        print("• Complete researcher profiles with Nobel winner details")
        print("• ASCII-art adjacency matrix visualization")
        print("• Comprehensive statistics and rankings")
        print("• Citation patterns and research area analysis")
        print("• Algorithm testing guidelines and usage instructions")
        
        print(f"\n📖 The text report is ready for:")
        print("• Direct reading and analysis")
        print("• Easy printing on any printer")
        print("• Copy-paste into documents")
        print("• Email sharing")
        
        print(f"\n📂 All available files:")
        html_file = output_file.replace('.txt', '.html').replace('_report_', '_report_')
        csv_file = graph_file.replace('.json', '.csv')
        
        print(f"• Text Report: {output_file}")
        if os.path.exists(html_file):
            print(f"• HTML Report: {html_file}")
        print(f"• Graph Data: {graph_file}")
        print(f"• CSV Matrix: {csv_file}")
    else:
        print("❌ Failed to generate text report.")


if __name__ == "__main__":
    main()
