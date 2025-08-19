"""
Citation Graph Generator for Algorithm Testing

This script creates a directed graph representing citation relationships between 20 researchers,
where some are Nobel Prize winners. The graph is represented as an adjacency matrix.

If researcher A cites researcher B, there will be a directed edge from A to B.
Matrix[i][j] = 1 means researcher i cites researcher j.
"""

import numpy as np
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx

class CitationGraphGenerator:
    def __init__(self, num_researchers=100, num_nobel_winners=5):
        self.num_researchers = num_researchers
        self.num_nobel_winners = num_nobel_winners
        self.researchers = []
        self.nobel_winners = set()
        self.adjacency_matrix = None
        self.citation_count = {}
        
    def generate_researchers(self):
        """Generate list of researchers with some being Nobel Prize winners"""
        # Generate researcher names dynamically
        researcher_names = [f"Researcher {i}" for i in range(self.num_researchers)]

        # Nobel Prize categories for winners
        nobel_categories = [
            "Physics", "Chemistry", "Medicine", "Economic Sciences", "Literature"
        ]
        
        # Award years for Nobel winners
        award_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        
        # Generate researcher data
        for i in range(self.num_researchers):
            researcher = {
                "id": i,
                "name": researcher_names[i],
                "is_nobel_winner": False,
                "nobel_category": None,
                "award_year": None,
                "research_area": self._generate_research_area(),
                "h_index": random.randint(10, 100),
                "total_citations": random.randint(500, 5000)
            }
            self.researchers.append(researcher)
            self.citation_count[i] = 0
        
        # Randomly select Nobel Prize winners
        nobel_indices = random.sample(range(self.num_researchers), self.num_nobel_winners)
        for idx in nobel_indices:
            self.researchers[idx]["is_nobel_winner"] = True
            self.researchers[idx]["nobel_category"] = random.choice(nobel_categories)
            self.researchers[idx]["award_year"] = random.choice(award_years)
            self.researchers[idx]["h_index"] = random.randint(50, 150)  # Higher h-index for Nobel winners
            self.researchers[idx]["total_citations"] = random.randint(2000, 10000)  # More citations
            self.nobel_winners.add(idx)
    
    def _generate_research_area(self):
        """Generate research area for researchers"""
        areas = [
            "Machine Learning", "Quantum Physics", "Biochemistry", "Neuroscience",
            "Astrophysics", "Materials Science", "Computer Vision", "Genetics",
            "Robotics", "Climate Science", "Nanotechnology", "Bioinformatics"
        ]
        return random.choice(areas)
    
    def generate_citation_matrix(self, citation_probability=0.15, nobel_bias=2.0):
        """
        Generate adjacency matrix representing citation relationships
        
        Args:
            citation_probability: Base probability of citation between any two researchers
            nobel_bias: Multiplier for probability of citing Nobel Prize winners
        """
        self.adjacency_matrix = np.zeros((self.num_researchers, self.num_researchers), dtype=int)
        
        for i in range(self.num_researchers):
            for j in range(self.num_researchers):
                if i != j:  # A researcher cannot cite themselves
                    # Base citation probability
                    prob = citation_probability
                    
                    # Increase probability if citing a Nobel Prize winner
                    if j in self.nobel_winners:
                        prob *= nobel_bias
                    
                    # Increase probability based on similar research areas
                    if self.researchers[i]["research_area"] == self.researchers[j]["research_area"]:
                        prob *= 1.5
                    
                    # Generate citation
                    if random.random() < prob:
                        self.adjacency_matrix[i][j] = 1
                        self.citation_count[j] += 1
    
    def get_graph_statistics(self):
        """Calculate and return graph statistics"""
        total_edges = np.sum(self.adjacency_matrix)
        in_degrees = np.sum(self.adjacency_matrix, axis=0)  # Citations received
        out_degrees = np.sum(self.adjacency_matrix, axis=1)  # Citations made
        
        # Nobel winners' statistics
        nobel_citations_received = sum(in_degrees[i] for i in self.nobel_winners)
        nobel_citations_made = sum(out_degrees[i] for i in self.nobel_winners)
        
        stats = {
            "total_researchers": self.num_researchers,
            "nobel_winners": len(self.nobel_winners),
            "total_citations": int(total_edges),
            "average_citations_per_researcher": float(total_edges / self.num_researchers),
            "nobel_winners_citations_received": int(nobel_citations_received),
            "nobel_winners_citations_made": int(nobel_citations_made),
            "graph_density": float(total_edges / (self.num_researchers * (self.num_researchers - 1))),
            "max_citations_received": int(np.max(in_degrees)),
            "max_citations_made": int(np.max(out_degrees))
        }
        
        return stats
    
    def visualize_graph(self, save_path=None):
        """Create a visualization of the citation graph"""
        G = nx.DiGraph()
        
        # Add nodes
        for i, researcher in enumerate(self.researchers):
            G.add_node(i, name=researcher["name"], is_nobel=researcher["is_nobel_winner"])
        
        # Add edges
        for i in range(self.num_researchers):
            for j in range(self.num_researchers):
                if self.adjacency_matrix[i][j] == 1:
                    G.add_edge(i, j)
        
        # Create layout
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        nobel_nodes = [i for i in self.nobel_winners]
        regular_nodes = [i for i in range(self.num_researchers) if i not in self.nobel_winners]
        
        # Draw regular researchers
        nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, 
                              node_color='lightblue', node_size=500, alpha=0.7)
        
        # Draw Nobel Prize winners
        nx.draw_networkx_nodes(G, pos, nodelist=nobel_nodes, 
                              node_color='gold', node_size=800, alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=20, alpha=0.6, arrowstyle='->')
        
        # Add labels
        labels = {i: f"{i}" for i in range(self.num_researchers)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Citation Network Graph\n(Gold nodes = Nobel Prize Winners)", fontsize=16)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_data(self, filename_prefix="citation_graph"):
        """Save graph data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save adjacency matrix
        matrix_filename = f"{filename_prefix}_matrix_{timestamp}.json"
        matrix_data = {
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "researchers": self.researchers,
            "statistics": self.get_graph_statistics(),
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "description": "Directed citation graph with adjacency matrix representation",
                "matrix_interpretation": "matrix[i][j] = 1 means researcher i cites researcher j"
            }
        }
        
        with open(matrix_filename, 'w', encoding='utf-8') as f:
            json.dump(matrix_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy viewing
        csv_filename = f"{filename_prefix}_matrix_{timestamp}.csv"
        np.savetxt(csv_filename, self.adjacency_matrix, delimiter=',', fmt='%d')
        
        # Save researcher details
        researchers_filename = f"{filename_prefix}_researchers_{timestamp}.json"
        with open(researchers_filename, 'w', encoding='utf-8') as f:
            json.dump(self.researchers, f, indent=2, ensure_ascii=False)
        
        return {
            "matrix_file": matrix_filename,
            "csv_file": csv_filename,
            "researchers_file": researchers_filename
        }
    
    def print_summary(self):
        """Print a summary of the generated graph"""
        stats = self.get_graph_statistics()
        
        print("=== Citation Graph Summary ===")
        print(f"Total Researchers: {stats['total_researchers']}")
        print(f"Nobel Prize Winners: {stats['nobel_winners']}")
        print(f"Total Citation Edges: {stats['total_citations']}")
        print(f"Graph Density: {stats['graph_density']:.3f}")
        print(f"Average Citations per Researcher: {stats['average_citations_per_researcher']:.2f}")
        
        print("\n=== Nobel Prize Winners ===")
        for i in self.nobel_winners:
            researcher = self.researchers[i]
            citations_received = np.sum(self.adjacency_matrix[:, i])
            citations_made = np.sum(self.adjacency_matrix[i, :])
            print(f"ID {i}: {researcher['name']} ({researcher['nobel_category']} {researcher['award_year']}) - "
                  f"Received: {citations_received}, Made: {citations_made}")
        
        print("\n=== Top 5 Most Cited Researchers ===")
        citations_received = [(i, np.sum(self.adjacency_matrix[:, i])) for i in range(self.num_researchers)]
        citations_received.sort(key=lambda x: x[1], reverse=True)
        
        for i, (researcher_id, citations) in enumerate(citations_received[:5]):
            researcher = self.researchers[researcher_id]
            nobel_status = "🏆" if researcher["is_nobel_winner"] else ""
            print(f"{i+1}. ID {researcher_id}: {researcher['name']} {nobel_status} - {citations} citations")


def main():
    """Main function to generate and display the citation graph"""
    print("Generating Citation Graph for Algorithm Testing...")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create graph generator
    generator = CitationGraphGenerator(num_researchers=20, num_nobel_winners=5)
    
    # Generate researchers and citation matrix
    generator.generate_researchers()
    generator.generate_citation_matrix(citation_probability=0.12, nobel_bias=2.5)
    
    # Print summary
    generator.print_summary()
    
    # Save data
    print("\nSaving graph data...")
    files = generator.save_data()
    print(f"Files saved:")
    for file_type, filename in files.items():
        print(f"  {file_type}: {filename}")
    
    # Display adjacency matrix
    print("\n=== Adjacency Matrix (20x20) ===")
    print("(Rows = citing researcher, Columns = cited researcher)")
    print("Matrix[i][j] = 1 means researcher i cites researcher j\n")
    
    # Print matrix with row/column headers
    print("   ", end="")
    for j in range(generator.num_researchers):
        print(f"{j:2d}", end=" ")
    print()
    
    for i in range(generator.num_researchers):
        print(f"{i:2d}:", end=" ")
        for j in range(generator.num_researchers):
            print(f"{generator.adjacency_matrix[i][j]:2d}", end=" ")
        print()
    
    # Visualize graph
    print("\nGenerating visualization...")
    try:
        generator.visualize_graph("citation_graph_visualization.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")
        print("Install matplotlib and networkx for visualization: pip install matplotlib networkx")
    
    return generator


if __name__ == "__main__":
    # Generate the test graph
    graph_generator = main()
    
    print("\n=== Graph Ready for Algorithm Testing ===")
    print("The adjacency matrix represents the citation network where:")
    print("- Each researcher is assigned an ID (0-19)")
    print("- Nobel Prize winners are marked with 🏆")
    print("- Matrix[i][j] = 1 indicates researcher i cites researcher j")
    print("- Use this graph to test your selected algorithm!")
