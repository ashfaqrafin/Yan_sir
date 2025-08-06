"""
Simple Citation Graph Generator for Algorithm Testing

This script creates a directed graph representing citation relationships between 20 researchers,
where some are Nobel Prize winners. The graph is represented as an adjacency matrix.

If researcher A cites researcher B, there will be a directed edge from A to B.
Matrix[i][j] = 1 means researcher i cites researcher j.
"""

import json
import random
from datetime import datetime

class SimpleCitationGraphGenerator:
    def __init__(self, num_researchers=20, num_nobel_winners=5):
        self.num_researchers = num_researchers
        self.num_nobel_winners = num_nobel_winners
        self.researchers = []
        self.nobel_winners = set()
        self.adjacency_matrix = []
        self.citation_count = {}
        
    def generate_researchers(self):
        """Generate list of researchers with some being Nobel Prize winners"""
        # Sample researcher names (mix of real and fictional)
        researcher_names = [
            "Dr. Alice Johnson", "Prof. Bob Smith", "Dr. Carol Davis", "Prof. David Wilson",
            "Dr. Eva Brown", "Prof. Frank Miller", "Dr. Grace Taylor", "Prof. Henry Moore",
            "Dr. Iris Clark", "Prof. Jack White", "Dr. Kate Green", "Prof. Liam Hall",
            "Dr. Maya Patel", "Prof. Noah Kim", "Dr. Olivia Lee", "Prof. Paul Zhang",
            "Dr. Quinn Rodriguez", "Prof. Rachel Chen", "Dr. Sam Williams", "Prof. Tina Liu"
        ]
        
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
        # Initialize matrix with zeros
        self.adjacency_matrix = [[0 for _ in range(self.num_researchers)] 
                                for _ in range(self.num_researchers)]
        
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
        total_edges = sum(sum(row) for row in self.adjacency_matrix)
        in_degrees = [sum(self.adjacency_matrix[i][j] for i in range(self.num_researchers)) 
                      for j in range(self.num_researchers)]  # Citations received
        out_degrees = [sum(row) for row in self.adjacency_matrix]  # Citations made
        
        # Nobel winners' statistics
        nobel_citations_received = sum(in_degrees[i] for i in self.nobel_winners)
        nobel_citations_made = sum(out_degrees[i] for i in self.nobel_winners)
        
        stats = {
            "total_researchers": self.num_researchers,
            "nobel_winners": len(self.nobel_winners),
            "total_citations": total_edges,
            "average_citations_per_researcher": total_edges / self.num_researchers,
            "nobel_winners_citations_received": nobel_citations_received,
            "nobel_winners_citations_made": nobel_citations_made,
            "graph_density": total_edges / (self.num_researchers * (self.num_researchers - 1)),
            "max_citations_received": max(in_degrees),
            "max_citations_made": max(out_degrees),
            "in_degrees": in_degrees,
            "out_degrees": out_degrees
        }
        
        return stats
    
    def save_data(self, filename_prefix="citation_graph"):
        """Save graph data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save adjacency matrix and all data
        matrix_filename = f"{filename_prefix}_matrix_{timestamp}.json"
        matrix_data = {
            "adjacency_matrix": self.adjacency_matrix,
            "researchers": self.researchers,
            "nobel_winners_ids": list(self.nobel_winners),
            "statistics": self.get_graph_statistics(),
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "description": "Directed citation graph with adjacency matrix representation",
                "matrix_interpretation": "matrix[i][j] = 1 means researcher i cites researcher j",
                "num_researchers": self.num_researchers,
                "num_nobel_winners": self.num_nobel_winners
            }
        }
        
        with open(matrix_filename, 'w', encoding='utf-8') as f:
            json.dump(matrix_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy viewing
        csv_filename = f"{filename_prefix}_matrix_{timestamp}.csv"
        with open(csv_filename, 'w') as f:
            # Write header
            f.write("," + ",".join([f"R{i}" for i in range(self.num_researchers)]) + "\n")
            # Write matrix rows
            for i, row in enumerate(self.adjacency_matrix):
                f.write(f"R{i}," + ",".join(map(str, row)) + "\n")
        
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
            citations_received = sum(self.adjacency_matrix[j][i] for j in range(self.num_researchers))
            citations_made = sum(self.adjacency_matrix[i])
            print(f"ID {i}: {researcher['name']} ({researcher['nobel_category']} {researcher['award_year']}) - "
                  f"Received: {citations_received}, Made: {citations_made}")
        
        print("\n=== Top 5 Most Cited Researchers ===")
        citations_received = [(i, sum(self.adjacency_matrix[j][i] for j in range(self.num_researchers))) 
                             for i in range(self.num_researchers)]
        citations_received.sort(key=lambda x: x[1], reverse=True)
        
        for i, (researcher_id, citations) in enumerate(citations_received[:5]):
            researcher = self.researchers[researcher_id]
            nobel_status = "🏆" if researcher["is_nobel_winner"] else ""
            print(f"{i+1}. ID {researcher_id}: {researcher['name']} {nobel_status} - {citations} citations")
    
    def print_adjacency_matrix(self):
        """Print the adjacency matrix in a readable format"""
        print("\n=== Adjacency Matrix (20x20) ===")
        print("(Rows = citing researcher, Columns = cited researcher)")
        print("Matrix[i][j] = 1 means researcher i cites researcher j\n")
        
        # Print column headers
        print("   ", end="")
        for j in range(self.num_researchers):
            print(f"{j:2d}", end=" ")
        print()
        
        # Print matrix with row headers
        for i in range(self.num_researchers):
            print(f"{i:2d}:", end=" ")
            for j in range(self.num_researchers):
                print(f"{self.adjacency_matrix[i][j]:2d}", end=" ")
            print()


def main():
    """Main function to generate and display the citation graph"""
    print("Generating Citation Graph for Algorithm Testing...")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create graph generator
    generator = SimpleCitationGraphGenerator(num_researchers=20, num_nobel_winners=5)
    
    # Generate researchers and citation matrix
    print("Step 1: Generating researchers...")
    generator.generate_researchers()
    
    print("Step 2: Generating citation relationships...")
    generator.generate_citation_matrix(citation_probability=0.12, nobel_bias=2.5)
    
    # Print summary
    generator.print_summary()
    
    # Display adjacency matrix
    generator.print_adjacency_matrix()
    
    # Save data
    print("\nStep 3: Saving graph data...")
    files = generator.save_data()
    print(f"Files saved:")
    for file_type, filename in files.items():
        print(f"  {file_type}: {filename}")
    
    print("\n" + "=" * 60)
    print("=== Graph Ready for Algorithm Testing ===")
    print("The adjacency matrix represents the citation network where:")
    print("- Each researcher is assigned an ID (0-19)")
    print("- Nobel Prize winners are marked with 🏆")
    print("- Matrix[i][j] = 1 indicates researcher i cites researcher j")
    print("- Use this graph to test your selected algorithm!")
    print("=" * 60)
    
    return generator


if __name__ == "__main__":
    # Generate the test graph
    graph_generator = main()
