"""
Sparse Matrix Implementation for Large-Scale Researcher Ranking

This implementation is specifically designed for handling citation graphs
with millions of researchers using memory-efficient sparse matrix operations.

Key Features:
- Compressed Sparse Row (CSR) format for efficient row operations
- Memory-efficient algorithms for ranking computation
- Scalable to 5+ million researchers
- Batch processing for memory management
- Incremental ranking updates
"""

import json
import time
import math
import gc
from typing import Dict, List, Tuple, Iterator, Optional
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class SparseCitationGraph:
    """
    Compressed Sparse Row (CSR) representation of citation graph
    Optimized for large-scale researcher ranking algorithms
    """
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        # CSR format components
        self.row_ptr = [0] * (num_nodes + 1)  # Row pointers
        self.col_indices = []                 # Column indices  
        self.values = []                      # Edge weights (1.0 for citations)
        self.finalized = False
        
        # Temporary storage during construction
        self._temp_edges = defaultdict(dict)
        
    def add_edge(self, from_node: int, to_node: int, weight: float = 1.0):
        """Add edge to the graph during construction phase"""
        if self.finalized:
            raise ValueError("Graph is already finalized")
        if 0 <= from_node < self.num_nodes and 0 <= to_node < self.num_nodes:
            self._temp_edges[from_node][to_node] = weight
    
    def finalize(self):
        """Convert temporary storage to CSR format"""
        if self.finalized:
            return
            
        print("Finalizing sparse matrix (converting to CSR format)...")
        start_time = time.time()
        
        self.col_indices.clear()
        self.values.clear()
        
        current_index = 0
        for row in range(self.num_nodes):
            self.row_ptr[row] = current_index
            
            if row in self._temp_edges:
                # Sort by column for better cache locality
                for col in sorted(self._temp_edges[row].keys()):
                    self.col_indices.append(col)
                    self.values.append(self._temp_edges[row][col])
                    current_index += 1
        
        self.row_ptr[self.num_nodes] = current_index
        
        # Clear temporary storage to free memory
        self._temp_edges.clear()
        gc.collect()
        
        self.finalized = True
        
        finalize_time = time.time() - start_time
        print(f"Matrix finalization completed in {finalize_time:.2f} seconds")
        print(f"Sparse matrix: {self.num_nodes} nodes, {len(self.col_indices)} edges")
        print(f"Memory usage: ~{self.memory_usage_mb():.1f} MB")
    
    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB"""
        if not self.finalized:
            # Rough estimate for temporary storage
            temp_edges = sum(len(row) for row in self._temp_edges.values())
            return (temp_edges * 16 + self.num_nodes * 8) / (1024 * 1024)
        
        # CSR format memory usage
        row_ptr_bytes = len(self.row_ptr) * 8  # 8 bytes per int64
        col_indices_bytes = len(self.col_indices) * 8
        values_bytes = len(self.values) * 8  # 8 bytes per float64
        
        return (row_ptr_bytes + col_indices_bytes + values_bytes) / (1024 * 1024)
    
    def get_out_degree(self, node: int) -> int:
        """Get out-degree of a node (number of citations made)"""
        if not self.finalized:
            return len(self._temp_edges.get(node, {}))
        
        return self.row_ptr[node + 1] - self.row_ptr[node]
    
    def get_out_neighbors(self, node: int) -> Iterator[Tuple[int, float]]:
        """Get outgoing neighbors of a node (who this node cites)"""
        if not self.finalized:
            if node in self._temp_edges:
                for neighbor, weight in self._temp_edges[node].items():
                    yield neighbor, weight
            return
        
        start_idx = self.row_ptr[node]
        end_idx = self.row_ptr[node + 1]
        
        for i in range(start_idx, end_idx):
            yield self.col_indices[i], self.values[i]
    
    def get_in_neighbors(self, node: int) -> List[Tuple[int, float]]:
        """Get incoming neighbors of a node (who cites this node) - less efficient"""
        if not self.finalized:
            self.finalize()
        
        in_neighbors = []
        
        # Scan all rows to find citations to this node
        for row in range(self.num_nodes):
            start_idx = self.row_ptr[row]
            end_idx = self.row_ptr[row + 1]
            
            for i in range(start_idx, end_idx):
                if self.col_indices[i] == node:
                    in_neighbors.append((row, self.values[i]))
                    break  # Found the edge, move to next row
        
        return in_neighbors
    
    def has_edge(self, from_node: int, to_node: int) -> bool:
        """Check if edge exists between two nodes"""
        for neighbor, _ in self.get_out_neighbors(from_node):
            if neighbor == to_node:
                return True
        return False
    
    def get_edge_weight(self, from_node: int, to_node: int) -> float:
        """Get weight of edge between two nodes"""
        for neighbor, weight in self.get_out_neighbors(from_node):
            if neighbor == to_node:
                return weight
        return 0.0

class LargeScaleRanking:
    """
    Memory-efficient implementation of researcher ranking algorithms
    for datasets with millions of researchers
    """
    
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, batch_size: int = 10000):
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size  # For batch processing large datasets
        
        self.researchers = {}
        self.citation_graph: Optional[SparseCitationGraph] = None
        self.num_researchers = 0
        
        # Performance tracking
        self.timings = {}
        
    def load_from_edge_list(self, edges: List[Tuple[int, int]], 
                           researchers: Dict[int, Dict], 
                           num_researchers: int):
        """Load citation graph from edge list format"""
        print(f"Loading citation graph with {num_researchers} researchers and {len(edges)} edges...")
        start_time = time.time()
        
        self.researchers = researchers
        self.num_researchers = num_researchers
        
        # Create sparse graph
        self.citation_graph = SparseCitationGraph(num_researchers)
        
        # Add edges in batches to manage memory
        batch_count = 0
        for i, (from_node, to_node) in enumerate(edges):
            self.citation_graph.add_edge(from_node, to_node)
            
            if (i + 1) % 100000 == 0:
                batch_count += 1
                print(f"  Processed {i + 1:,} edges ({batch_count * 100000:,} total)")
                
                # Periodic garbage collection for large datasets
                if batch_count % 10 == 0:
                    gc.collect()
        
        # Finalize the sparse matrix
        self.citation_graph.finalize()
        
        load_time = time.time() - start_time
        self.timings['data_loading'] = load_time
        print(f"Data loading completed in {load_time:.2f} seconds")
    
    def load_from_json(self, json_file: str):
        """Load data from JSON file (same format as original)"""
        print(f"Loading data from {json_file}...")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract researchers
        researchers = {}
        for researcher_data in data.get('researchers', []):
            researchers[researcher_data['id']] = researcher_data
        
        # Extract edges
        edges = []
        if 'sparse_adjacency_matrix' in data:
            edges = data['sparse_adjacency_matrix']['edges']
        elif 'adjacency_matrix_dense' in data:
            # Convert dense to edge list
            dense_matrix = data['adjacency_matrix_dense']
            for i, row in enumerate(dense_matrix):
                for j, value in enumerate(row):
                    if value > 0:
                        edges.append((i, j))
        
        self.load_from_edge_list(edges, researchers, len(researchers))
    
    def compute_basic_ranking_sparse(self) -> Dict[int, float]:
        """
        Compute basic ranking for large sparse graphs
        Uses memory-efficient algorithms
        """
        print("Computing basic ranking for large sparse graph...")
        start_time = time.time()
        
        if not self.citation_graph.finalized:
            self.citation_graph.finalize()
        
        rankings = {}
        
        # Precompute out-degrees for efficiency
        print("  Precomputing out-degrees...")
        out_degrees = {}
        for node in range(self.num_researchers):
            out_degrees[node] = self.citation_graph.get_out_degree(node)
        
        # Process researchers in batches to manage memory
        print(f"  Processing {self.num_researchers} researchers in batches of {self.batch_size}...")
        
        for batch_start in range(0, self.num_researchers, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_researchers)
            
            # Process batch
            for i in range(batch_start, batch_end):
                rank_score = 0.0
                
                # Get all researchers who cite researcher i
                in_neighbors = self.citation_graph.get_in_neighbors(i)
                
                for citing_node, edge_weight in in_neighbors:
                    if citing_node < self.num_researchers:  # Only count actual researchers
                        # Researcher weight (using h_index as significance measure)
                        researcher = self.researchers.get(citing_node, {})
                        w_j = max(researcher.get('h_index', 1), 1)
                        
                        # Total citations made by citing researcher
                        total_citations = out_degrees[citing_node]
                        
                        if total_citations > 0:
                            rank_score += w_j * (edge_weight / total_citations)
                
                rankings[i] = rank_score
            
            # Progress update and memory management
            if (batch_start // self.batch_size + 1) % 10 == 0:
                processed = min(batch_end, self.num_researchers)
                print(f"    Processed {processed:,}/{self.num_researchers:,} researchers")
                gc.collect()  # Periodic garbage collection
        
        computation_time = time.time() - start_time
        self.timings['basic_ranking'] = computation_time
        print(f"Basic ranking completed in {computation_time:.2f} seconds")
        
        return rankings
    
    def compute_pagerank_style_ranking(self, damping: float = 0.85, 
                                      max_iterations: int = 50, 
                                      tolerance: float = 1e-6) -> Dict[int, float]:
        """
        Compute PageRank-style ranking for researcher networks
        More efficient for very large graphs
        """
        print(f"Computing PageRank-style ranking (damping={damping}, max_iter={max_iterations})...")
        start_time = time.time()
        
        if not self.citation_graph.finalized:
            self.citation_graph.finalize()
        
        # Initialize rankings
        initial_value = 1.0 / self.num_researchers
        rankings = {i: initial_value for i in range(self.num_researchers)}
        new_rankings = rankings.copy()
        
        # Precompute out-degrees and researcher weights
        out_degrees = {}
        researcher_weights = {}
        
        for node in range(self.num_researchers):
            out_degrees[node] = max(self.citation_graph.get_out_degree(node), 1)  # Avoid division by zero
            researcher = self.researchers.get(node, {})
            researcher_weights[node] = max(researcher.get('h_index', 1), 1)
        
        # Iterative computation
        for iteration in range(max_iterations):
            # Reset new rankings
            for i in range(self.num_researchers):
                new_rankings[i] = (1 - damping) / self.num_researchers
            
            # Compute ranking updates
            total_change = 0.0
            
            for i in range(self.num_researchers):
                # Get all researchers who cite researcher i
                in_neighbors = self.citation_graph.get_in_neighbors(i)
                
                rank_contribution = 0.0
                for citing_node, edge_weight in in_neighbors:
                    if citing_node < self.num_researchers:
                        # PageRank-style update with researcher weights
                        contribution = (rankings[citing_node] * researcher_weights[citing_node] * 
                                      edge_weight / out_degrees[citing_node])
                        rank_contribution += contribution
                
                new_rankings[i] += damping * rank_contribution
                total_change += abs(new_rankings[i] - rankings[i])
            
            # Update rankings
            rankings, new_rankings = new_rankings, rankings
            
            # Check convergence
            avg_change = total_change / self.num_researchers
            if avg_change < tolerance:
                print(f"  Converged after {iteration + 1} iterations (avg_change: {avg_change:.2e})")
                break
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: avg_change = {avg_change:.2e}")
        
        computation_time = time.time() - start_time
        self.timings['pagerank_ranking'] = computation_time
        print(f"PageRank-style ranking completed in {computation_time:.2f} seconds")
        
        return rankings
    
    def compute_collaboration_ranking_sparse(self, collaboration_strength: float = 0.2) -> Dict[int, float]:
        """
        Compute collaboration-aware ranking for large sparse graphs
        Uses approximation methods for efficiency
        """
        print("Computing collaboration ranking for large sparse graph...")
        start_time = time.time()
        
        # Use research area similarity for collaboration estimation
        area_nodes = defaultdict(list)
        for node_id, researcher in self.researchers.items():
            area = researcher.get('research_area', 'Unknown')
            area_nodes[area].append(node_id)
        
        rankings = {}
        
        # Process in batches
        for batch_start in range(0, self.num_researchers, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_researchers)
            
            for i in range(batch_start, batch_end):
                rank_score = 0.0
                researcher_i = self.researchers.get(i, {})
                area_i = researcher_i.get('research_area', 'Unknown')
                
                # Citation component (α weight)
                citation_score = 0.0
                in_neighbors = self.citation_graph.get_in_neighbors(i)
                
                for citing_node, edge_weight in in_neighbors:
                    if citing_node < self.num_researchers:
                        researcher_j = self.researchers.get(citing_node, {})
                        w_j = max(researcher_j.get('h_index', 1), 1)
                        out_degree = max(self.citation_graph.get_out_degree(citing_node), 1)
                        
                        citation_score += w_j * (edge_weight / out_degree)
                
                # Collaboration component (β weight) - approximated
                collaboration_score = 0.0
                
                # Estimate collaboration based on research area and mutual citations
                for node_j in area_nodes[area_i]:
                    if node_j != i:
                        researcher_j = self.researchers.get(node_j, {})
                        w_j = max(researcher_j.get('h_index', 1), 1)
                        
                        # Base collaboration within same area
                        base_collab = collaboration_strength
                        
                        # Boost if mutual citations exist
                        if (self.citation_graph.has_edge(i, node_j) or 
                            self.citation_graph.has_edge(node_j, i)):
                            base_collab *= 2.0
                        
                        collaboration_score += w_j * base_collab
                
                # Combined score
                rank_score = self.alpha * citation_score + self.beta * collaboration_score
                rankings[i] = rank_score
            
            # Progress update
            if (batch_start // self.batch_size + 1) % 10 == 0:
                processed = min(batch_end, self.num_researchers)
                print(f"  Processed {processed:,}/{self.num_researchers:,} researchers")
        
        computation_time = time.time() - start_time
        self.timings['collaboration_ranking'] = computation_time
        print(f"Collaboration ranking completed in {computation_time:.2f} seconds")
        
        return rankings
    
    def get_top_k(self, rankings: Dict[int, float], k: int = 10) -> List[Tuple[int, str, float]]:
        """Get top-k researchers from rankings"""
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for node_id, score in sorted_rankings[:k]:
            researcher = self.researchers.get(node_id, {})
            name = researcher.get('name', f'Researcher_{node_id}')
            result.append((node_id, name, score))
        
        return result
    
    def print_performance_stats(self):
        """Print performance statistics"""
        print(f"\nPerformance Statistics:")
        print("-" * 40)
        
        if self.citation_graph:
            print(f"Graph size: {self.citation_graph.num_nodes:,} nodes, {len(self.citation_graph.col_indices):,} edges")
            print(f"Memory usage: {self.citation_graph.memory_usage_mb():.1f} MB")
            density = len(self.citation_graph.col_indices) / (self.citation_graph.num_nodes ** 2)
            print(f"Graph density: {density:.2e}")
        
        total_time = sum(self.timings.values())
        print(f"Total computation time: {total_time:.2f} seconds")
        
        for operation, time_taken in self.timings.items():
            percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
            print(f"  {operation}: {time_taken:.2f}s ({percentage:.1f}%)")

def create_synthetic_large_dataset(num_researchers: int, avg_citations_per_researcher: int = 20) -> Tuple[List[Tuple[int, int]], Dict[int, Dict]]:
    """
    Create synthetic dataset for testing large-scale algorithms
    
    Args:
        num_researchers: Number of researchers to generate
        avg_citations_per_researcher: Average number of citations per researcher
    
    Returns:
        Tuple of (edge_list, researchers_dict)
    """
    import random
    
    print(f"Generating synthetic dataset: {num_researchers:,} researchers...")
    
    # Generate researchers with realistic distributions
    researchers = {}
    research_areas = ["AI", "ML", "CV", "NLP", "Robotics", "Bio", "Physics", "Chemistry", "Medicine", "Math"]
    
    for i in range(num_researchers):
        # Generate realistic h-index (log-normal distribution)
        h_index = max(1, int(random.lognormvariate(math.log(20), 0.8)))
        
        researchers[i] = {
            'id': i,
            'name': f'Researcher_{i:07d}',
            'h_index': h_index,
            'research_area': random.choice(research_areas),
            'total_citations': max(10, int(random.lognormvariate(math.log(500), 1.2))),
            'is_nobel_winner': random.random() < 0.0001  # Very rare
        }
    
    # Generate citation edges using preferential attachment model
    print("Generating citation edges using preferential attachment...")
    edges = []
    
    # Track in-degrees for preferential attachment
    in_degrees = [0] * num_researchers
    
    total_edges = num_researchers * avg_citations_per_researcher
    
    for _ in range(total_edges):
        # Choose citing researcher (uniform random)
        from_node = random.randint(0, num_researchers - 1)
        
        # Choose cited researcher (preferential attachment)
        if sum(in_degrees) == 0:
            # First edge, choose randomly
            to_node = random.randint(0, num_researchers - 1)
        else:
            # Preferential attachment with small random component
            weights = [max(degree + 1, 1) for degree in in_degrees]  # +1 for new nodes
            to_node = random.choices(range(num_researchers), weights=weights)[0]
        
        if from_node != to_node:  # No self-citations
            edges.append((from_node, to_node))
            in_degrees[to_node] += 1
    
    # Remove duplicates
    edges = list(set(edges))
    
    print(f"Generated {len(edges):,} unique citation edges")
    return edges, researchers

def main():
    """Main function for testing large-scale ranking algorithms"""
    print("Large-Scale Researcher Ranking Algorithm Test")
    print("=" * 55)
    
    # Test sizes for demonstration (can scale to millions)
    test_configurations = [
        (1000, 15, "Small test (1K researchers)"),
        (10000, 20, "Medium test (10K researchers)"),
        # (100000, 25, "Large test (100K researchers)"),  # Uncomment for larger tests
        # (1000000, 30, "Very large test (1M researchers)"),  # Requires significant RAM
    ]
    
    for num_researchers, avg_citations, description in test_configurations:
        print(f"\n{description}")
        print("-" * len(description))
        
        try:
            # Generate synthetic data
            edges, researchers = create_synthetic_large_dataset(num_researchers, avg_citations)
            
            # Initialize ranking system
            ranking_system = LargeScaleRanking(alpha=0.6, beta=0.4, batch_size=5000)
            
            # Load data and compute rankings
            ranking_system.load_from_edge_list(edges, researchers, num_researchers)
            
            # Test different ranking methods
            print("\n1. Basic Ranking (Equation 1):")
            basic_rankings = ranking_system.compute_basic_ranking_sparse()
            
            top_basic = ranking_system.get_top_k(basic_rankings, 5)
            for i, (node_id, name, score) in enumerate(top_basic):
                h_index = researchers[node_id]['h_index']
                area = researchers[node_id]['research_area']
                print(f"  {i+1}. {name} (H-index: {h_index}, {area}) - Score: {score:.4f}")
            
            print("\n2. PageRank-style Ranking:")
            pagerank_rankings = ranking_system.compute_pagerank_style_ranking()
            
            top_pagerank = ranking_system.get_top_k(pagerank_rankings, 5)
            for i, (node_id, name, score) in enumerate(top_pagerank):
                h_index = researchers[node_id]['h_index']
                area = researchers[node_id]['research_area']
                print(f"  {i+1}. {name} (H-index: {h_index}, {area}) - Score: {score:.6f}")
            
            print("\n3. Collaboration-aware Ranking:")
            collab_rankings = ranking_system.compute_collaboration_ranking_sparse()
            
            top_collab = ranking_system.get_top_k(collab_rankings, 5)
            for i, (node_id, name, score) in enumerate(top_collab):
                h_index = researchers[node_id]['h_index']
                area = researchers[node_id]['research_area']
                print(f"  {i+1}. {name} (H-index: {h_index}, {area}) - Score: {score:.4f}")
            
            # Performance statistics
            ranking_system.print_performance_stats()
            
        except Exception as e:
            print(f"Error in {description}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*55)
    print("SCALING NOTES FOR 5+ MILLION RESEARCHERS:")
    print("- Use batch_size=50000 or higher for very large datasets")
    print("- Consider distributed processing for >10M researchers") 
    print("- Memory requirement: ~8-16 bytes per citation edge")
    print("- For 5M researchers with 0.01% density: ~200GB RAM needed")
    print("- Use SSD storage for virtual memory and data persistence")
    print("="*55)

if __name__ == "__main__":
    main()
