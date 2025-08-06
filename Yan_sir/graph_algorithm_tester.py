"""
Graph Algorithm Testing Framework

This script demonstrates how to use the generated citation graph to test various algorithms.
It includes examples of common graph algorithms that can be applied to citation networks.
"""

import json
import sys
from collections import deque, defaultdict

class CitationGraphAlgorithms:
    def __init__(self, graph_file):
        """Initialize with the citation graph data"""
        with open(graph_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.adjacency_matrix = self.data['adjacency_matrix']
        self.researchers = self.data['researchers']
        self.nobel_winners = set(self.data['nobel_winners_ids'])
        self.n = len(self.adjacency_matrix)
        
        # Create adjacency list for easier traversal
        self.adjacency_list = self._matrix_to_list()
    
    def _matrix_to_list(self):
        """Convert adjacency matrix to adjacency list"""
        adj_list = defaultdict(list)
        for i in range(self.n):
            for j in range(self.n):
                if self.adjacency_matrix[i][j] == 1:
                    adj_list[i].append(j)
        return adj_list
    
    def pagerank(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        """
        Calculate PageRank scores for all researchers
        Higher scores indicate more "influential" researchers
        """
        print("Running PageRank Algorithm...")
        
        # Initialize PageRank values
        pr = [1.0 / self.n for _ in range(self.n)]
        new_pr = [0.0 for _ in range(self.n)]
        
        # Calculate out-degrees
        out_degrees = [sum(row) for row in self.adjacency_matrix]
        
        for iteration in range(max_iterations):
            # Calculate new PageRank values
            for i in range(self.n):
                new_pr[i] = (1 - damping_factor) / self.n
                for j in range(self.n):
                    if self.adjacency_matrix[j][i] == 1 and out_degrees[j] > 0:
                        new_pr[i] += damping_factor * pr[j] / out_degrees[j]
            
            # Check for convergence
            diff = sum(abs(new_pr[i] - pr[i]) for i in range(self.n))
            if diff < tolerance:
                print(f"PageRank converged after {iteration + 1} iterations")
                break
            
            pr, new_pr = new_pr, pr
        
        # Rank researchers by PageRank score
        ranked = [(i, pr[i]) for i in range(self.n)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return pr, ranked
    
    def find_shortest_paths(self, source):
        """
        Find shortest paths from source to all other researchers using BFS
        Returns distances and paths
        """
        print(f"Finding shortest paths from researcher {source}...")
        
        distances = [-1] * self.n
        paths = [[] for _ in range(self.n)]
        
        queue = deque([source])
        distances[source] = 0
        paths[source] = [source]
        
        while queue:
            current = queue.popleft()
            
            for neighbor in self.adjacency_list[current]:
                if distances[neighbor] == -1:  # Not visited
                    distances[neighbor] = distances[current] + 1
                    paths[neighbor] = paths[current] + [neighbor]
                    queue.append(neighbor)
        
        return distances, paths
    
    def find_strongly_connected_components(self):
        """
        Find strongly connected components using Tarjan's algorithm
        """
        print("Finding strongly connected components...")
        
        index_counter = [0]
        stack = []
        lowlinks = [0] * self.n
        index = [0] * self.n
        on_stack = [False] * self.n
        index_assigned = [False] * self.n
        components = []
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            index_assigned[node] = True
            stack.append(node)
            on_stack[node] = True
            
            for neighbor in self.adjacency_list[node]:
                if not index_assigned[neighbor]:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack[neighbor]:
                    lowlinks[node] = min(lowlinks[node], index[neighbor])
            
            if lowlinks[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break
                components.append(component)
        
        for node in range(self.n):
            if not index_assigned[node]:
                strongconnect(node)
        
        return components
    
    def analyze_nobel_winner_influence(self):
        """
        Analyze the citation patterns of Nobel Prize winners
        """
        print("Analyzing Nobel Prize winner influence...")
        
        analysis = {}
        
        for nobel_id in self.nobel_winners:
            researcher = self.researchers[nobel_id]
            
            # Count citations received (in-degree)
            citations_received = sum(self.adjacency_matrix[i][nobel_id] for i in range(self.n))
            
            # Count citations made (out-degree)
            citations_made = sum(self.adjacency_matrix[nobel_id])
            
            # Find who cites this Nobel winner
            cited_by = [i for i in range(self.n) if self.adjacency_matrix[i][nobel_id] == 1]
            
            # Find whom this Nobel winner cites
            cites = [i for i in range(self.n) if self.adjacency_matrix[nobel_id][i] == 1]
            
            analysis[nobel_id] = {
                'name': researcher['name'],
                'category': researcher['nobel_category'],
                'year': researcher['award_year'],
                'citations_received': citations_received,
                'citations_made': citations_made,
                'cited_by': cited_by,
                'cites': cites,
                'cited_by_nobel_winners': len([i for i in cited_by if i in self.nobel_winners]),
                'cites_nobel_winners': len([i for i in cites if i in self.nobel_winners])
            }
        
        return analysis
    
    def find_citation_clusters(self):
        """
        Find clusters of researchers who frequently cite each other
        """
        print("Finding citation clusters...")
        
        # Use simple clustering based on mutual citations
        clusters = []
        visited = set()
        
        for i in range(self.n):
            if i in visited:
                continue
            
            cluster = [i]
            visited.add(i)
            
            for j in range(i + 1, self.n):
                if j in visited:
                    continue
                
                # Check if there's mutual citation or strong connection
                mutual_citation = (self.adjacency_matrix[i][j] == 1 and 
                                 self.adjacency_matrix[j][i] == 1)
                
                if mutual_citation:
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def run_all_algorithms(self):
        """Run all algorithms and display results"""
        print("=" * 80)
        print("CITATION GRAPH ALGORITHM TESTING FRAMEWORK")
        print("=" * 80)
        
        print(f"\nGraph Statistics:")
        print(f"- Total Researchers: {self.n}")
        print(f"- Nobel Prize Winners: {len(self.nobel_winners)}")
        print(f"- Total Citations: {sum(sum(row) for row in self.adjacency_matrix)}")
        
        # 1. PageRank Analysis
        print("\n" + "=" * 60)
        print("1. PAGERANK ANALYSIS")
        print("=" * 60)
        
        pr_scores, pr_ranked = self.pagerank()
        
        print("\nTop 10 Most Influential Researchers (by PageRank):")
        for i, (researcher_id, score) in enumerate(pr_ranked[:10]):
            researcher = self.researchers[researcher_id]
            nobel_status = "🏆" if researcher_id in self.nobel_winners else ""
            print(f"{i+1:2d}. ID {researcher_id:2d}: {researcher['name']} {nobel_status} "
                  f"(Score: {score:.4f})")
        
        # 2. Shortest Paths from Nobel Winners
        print("\n" + "=" * 60)
        print("2. SHORTEST PATHS ANALYSIS")
        print("=" * 60)
        
        for nobel_id in list(self.nobel_winners)[:2]:  # Show first 2 Nobel winners
            researcher = self.researchers[nobel_id]
            print(f"\nShortest paths from {researcher['name']} (ID {nobel_id}):")
            
            distances, paths = self.find_shortest_paths(nobel_id)
            
            reachable = [(i, distances[i]) for i in range(self.n) if distances[i] > 0]
            reachable.sort(key=lambda x: x[1])
            
            for target_id, dist in reachable[:5]:  # Show first 5
                target = self.researchers[target_id]
                nobel_status = "🏆" if target_id in self.nobel_winners else ""
                print(f"  → {target['name']} {nobel_status} (distance: {dist})")
        
        # 3. Strongly Connected Components
        print("\n" + "=" * 60)
        print("3. STRONGLY CONNECTED COMPONENTS")
        print("=" * 60)
        
        components = self.find_strongly_connected_components()
        print(f"Found {len(components)} strongly connected components:")
        
        for i, component in enumerate(components):
            if len(component) > 1:
                print(f"\nComponent {i+1} ({len(component)} researchers):")
                for researcher_id in component:
                    researcher = self.researchers[researcher_id]
                    nobel_status = "🏆" if researcher_id in self.nobel_winners else ""
                    print(f"  - ID {researcher_id}: {researcher['name']} {nobel_status}")
        
        # 4. Nobel Winner Analysis
        print("\n" + "=" * 60)
        print("4. NOBEL PRIZE WINNER ANALYSIS")
        print("=" * 60)
        
        nobel_analysis = self.analyze_nobel_winner_influence()
        
        for nobel_id, analysis in nobel_analysis.items():
            print(f"\n{analysis['name']} ({analysis['category']} {analysis['year']}):")
            print(f"  Citations received: {analysis['citations_received']}")
            print(f"  Citations made: {analysis['citations_made']}")
            print(f"  Cited by Nobel winners: {analysis['cited_by_nobel_winners']}")
            print(f"  Cites Nobel winners: {analysis['cites_nobel_winners']}")
        
        # 5. Citation Clusters
        print("\n" + "=" * 60)
        print("5. CITATION CLUSTERS")
        print("=" * 60)
        
        clusters = self.find_citation_clusters()
        
        if clusters:
            print(f"Found {len(clusters)} citation clusters:")
            for i, cluster in enumerate(clusters):
                print(f"\nCluster {i+1}:")
                for researcher_id in cluster:
                    researcher = self.researchers[researcher_id]
                    nobel_status = "🏆" if researcher_id in self.nobel_winners else ""
                    print(f"  - ID {researcher_id}: {researcher['name']} {nobel_status}")
        else:
            print("No mutual citation clusters found.")
        
        print("\n" + "=" * 80)
        print("ALGORITHM TESTING COMPLETE")
        print("=" * 80)


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
    print(f"Using graph file: {graph_file}")
    
    # Run algorithm tests
    algorithms = CitationGraphAlgorithms(graph_file)
    algorithms.run_all_algorithms()


if __name__ == "__main__":
    main()
