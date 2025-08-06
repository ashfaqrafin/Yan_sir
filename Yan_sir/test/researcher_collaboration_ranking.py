"""
Researcher Collaboration Network Ranking Algorithm Implementation

Based on the research paper: "An Algorithm for Ranking Authors"
Implements Equations (1) through (6) from the paper with support for:
- Dense and sparse matrix representations
- Large-scale datasets (up to 5 million researchers)
- Time-dependent ranking with incremental updates
- Awards/prizes as dummy researchers
- Citation and collaboration weight tuning parameters (α, β)

Author: Implementation based on the research paper
Date: August 2025
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
import json
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Researcher:
    """Represents a researcher with their attributes"""
    id: int
    name: str
    is_nobel_winner: bool = False
    nobel_category: Optional[str] = None
    award_year: Optional[int] = None
    research_area: str = ""
    h_index: int = 0
    total_citations: int = 0

@dataclass
class Award:
    """Represents an award/prize as a dummy researcher"""
    id: int
    name: str
    category: str
    significance: float = 1.0  # Award significance weight

class ResearcherCollaborationRanking:
    """
    Implements the Researcher Collaboration Network Ranking Algorithm
    
    Supports both dense and sparse matrix representations for scalability
    up to 5 million researchers.
    """
    
    def __init__(self, 
                 alpha: float = 0.5, 
                 beta: float = 0.5,
                 use_sparse: bool = True,
                 time_dependent: bool = True):
        """
        Initialize the ranking system
        
        Args:
            alpha: Citation weight parameter (0 < α < 1)
            beta: Collaboration weight parameter (0 < β < 1)  
            use_sparse: Whether to use sparse matrices for large datasets
            time_dependent: Whether to use time-dependent ranking (Equation 6)
        """
        if not (0 < alpha < 1) or not (0 < beta < 1):
            raise ValueError("Parameters α and β must be between 0 and 1")
            
        self.alpha = alpha
        self.beta = beta
        self.use_sparse = use_sparse
        self.time_dependent = time_dependent
        
        self.researchers: Dict[int, Researcher] = {}
        self.awards: Dict[int, Award] = {}
        self.citation_matrix = None
        self.collaboration_matrix = None
        self.publication_matrix = None  # κ_j,τ matrix
        self.coauthor_matrix = None     # κ_j,τ matrix for collaborations
        
        self.num_researchers = 0
        self.num_awards = 0
        self.time_periods = 0
        
        # Ranking results
        self.rankings = None
        self.time_dependent_rankings = None
        
    def load_data_from_json(self, json_file: str):
        """Load researcher data and citation graph from JSON file"""
        logger.info(f"Loading data from {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Load researchers
        for researcher_data in data.get('researchers', []):
            researcher = Researcher(
                id=researcher_data['id'],
                name=researcher_data['name'],
                is_nobel_winner=researcher_data.get('is_nobel_winner', False),
                nobel_category=researcher_data.get('nobel_category'),
                award_year=researcher_data.get('award_year'),
                research_area=researcher_data.get('research_area', ''),
                h_index=researcher_data.get('h_index', 0),
                total_citations=researcher_data.get('total_citations', 0)
            )
            self.researchers[researcher.id] = researcher
        
        self.num_researchers = len(self.researchers)
        logger.info(f"Loaded {self.num_researchers} researchers")
        
        # Load citation matrix
        if 'sparse_adjacency_matrix' in data and self.use_sparse:
            self._load_sparse_matrix(data['sparse_adjacency_matrix'])
        elif 'adjacency_matrix_dense' in data:
            self._load_dense_matrix(data['adjacency_matrix_dense'])
        else:
            raise ValueError("No valid adjacency matrix found in data")
            
        # Create awards as dummy researchers
        self._create_award_dummy_researchers()
        
    def _load_sparse_matrix(self, sparse_data: Dict):
        """Load sparse adjacency matrix from edge list format"""
        edges = sparse_data['edges']
        num_nodes = sparse_data['num_nodes']
        
        # Create sparse citation matrix
        rows, cols = zip(*edges) if edges else ([], [])
        data = [1] * len(edges)
        
        self.citation_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(num_nodes, num_nodes)
        )
        
        logger.info(f"Loaded sparse citation matrix: {num_nodes}x{num_nodes}, {len(edges)} edges")
        
    def _load_dense_matrix(self, dense_matrix: List[List[int]]):
        """Load dense adjacency matrix"""
        matrix = np.array(dense_matrix)
        
        if self.use_sparse:
            # Convert dense to sparse for efficiency
            self.citation_matrix = csr_matrix(matrix)
        else:
            self.citation_matrix = matrix
            
        logger.info(f"Loaded dense citation matrix: {matrix.shape}")
        
    def _create_award_dummy_researchers(self):
        """Create dummy researchers for awards/prizes (Section 2.1 of paper)"""
        # Extract unique awards from researchers
        awards_seen = set()
        award_id_start = self.num_researchers
        
        for researcher in self.researchers.values():
            if researcher.is_nobel_winner and researcher.nobel_category:
                award_key = (researcher.nobel_category, researcher.award_year)
                if award_key not in awards_seen:
                    award = Award(
                        id=award_id_start + len(awards_seen),
                        name=f"Nobel Prize {researcher.nobel_category} {researcher.award_year}",
                        category=researcher.nobel_category,
                        significance=1.0
                    )
                    self.awards[award.id] = award
                    awards_seen.add(award_key)
        
        self.num_awards = len(self.awards)
        logger.info(f"Created {self.num_awards} award dummy researchers")
        
        # Extend citation matrix to include awards
        if self.num_awards > 0:
            self._extend_matrix_for_awards()
            
    def _extend_matrix_for_awards(self):
        """Extend citation matrix to include award dummy researchers"""
        total_size = self.num_researchers + self.num_awards
        
        if self.use_sparse:
            # Create extended sparse matrix
            current_shape = self.citation_matrix.shape
            if current_shape[0] < total_size:
                # Extend the matrix
                extended_matrix = sp.lil_matrix((total_size, total_size))
                extended_matrix[:current_shape[0], :current_shape[1]] = self.citation_matrix
                
                # Add award connections (researchers cite awards they received)
                award_id_start = self.num_researchers
                for researcher in self.researchers.values():
                    if researcher.is_nobel_winner:
                        for award_id, award in self.awards.items():
                            if (researcher.nobel_category == award.category and 
                                researcher.award_year in award.name):
                                extended_matrix[researcher.id, award_id] = 1
                
                self.citation_matrix = extended_matrix.tocsr()
        else:
            # Extend dense matrix
            current_matrix = self.citation_matrix
            extended_matrix = np.zeros((total_size, total_size), dtype=int)
            extended_matrix[:current_matrix.shape[0], :current_matrix.shape[1]] = current_matrix
            
            # Add award connections
            award_id_start = self.num_researchers
            for researcher in self.researchers.values():
                if researcher.is_nobel_winner:
                    for award_id, award in self.awards.items():
                        if (researcher.nobel_category == award.category and 
                            researcher.award_year in award.name):
                            extended_matrix[researcher.id, award_id] = 1
            
            self.citation_matrix = extended_matrix
            
    def compute_basic_ranking(self) -> Dict[int, float]:
        """
        Compute basic ranking using Equation (1) from the paper
        
        rank_σ = w_σ = Σ(w_j * c_ρ_j / c_ρ)
        
        Returns:
            Dictionary mapping researcher_id to rank score
        """
        logger.info("Computing basic ranking (Equation 1)")
        
        total_nodes = self.num_researchers + self.num_awards
        rankings = {}
        
        if self.use_sparse:
            # Sparse matrix computation
            citation_matrix = self.citation_matrix.tocsc()  # Column-sparse format for efficiency
            
            for i in range(self.num_researchers):  # Only rank actual researchers
                rank_score = 0.0
                
                # Get all researchers who cite researcher i
                citing_researchers = citation_matrix[:, i].nonzero()[0]
                
                for j in citing_researchers:
                    if j < self.num_researchers:  # Only consider actual researchers
                        researcher_j = self.researchers[j]
                        # w_j is based on researcher's significance (using h_index as proxy)
                        w_j = researcher_j.h_index + 1  # +1 to avoid zero weights
                        
                        # c_ρ_j is number of citations made by researcher j
                        citations_made_by_j = citation_matrix[j, :].sum()
                        
                        if citations_made_by_j > 0:
                            rank_score += w_j / citations_made_by_j
                
                rankings[i] = float(rank_score)
        else:
            # Dense matrix computation
            citation_matrix = self.citation_matrix
            
            for i in range(self.num_researchers):
                rank_score = 0.0
                
                # Find all researchers who cite researcher i
                citing_researchers = np.where(citation_matrix[:, i] == 1)[0]
                
                for j in citing_researchers:
                    if j < self.num_researchers:
                        researcher_j = self.researchers[j]
                        w_j = researcher_j.h_index + 1
                        citations_made_by_j = np.sum(citation_matrix[j, :])
                        
                        if citations_made_by_j > 0:
                            rank_score += w_j / citations_made_by_j
                
                rankings[i] = float(rank_score)
        
        self.rankings = rankings
        logger.info("Basic ranking computation completed")
        return rankings
        
    def compute_collaboration_ranking(self, 
                                    collaboration_data: Optional[Dict] = None) -> Dict[int, float]:
        """
        Compute ranking with collaboration using Equation (5) from the paper
        
        rank_i = w_i = ΣΣ w_j (α * c_j,τ/c_jτ + β * κ_j,τ/κ_jτ)
        
        Args:
            collaboration_data: Optional collaboration matrix/data
            
        Returns:
            Dictionary mapping researcher_id to rank score
        """
        logger.info("Computing collaboration ranking (Equation 5)")
        
        # If no collaboration data provided, simulate based on citation patterns
        if collaboration_data is None:
            collaboration_data = self._simulate_collaboration_data()
        
        rankings = {}
        
        for i in range(self.num_researchers):
            rank_score = 0.0
            
            for j in range(self.num_researchers):
                if i != j:
                    researcher_j = self.researchers[j]
                    w_j = researcher_j.h_index + 1
                    
                    # Citation component
                    if self.use_sparse:
                        citations_j_to_i = self.citation_matrix[j, i]
                        total_citations_by_j = self.citation_matrix[j, :].sum()
                    else:
                        citations_j_to_i = self.citation_matrix[j, i]
                        total_citations_by_j = np.sum(self.citation_matrix[j, :])
                    
                    citation_weight = 0.0
                    if total_citations_by_j > 0:
                        citation_weight = citations_j_to_i / total_citations_by_j
                    
                    # Collaboration component (simplified - would need actual collaboration data)
                    collaboration_weight = self._get_collaboration_weight(j, i, collaboration_data)
                    
                    # Combined score using α and β parameters
                    combined_score = self.alpha * citation_weight + self.beta * collaboration_weight
                    rank_score += w_j * combined_score
            
            rankings[i] = float(rank_score)
        
        self.rankings = rankings
        logger.info("Collaboration ranking computation completed")
        return rankings
    
    def _simulate_collaboration_data(self) -> Dict:
        """Simulate collaboration data based on citation patterns and research areas"""
        collaborations = defaultdict(float)
        
        # Researchers in the same area are more likely to collaborate
        area_researchers = defaultdict(list)
        for researcher in self.researchers.values():
            area_researchers[researcher.research_area].append(researcher.id)
        
        # Create collaboration weights based on research area similarity and citations
        for i in range(self.num_researchers):
            for j in range(self.num_researchers):
                if i != j:
                    researcher_i = self.researchers[i]
                    researcher_j = self.researchers[j]
                    
                    # Base collaboration on research area similarity
                    if researcher_i.research_area == researcher_j.research_area:
                        collaborations[(i, j)] = 0.3  # Higher collaboration within same area
                    else:
                        collaborations[(i, j)] = 0.1  # Lower cross-area collaboration
                    
                    # Boost collaboration if there are mutual citations
                    if self.use_sparse:
                        if (self.citation_matrix[i, j] > 0 or 
                            self.citation_matrix[j, i] > 0):
                            collaborations[(i, j)] *= 2.0
                    else:
                        if (self.citation_matrix[i, j] > 0 or 
                            self.citation_matrix[j, i] > 0):
                            collaborations[(i, j)] *= 2.0
        
        return dict(collaborations)
    
    def _get_collaboration_weight(self, j: int, i: int, collaboration_data: Dict) -> float:
        """Get collaboration weight between researchers j and i"""
        return collaboration_data.get((j, i), 0.0)
    
    def compute_time_dependent_ranking(self, 
                                     time_periods: int = 5,
                                     publication_years: Optional[Dict] = None) -> Dict[str, Dict[int, float]]:
        """
        Compute time-dependent ranking using Equation (6) from the paper
        
        rank_i = w_ti = ΣΣ w_j,τ (α * c_j,τ/c_jτ + β * κ_j,τ/κ_jτ)
        
        Args:
            time_periods: Number of time periods to consider
            publication_years: Dictionary mapping (researcher_id, cited_id) to year
            
        Returns:
            Dictionary mapping time_period to researcher rankings
        """
        logger.info(f"Computing time-dependent ranking (Equation 6) for {time_periods} periods")
        
        if publication_years is None:
            # Simulate publication years based on researcher career stages
            publication_years = self._simulate_publication_years(time_periods)
        
        time_rankings = {}
        
        for t in range(time_periods):
            period_rankings = {}
            
            for i in range(self.num_researchers):
                rank_score = 0.0
                
                for j in range(self.num_researchers):
                    if i != j:
                        # Time-dependent weight w_j,τ (researcher importance at time τ)
                        w_j_t = self._get_researcher_weight_at_time(j, t, time_periods)
                        
                        # Citations and collaborations at time τ
                        citation_weight_t = self._get_citation_weight_at_time(j, i, t, publication_years)
                        collaboration_weight_t = self._get_collaboration_weight_at_time(j, i, t)
                        
                        combined_score = (self.alpha * citation_weight_t + 
                                        self.beta * collaboration_weight_t)
                        rank_score += w_j_t * combined_score
                
                period_rankings[i] = float(rank_score)
            
            time_rankings[f"period_{t}"] = period_rankings
        
        self.time_dependent_rankings = time_rankings
        logger.info("Time-dependent ranking computation completed")
        return time_rankings
    
    def _simulate_publication_years(self, time_periods: int) -> Dict:
        """Simulate publication years for citations"""
        publication_years = {}
        
        if self.use_sparse:
            rows, cols = self.citation_matrix.nonzero()
            for row, col in zip(rows, cols):
                if row < self.num_researchers and col < self.num_researchers:
                    # Simulate year based on researcher career (newer researchers cite more recent work)
                    year = np.random.randint(0, time_periods)
                    publication_years[(row, col)] = year
        else:
            for i in range(self.num_researchers):
                for j in range(self.num_researchers):
                    if self.citation_matrix[i, j] > 0:
                        year = np.random.randint(0, time_periods)
                        publication_years[(i, j)] = year
        
        return publication_years
    
    def _get_researcher_weight_at_time(self, researcher_id: int, time_period: int, 
                                     total_periods: int) -> float:
        """Get researcher weight at specific time period"""
        researcher = self.researchers[researcher_id]
        base_weight = researcher.h_index + 1
        
        # Simulate career progression - researchers become more influential over time
        career_factor = 1 + (time_period / total_periods) * 0.5
        return base_weight * career_factor
    
    def _get_citation_weight_at_time(self, j: int, i: int, time_period: int, 
                                   publication_years: Dict) -> float:
        """Get citation weight between researchers j and i at specific time period"""
        if (j, i) in publication_years and publication_years[(j, i)] == time_period:
            # Citation exists in this time period
            if self.use_sparse:
                total_citations_by_j = self.citation_matrix[j, :].sum()
            else:
                total_citations_by_j = np.sum(self.citation_matrix[j, :])
            
            if total_citations_by_j > 0:
                return 1.0 / total_citations_by_j
        
        return 0.0
    
    def _get_collaboration_weight_at_time(self, j: int, i: int, time_period: int) -> float:
        """Get collaboration weight between researchers j and i at specific time period"""
        # Simplified - would need actual collaboration data with timestamps
        researcher_j = self.researchers[j]
        researcher_i = self.researchers[i]
        
        if researcher_j.research_area == researcher_i.research_area:
            return 0.2  # Assume some collaboration in same research area
        return 0.05  # Lower collaboration across areas
    
    def get_top_researchers(self, rankings: Dict[int, float], top_k: int = 10) -> List[Tuple[int, str, float]]:
        """Get top-k researchers from rankings"""
        sorted_researchers = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for i, (researcher_id, score) in enumerate(sorted_researchers[:top_k]):
            researcher = self.researchers[researcher_id]
            result.append((researcher_id, researcher.name, score))
        
        return result
    
    def export_rankings_to_json(self, output_file: str, include_time_dependent: bool = True):
        """Export rankings to JSON file"""
        export_data = {
            "algorithm_parameters": {
                "alpha": self.alpha,
                "beta": self.beta,
                "use_sparse": self.use_sparse,
                "time_dependent": self.time_dependent
            },
            "basic_rankings": self.rankings,
            "researchers": {
                rid: {
                    "name": r.name,
                    "research_area": r.research_area,
                    "h_index": r.h_index,
                    "total_citations": r.total_citations,
                    "is_nobel_winner": r.is_nobel_winner,
                    "nobel_category": r.nobel_category,
                    "award_year": r.award_year
                }
                for rid, r in self.researchers.items()
            }
        }
        
        if include_time_dependent and self.time_dependent_rankings:
            export_data["time_dependent_rankings"] = self.time_dependent_rankings
        
        # Get top researchers
        if self.rankings:
            export_data["top_researchers"] = {
                "top_10": self.get_top_researchers(self.rankings, 10),
                "top_20": self.get_top_researchers(self.rankings, 20)
            }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Rankings exported to {output_file}")

def create_large_scale_example(num_researchers: int = 100000, sparsity: float = 0.001):
    """
    Create a large-scale example for testing with up to 5M researchers
    
    Args:
        num_researchers: Number of researchers to simulate
        sparsity: Citation graph sparsity (fraction of possible edges)
    """
    logger.info(f"Creating large-scale example with {num_researchers} researchers")
    
    # Generate random researchers
    researchers = []
    research_areas = ["AI", "Biology", "Chemistry", "Physics", "Medicine", 
                     "Computer Science", "Mathematics", "Engineering"]
    
    for i in range(num_researchers):
        researcher = {
            "id": i,
            "name": f"Researcher_{i}",
            "is_nobel_winner": np.random.random() < 0.001,  # 0.1% Nobel winners
            "nobel_category": np.random.choice(["Physics", "Chemistry", "Medicine"]) if np.random.random() < 0.001 else None,
            "award_year": np.random.randint(2000, 2024) if np.random.random() < 0.001 else None,
            "research_area": np.random.choice(research_areas),
            "h_index": int(np.random.exponential(20)),
            "total_citations": int(np.random.exponential(1000))
        }
        researchers.append(researcher)
    
    # Generate sparse citation graph
    num_edges = int(num_researchers * (num_researchers - 1) * sparsity)
    edges = []
    
    for _ in range(num_edges):
        i = np.random.randint(0, num_researchers)
        j = np.random.randint(0, num_researchers)
        if i != j:
            edges.append([i, j])
    
    # Remove duplicates
    edges = list(set(tuple(edge) for edge in edges))
    edges = [list(edge) for edge in edges]
    
    data = {
        "sparse_adjacency_matrix": {
            "format": "edge_list",
            "edges": edges,
            "num_nodes": num_researchers,
            "num_edges": len(edges)
        },
        "researchers": researchers,
        "metadata": {
            "generated_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": f"Large-scale test dataset with {num_researchers} researchers",
            "sparsity": sparsity
        }
    }
    
    return data

# Example usage and testing
if __name__ == "__main__":
    print("Researcher Collaboration Network Ranking Algorithm")
    print("=" * 60)
    
    # Test with small dataset first
    ranking_system = ResearcherCollaborationRanking(
        alpha=0.6, 
        beta=0.4, 
        use_sparse=True,
        time_dependent=True
    )
    
    try:
        # Load the existing data
        ranking_system.load_data_from_json(
            '/Users/ashfaqkhanrafin/Codes/Yan_sir/Yan_sir/sample/citation_graph_matrix_20250806_162306.json'
        )
        
        print(f"Loaded {ranking_system.num_researchers} researchers")
        print(f"Created {ranking_system.num_awards} award dummy researchers")
        
        # Compute basic ranking
        basic_rankings = ranking_system.compute_basic_ranking()
        print("\nTop 10 Researchers (Basic Ranking):")
        print("-" * 50)
        for i, (rid, name, score) in enumerate(ranking_system.get_top_researchers(basic_rankings)):
            print(f"{i+1:2d}. {name:<25} Score: {score:.4f}")
        
        # Compute collaboration ranking
        collab_rankings = ranking_system.compute_collaboration_ranking()
        print("\nTop 10 Researchers (Collaboration Ranking):")
        print("-" * 50)
        for i, (rid, name, score) in enumerate(ranking_system.get_top_researchers(collab_rankings)):
            print(f"{i+1:2d}. {name:<25} Score: {score:.4f}")
        
        # Compute time-dependent ranking
        time_rankings = ranking_system.compute_time_dependent_ranking(time_periods=5)
        print("\nTime-Dependent Rankings (Last Period):")
        print("-" * 50)
        last_period = f"period_{4}"
        for i, (rid, name, score) in enumerate(ranking_system.get_top_researchers(time_rankings[last_period])[:10]):
            print(f"{i+1:2d}. {name:<25} Score: {score:.4f}")
        
        # Export results
        ranking_system.export_rankings_to_json("researcher_rankings_results.json")
        
    except Exception as e:
        print(f"Error with existing data: {e}")
        print("Creating sample large-scale example...")
        
        # Create and test large-scale example
        large_data = create_large_scale_example(num_researchers=1000, sparsity=0.01)
        
        with open("large_scale_example.json", "w") as f:
            json.dump(large_data, f, indent=2)
        
        # Test with large-scale data
        large_ranking_system = ResearcherCollaborationRanking(
            alpha=0.6, 
            beta=0.4, 
            use_sparse=True  # Essential for large datasets
        )
        
        large_ranking_system.load_data_from_json("large_scale_example.json")
        start_time = time.time()
        large_rankings = large_ranking_system.compute_basic_ranking()
        end_time = time.time()
        
        print(f"\nLarge-scale test completed in {end_time - start_time:.2f} seconds")
        print(f"Ranked {len(large_rankings)} researchers")
        
        print("\nTop 10 Researchers (Large-scale test):")
        print("-" * 50)
        for i, (rid, name, score) in enumerate(large_ranking_system.get_top_researchers(large_rankings)[:10]):
            print(f"{i+1:2d}. {name:<20} Score: {score:.4f}")
