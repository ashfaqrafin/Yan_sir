"""
Demonstration of Researcher Collaboration Network Ranking Algorithm

This script demonstrates the implementation of the ranking algorithm from the research paper
"An Algorithm for Ranking Authors" using your existing citation graph data.

Implements:
- Equation (1): Basic ranking with awards as dummy researchers
- Equation (5): Ranking with citation and collaboration weights (α, β parameters)
- Equation (6): Time-dependent ranking with incremental updates

Works with both dense and sparse matrix representations.
"""

import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

def load_citation_data(json_file: str) -> Tuple[Dict, List[List[int]], Dict]:
    """Load citation data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    researchers = {r['id']: r for r in data['researchers']}
    
    # Get citation matrix (prefer dense for this demo)
    if 'adjacency_matrix_dense' in data:
        citation_matrix = data['adjacency_matrix_dense']
    else:
        # Convert sparse to dense
        n = data['sparse_adjacency_matrix']['num_nodes']
        citation_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for edge in data['sparse_adjacency_matrix']['edges']:
            i, j = edge
            citation_matrix[i][j] = 1
    
    # Create award dummy researchers
    awards = {}
    award_id = len(researchers)
    awards_seen = set()
    
    for researcher in researchers.values():
        if researcher.get('is_nobel_winner', False):
            award_key = (researcher.get('nobel_category'), researcher.get('award_year'))
            if award_key not in awards_seen:
                awards[award_id] = {
                    'name': f"Nobel Prize {researcher.get('nobel_category')} {researcher.get('award_year')}",
                    'category': researcher.get('nobel_category'),
                    'year': researcher.get('award_year'),
                    'type': 'award'
                }
                awards_seen.add(award_key)
                award_id += 1
    
    return researchers, citation_matrix, awards

def extend_matrix_for_awards(citation_matrix: List[List[int]], researchers: Dict, awards: Dict) -> List[List[int]]:
    """Extend citation matrix to include award dummy researchers"""
    n_researchers = len(researchers)
    n_awards = len(awards)
    n_total = n_researchers + n_awards
    
    # Create extended matrix
    extended_matrix = [[0 for _ in range(n_total)] for _ in range(n_total)]
    
    # Copy original matrix
    for i in range(n_researchers):
        for j in range(n_researchers):
            extended_matrix[i][j] = citation_matrix[i][j]
    
    # Add connections from researchers to their awards
    award_mapping = {}
    award_id_start = n_researchers
    
    for i, award in enumerate(awards.values()):
        award_mapping[(award['category'], award['year'])] = award_id_start + i
    
    for researcher_id, researcher in researchers.items():
        if researcher.get('is_nobel_winner', False):
            award_key = (researcher.get('nobel_category'), researcher.get('award_year'))
            if award_key in award_mapping:
                award_idx = award_mapping[award_key]
                extended_matrix[researcher_id][award_idx] = 1
    
    return extended_matrix

def compute_basic_ranking(citation_matrix: List[List[int]], researchers: Dict) -> Dict[int, float]:
    """
    Compute basic ranking using Equation (1) from the paper:
    rank_σ = w_σ = Σ(w_j * c_ρj / c_ρ)
    
    Where:
    - w_j is the weight (significance) of researcher j
    - c_ρj is 1 if researcher j cites the award ρ, 0 otherwise
    - c_ρ is the total number of researchers who received award ρ
    """
    n = len(researchers)
    rankings = {}
    
    print("Computing basic ranking (Equation 1)...")
    
    for i in range(n):  # For each researcher i
        rank_score = 0.0
        
        # Find all researchers who cite researcher i
        for j in range(len(citation_matrix)):  # j cites i
            if j < n and citation_matrix[j][i] > 0:  # j is a researcher and cites i
                # w_j: weight of researcher j (using h_index as significance measure)
                w_j = researchers[j].get('h_index', 1)
                
                # c_ρ: total citations made by researcher j
                total_citations_by_j = sum(citation_matrix[j])
                
                if total_citations_by_j > 0:
                    # Add weighted contribution
                    rank_score += w_j / total_citations_by_j
        
        rankings[i] = rank_score
    
    return rankings

def simulate_collaboration_data(researchers: Dict, citation_matrix: List[List[int]]) -> Dict[Tuple[int, int], float]:
    """Simulate collaboration data based on research areas and citation patterns"""
    collaborations = {}
    n = len(researchers)
    
    # Group researchers by research area
    area_groups = defaultdict(list)
    for rid, researcher in researchers.items():
        area_groups[researcher.get('research_area', 'Unknown')].append(rid)
    
    # Create collaboration weights
    for i in range(n):
        for j in range(n):
            if i != j:
                researcher_i = researchers[i]
                researcher_j = researchers[j]
                
                # Base collaboration based on research area
                if researcher_i.get('research_area') == researcher_j.get('research_area'):
                    base_collab = 0.3  # Higher collaboration within same area
                else:
                    base_collab = 0.1  # Lower cross-area collaboration
                
                # Boost if there are mutual citations
                mutual_cites = citation_matrix[i][j] + citation_matrix[j][i]
                if mutual_cites > 0:
                    base_collab *= 2.0
                
                # Consider h-index similarity
                h_i = researcher_i.get('h_index', 1)
                h_j = researcher_j.get('h_index', 1)
                similarity = 1.0 / (1.0 + abs(h_i - h_j) / 50.0)  # Normalize difference
                
                collaborations[(i, j)] = base_collab * similarity
    
    return collaborations

def compute_collaboration_ranking(citation_matrix: List[List[int]], 
                                 researchers: Dict, 
                                 alpha: float = 0.5, 
                                 beta: float = 0.5) -> Dict[int, float]:
    """
    Compute ranking with collaboration using Equation (5) from the paper:
    rank_i = w_i = ΣΣ w_j (α * c_j,τ/c_jτ + β * κ_j,τ/κ_jτ)
    
    Where:
    - α, β are tuning parameters (0 < α, β < 1)
    - c_j,τ/c_jτ is the citation weight (citations from j to i / total citations by j)
    - κ_j,τ/κ_jτ is the collaboration weight (collaborations from j to i / total collaborations by j)
    """
    n = len(researchers)
    rankings = {}
    
    print(f"Computing collaboration ranking (Equation 5) with α={alpha}, β={beta}...")
    
    # Get collaboration data
    collaborations = simulate_collaboration_data(researchers, citation_matrix)
    
    # Precompute total citations and collaborations for each researcher
    total_citations = {}
    total_collaborations = {}
    
    for j in range(n):
        total_citations[j] = sum(citation_matrix[j][:n])  # Only count citations to researchers
        total_collaborations[j] = sum(collaborations.get((j, i), 0.0) for i in range(n) if i != j)
    
    for i in range(n):  # For each researcher i
        rank_score = 0.0
        
        for j in range(n):  # From each researcher j
            if i != j:
                # w_j: weight of researcher j
                w_j = researchers[j].get('h_index', 1)
                
                # Citation component: c_j,i / c_j
                citation_weight = 0.0
                if total_citations[j] > 0:
                    citation_weight = citation_matrix[j][i] / total_citations[j]
                
                # Collaboration component: κ_j,i / κ_j
                collaboration_weight = 0.0
                if total_collaborations[j] > 0:
                    collaboration_weight = collaborations.get((j, i), 0.0) / total_collaborations[j]
                
                # Combined weighted score
                combined_score = alpha * citation_weight + beta * collaboration_weight
                rank_score += w_j * combined_score
        
        rankings[i] = rank_score
    
    return rankings

def simulate_time_periods(researchers: Dict, citation_matrix: List[List[int]], periods: int = 5) -> Dict:
    """Simulate time-dependent data for demonstration"""
    n = len(researchers)
    
    # Simulate when citations were made (based on researcher "age" approximated by h-index)
    citation_times = {}
    for i in range(n):
        for j in range(n):
            if citation_matrix[i][j] > 0:
                # Newer researchers (lower h-index) tend to make citations in later periods
                researcher_age = researchers[i].get('h_index', 1)
                base_period = min(periods - 1, researcher_age // 25)  # Rough approximation
                # Add some randomness but keep it deterministic
                import random
                random.seed(i * n + j)
                time_period = max(0, min(periods - 1, base_period + random.randint(-1, 1)))
                citation_times[(i, j)] = time_period
    
    # Simulate researcher weights over time (career progression)
    researcher_weights_over_time = {}
    for rid, researcher in researchers.items():
        base_weight = researcher.get('h_index', 1)
        weights = []
        for t in range(periods):
            # Career progression: researchers become more influential over time
            career_factor = 1.0 + (t / max(periods - 1, 1)) * 0.3
            
            # Nobel winners get additional boost in later periods
            if researcher.get('is_nobel_winner', False):
                award_boost = 1.0 + (t / max(periods - 1, 1)) * 0.5
                career_factor *= award_boost
            
            weights.append(base_weight * career_factor)
        researcher_weights_over_time[rid] = weights
    
    return {
        'citation_times': citation_times,
        'researcher_weights': researcher_weights_over_time
    }

def compute_time_dependent_ranking(citation_matrix: List[List[int]], 
                                  researchers: Dict, 
                                  time_periods: int = 5,
                                  alpha: float = 0.5, 
                                  beta: float = 0.5) -> Dict[str, Dict[int, float]]:
    """
    Compute time-dependent ranking using Equation (6) from the paper:
    rank_i = w_ti = ΣΣ w_j,τ (α * c_j,τ/c_jτ + β * κ_j,τ/κ_jτ)
    
    Where the weights and relationships are time-dependent.
    """
    n = len(researchers)
    
    print(f"Computing time-dependent ranking (Equation 6) for {time_periods} periods...")
    
    # Get time-dependent data
    time_data = simulate_time_periods(researchers, citation_matrix, time_periods)
    citation_times = time_data['citation_times']
    researcher_weights = time_data['researcher_weights']
    
    # Get collaboration data
    collaborations = simulate_collaboration_data(researchers, citation_matrix)
    
    rankings_by_period = {}
    
    for t in range(time_periods):
        period_rankings = {}
        
        # Get citations and collaborations active in period t
        period_citations = defaultdict(dict)
        period_citation_totals = defaultdict(int)
        
        for (i, j), time_period in citation_times.items():
            if time_period == t:
                period_citations[i][j] = 1
                period_citation_totals[i] += 1
        
        for i in range(n):
            rank_score = 0.0
            
            for j in range(n):
                if i != j:
                    # Time-dependent weight w_j,τ
                    w_j_t = researcher_weights[j][t]
                    
                    # Citation component for period t
                    citation_weight_t = 0.0
                    if period_citation_totals[j] > 0:
                        citation_weight_t = period_citations[j].get(i, 0) / period_citation_totals[j]
                    
                    # Collaboration component (simplified - assume constant over time)
                    collab_total = sum(collaborations.get((j, k), 0.0) for k in range(n) if k != j)
                    collaboration_weight_t = 0.0
                    if collab_total > 0:
                        collaboration_weight_t = collaborations.get((j, i), 0.0) / collab_total
                    
                    # Combined score
                    combined_score = alpha * citation_weight_t + beta * collaboration_weight_t
                    rank_score += w_j_t * combined_score
            
            period_rankings[i] = rank_score
        
        rankings_by_period[f"period_{t}"] = period_rankings
    
    return rankings_by_period

def get_top_researchers(rankings: Dict[int, float], researchers: Dict, top_k: int = 10) -> List[Tuple[int, str, float, bool]]:
    """Get top-k researchers from rankings"""
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    
    result = []
    for i, (rid, score) in enumerate(sorted_rankings[:top_k]):
        researcher = researchers[rid]
        is_nobel = researcher.get('is_nobel_winner', False)
        result.append((rid, researcher['name'], score, is_nobel))
    
    return result

def print_rankings(rankings: Dict[int, float], researchers: Dict, title: str, top_k: int = 10):
    """Print formatted rankings"""
    print(f"\n{title}")
    print("-" * 70)
    
    top_researchers = get_top_researchers(rankings, researchers, top_k)
    
    for i, (rid, name, score, is_nobel) in enumerate(top_researchers):
        nobel_symbol = "🏆" if is_nobel else "  "
        h_index = researchers[rid].get('h_index', 0)
        area = researchers[rid].get('research_area', 'Unknown')
        print(f"{i+1:2d}. {nobel_symbol} {name:<25} Score: {score:8.4f} (H-index: {h_index:3d}, {area})")

def analyze_rankings(rankings: Dict[int, float], researchers: Dict) -> Dict:
    """Analyze ranking results"""
    scores = list(rankings.values())
    nobel_winners = [rid for rid, r in researchers.items() if r.get('is_nobel_winner', False)]
    
    top_10 = get_top_researchers(rankings, researchers, 10)
    top_20 = get_top_researchers(rankings, researchers, 20)
    
    nobel_in_top_10 = sum(1 for _, _, _, is_nobel in top_10 if is_nobel)
    nobel_in_top_20 = sum(1 for _, _, _, is_nobel in top_20 if is_nobel)
    
    return {
        'total_researchers': len(rankings),
        'nobel_winners_total': len(nobel_winners),
        'nobel_in_top_10': nobel_in_top_10,
        'nobel_in_top_20': nobel_in_top_20,
        'score_range': (min(scores) if scores else 0, max(scores) if scores else 0),
        'mean_score': sum(scores) / len(scores) if scores else 0
    }

def main():
    """Main demonstration function"""
    print("Researcher Collaboration Network Ranking Algorithm Demo")
    print("=" * 65)
    print("Based on: 'An Algorithm for Ranking Authors'")
    print("Implements Equations (1), (5), and (6) from the research paper")
    print()
    
    # Load data
    json_file = '/Users/ashfaqkhanrafin/Codes/Yan_sir/Yan_sir/sample/citation_graph_matrix_20250806_162306.json'
    
    try:
        researchers, citation_matrix, awards = load_citation_data(json_file)
        print(f"Loaded {len(researchers)} researchers and {len(awards)} awards")
        
        # Extend matrix for awards (following paper's approach)
        extended_matrix = extend_matrix_for_awards(citation_matrix, researchers, awards)
        print(f"Extended citation matrix to {len(extended_matrix)}x{len(extended_matrix[0])}")
        
        # 1. Basic Ranking (Equation 1)
        print(f"\n1. BASIC RANKING (Equation 1)")
        print("="*50)
        basic_rankings = compute_basic_ranking(extended_matrix, researchers)
        print_rankings(basic_rankings, researchers, "Top 10 Researchers - Basic Ranking")
        
        # 2. Collaboration Ranking with different α, β values (Equation 5)
        print(f"\n2. COLLABORATION RANKING (Equation 5)")
        print("="*50)
        
        # Test different parameter combinations
        param_combinations = [
            (0.7, 0.3, "Citation-Heavy"),
            (0.5, 0.5, "Balanced"),
            (0.3, 0.7, "Collaboration-Heavy")
        ]
        
        for alpha, beta, label in param_combinations:
            print(f"\n{label} (α={alpha}, β={beta}):")
            collab_rankings = compute_collaboration_ranking(citation_matrix, researchers, alpha, beta)
            print_rankings(collab_rankings, researchers, f"Top 10 - {label}", 10)
        
        # 3. Time-Dependent Ranking (Equation 6)
        print(f"\n3. TIME-DEPENDENT RANKING (Equation 6)")
        print("="*50)
        
        time_rankings = compute_time_dependent_ranking(citation_matrix, researchers, 
                                                     time_periods=3, alpha=0.6, beta=0.4)
        
        for period, rankings in time_rankings.items():
            print_rankings(rankings, researchers, f"Top 10 - {period.replace('_', ' ').title()}", 10)
        
        # 4. Analysis and Comparison
        print(f"\n4. ANALYSIS AND COMPARISON")
        print("="*50)
        
        # Compare how Nobel winners rank in different methods
        final_collab_rankings = compute_collaboration_ranking(citation_matrix, researchers, 0.6, 0.4)
        
        basic_analysis = analyze_rankings(basic_rankings, researchers)
        collab_analysis = analyze_rankings(final_collab_rankings, researchers)
        
        print(f"\nBasic Ranking Analysis:")
        print(f"  Nobel winners in top 10: {basic_analysis['nobel_in_top_10']}/{basic_analysis['nobel_winners_total']}")
        print(f"  Score range: {basic_analysis['score_range'][0]:.4f} - {basic_analysis['score_range'][1]:.4f}")
        print(f"  Mean score: {basic_analysis['mean_score']:.4f}")
        
        print(f"\nCollaboration Ranking Analysis:")
        print(f"  Nobel winners in top 10: {collab_analysis['nobel_in_top_10']}/{collab_analysis['nobel_winners_total']}")
        print(f"  Score range: {collab_analysis['score_range'][0]:.4f} - {collab_analysis['score_range'][1]:.4f}")
        print(f"  Mean score: {collab_analysis['mean_score']:.4f}")
        
        # 5. Export results
        print(f"\n5. EXPORTING RESULTS")
        print("="*50)
        
        results = {
            'algorithm_info': {
                'paper_title': 'An Algorithm for Ranking Authors',
                'equations_implemented': ['Equation 1', 'Equation 5', 'Equation 6'],
                'parameters': {
                    'alpha_citation_weight': 0.6,
                    'beta_collaboration_weight': 0.4
                }
            },
            'dataset_info': {
                'num_researchers': len(researchers),
                'num_awards': len(awards),
                'citation_matrix_size': f"{len(citation_matrix)}x{len(citation_matrix[0])}",
                'extended_matrix_size': f"{len(extended_matrix)}x{len(extended_matrix[0])}"
            },
            'rankings': {
                'basic_ranking': {rid: score for rid, score in basic_rankings.items()},
                'collaboration_ranking': {rid: score for rid, score in final_collab_rankings.items()},
                'time_dependent_ranking': time_rankings
            },
            'top_researchers': {
                'basic_top_10': [(rid, name, score) for rid, name, score, _ in get_top_researchers(basic_rankings, researchers, 10)],
                'collaboration_top_10': [(rid, name, score) for rid, name, score, _ in get_top_researchers(final_collab_rankings, researchers, 10)]
            },
            'analysis': {
                'basic_ranking': basic_analysis,
                'collaboration_ranking': collab_analysis
            },
            'researchers': researchers
        }
        
        output_file = 'researcher_ranking_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results exported to: {output_file}")
        
        print(f"\n6. ALGORITHM NOTES")
        print("="*50)
        print("• Basic Ranking (Eq. 1): Uses awards as dummy researchers")
        print("• Collaboration Ranking (Eq. 5): Combines citations and collaborations with α, β weights")  
        print("• Time-Dependent Ranking (Eq. 6): Considers temporal evolution of researcher influence")
        print("• Sparse matrices recommended for datasets > 10,000 researchers")
        print("• Parameters α, β should sum to 1 for proper normalization")
        print("• Algorithm complexity: O(N²) for dense, O(E) for sparse matrices")
        
    except FileNotFoundError:
        print(f"Error: Could not find the data file at {json_file}")
        print("Please ensure the file path is correct.")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()
