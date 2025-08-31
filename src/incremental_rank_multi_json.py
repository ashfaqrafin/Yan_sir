"""
Incremental ranking algorithm using multi-year JSON citation and collaboration data.

This script handles JSON files containing citation and collaboration data and builds
sparse matrices while filtering out-of-bounds indices to prevent ValueError.
"""

import json
import numpy as np
from scipy.sparse import coo_matrix
import logging
from collections import defaultdict
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json_edges(file_path):
    """Load edges from a JSON file."""
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} does not exist")
        return []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'edges' in data:
                return data['edges']
            else:
                logger.error(f"Unexpected JSON format in {file_path}")
                return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return []


def build_matrix(file_path, n_nodes, value=1.0):
    """
    Build a sparse COO matrix from a JSON file containing edges.
    
    This function filters out any edges where src >= n_nodes or tgt >= n_nodes
    to prevent ValueError during sparse matrix construction.
    
    Args:
        file_path (str): Path to JSON file containing edge data
        n_nodes (int): Number of nodes in the graph (valid indices: 0 to n_nodes-1)
        value (float): Value to assign to edges (default: 1.0)
    
    Returns:
        scipy.sparse.coo_matrix: Sparse matrix with filtered edges
    """
    # Load edges from JSON file
    edges = load_json_edges(file_path)
    
    if not edges:
        logger.warning(f"No edges loaded from {file_path}, returning empty matrix")
        return coo_matrix((n_nodes, n_nodes))
    
    # Lists to store valid edges
    rows = []
    cols = []
    vals = []
    
    # Counters for logging
    total_edges = len(edges)
    filtered_edges = 0
    
    # Process each edge and filter out invalid ones
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            logger.warning(f"Invalid edge format: {edge}")
            filtered_edges += 1
            continue
            
        src = edge[0]
        tgt = edge[1]
        
        # Check if indices are valid integers
        if not isinstance(src, int) or not isinstance(tgt, int):
            logger.warning(f"Non-integer indices in edge: [{src}, {tgt}]")
            filtered_edges += 1
            continue
        
        # Filter out edges with out-of-bounds indices
        if src >= n_nodes or tgt >= n_nodes or src < 0 or tgt < 0:
            logger.debug(f"Filtering out-of-bounds edge: [{src}, {tgt}] (n_nodes={n_nodes})")
            filtered_edges += 1
            continue
        
        # Add valid edge
        rows.append(src)
        cols.append(tgt)
        vals.append(value)
    
    # Log filtering results
    valid_edges = total_edges - filtered_edges
    if filtered_edges > 0:
        logger.warning(f"Filtered {filtered_edges}/{total_edges} edges from {file_path} "
                      f"({filtered_edges/total_edges*100:.2f}% of edges were invalid)")
    
    logger.info(f"Building matrix from {file_path}: {valid_edges} valid edges")
    
    # Build COO matrix with filtered edges
    if not rows:  # No valid edges
        logger.warning(f"No valid edges found in {file_path}, returning empty matrix")
        return coo_matrix((n_nodes, n_nodes))
    
    try:
        matrix = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
        return matrix
    except Exception as e:
        logger.error(f"Error creating sparse matrix: {e}")
        return coo_matrix((n_nodes, n_nodes))


def normalize_matrix(matrix):
    """
    Row-normalize a sparse matrix so each row sums to 1.
    
    Args:
        matrix: scipy.sparse matrix
    
    Returns:
        scipy.sparse matrix: Row-normalized matrix
    """
    # Convert to CSR format for efficient row operations
    matrix_csr = matrix.tocsr()
    
    # Calculate row sums
    row_sums = np.array(matrix_csr.sum(axis=1)).flatten()
    
    # Find non-zero rows
    nonzero_rows = row_sums > 0
    
    # Normalize non-zero rows
    if np.any(nonzero_rows):
        matrix_csr.data = matrix_csr.data.astype(float)
        for i in range(matrix_csr.shape[0]):
            if row_sums[i] > 0:
                start = matrix_csr.indptr[i]
                end = matrix_csr.indptr[i + 1]
                matrix_csr.data[start:end] /= row_sums[i]
    
    return matrix_csr


def incremental_ranking(data_dir, years, n_nodes, alpha=0.6, beta=0.4, seed_vector=None):
    """
    Perform incremental ranking across multiple years.
    
    Args:
        data_dir (str): Directory containing citation and collaboration JSON files
        years (list): List of years to process
        n_nodes (int): Number of nodes in the graph
        alpha (float): Weight for citation matrix
        beta (float): Weight for collaboration matrix
        seed_vector (np.array): Initial ranking vector
    
    Returns:
        np.array: Final ranking vector
    """
    if seed_vector is None:
        # Initialize with uniform distribution
        seed_vector = np.ones(n_nodes) / n_nodes
    
    current_ranking = seed_vector.copy()
    
    for year in sorted(years):
        logger.info(f"Processing year {year}")
        
        # Build citation matrix
        citation_file = os.path.join(data_dir, f"citations_{year}.json")
        C = build_matrix(citation_file, n_nodes)
        C = normalize_matrix(C)
        
        # Build collaboration matrix
        collab_file = os.path.join(data_dir, f"collaborations_{year}.json")
        K = build_matrix(collab_file, n_nodes)
        K = normalize_matrix(K)
        
        # Combine matrices
        M = alpha * C + beta * K
        
        # Apply one step of ranking update
        # Simple approach: R_new = M^T * R_old
        current_ranking = M.T.dot(current_ranking)
        
        # Normalize to maintain probability distribution
        if np.sum(current_ranking) > 0:
            current_ranking = current_ranking / np.sum(current_ranking)
        
        logger.info(f"Year {year} completed. Top score: {np.max(current_ranking):.6f}")
    
    return current_ranking


def main():
    """Main function for testing the implementation."""
    # Example usage
    data_dir = "../data"
    years = [2020, 2021, 2022]
    n_nodes = 100  # Example: 100 researchers
    
    try:
        final_ranking = incremental_ranking(data_dir, years, n_nodes)
        
        # Display top 10 researchers
        top_indices = np.argsort(final_ranking)[::-1][:10]
        
        print("\nTop 10 researchers:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1:2d}. Researcher {idx}: {final_ranking[idx]:.6f}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()