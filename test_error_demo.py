#!/usr/bin/env python3
"""
Test to demonstrate the ValueError that would occur without bounds checking.
"""

import sys
import os
import numpy as np
from scipy.sparse import coo_matrix

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)


def demonstrate_error_without_bounds_checking():
    """Demonstrate the ValueError that occurs without bounds checking."""
    print("Demonstrating the ValueError without bounds checking...")
    
    # Example edges with out-of-bounds indices
    edges = [[0, 1], [1, 2], [100, 2], [1, 150], [200, 300]]
    n_nodes = 10
    
    # Extract source and target indices (without filtering)
    rows = [edge[0] for edge in edges]
    cols = [edge[1] for edge in edges]
    vals = [1.0] * len(edges)
    
    print(f"Attempting to create COO matrix with n_nodes={n_nodes}")
    print(f"Edges: {edges}")
    print(f"Rows: {rows}")
    print(f"Cols: {cols}")
    
    try:
        # This should raise a ValueError
        matrix = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
        print("❌ No error occurred - this is unexpected!")
        return False
    except ValueError as e:
        print(f"✓ Expected ValueError occurred: {e}")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def demonstrate_fix_with_bounds_checking():
    """Demonstrate how the bounds checking fixes the issue."""
    print("\nDemonstrating the fix with bounds checking...")
    
    from incremental_rank_multi_json import build_matrix
    
    # Create a temporary file with problematic data
    test_file = "/tmp/test_edges.json"
    edges = [[0, 1], [1, 2], [100, 2], [1, 150], [200, 300]]
    
    import json
    with open(test_file, 'w') as f:
        json.dump(edges, f)
    
    n_nodes = 10
    
    try:
        matrix = build_matrix(test_file, n_nodes)
        print(f"✓ Matrix created successfully with shape: {matrix.shape}")
        print(f"✓ Number of non-zero elements: {matrix.nnz}")
        
        # Verify all indices are within bounds
        matrix_coo = matrix.tocoo()
        if matrix_coo.nnz > 0:
            max_row = np.max(matrix_coo.row)
            max_col = np.max(matrix_coo.col)
            min_row = np.min(matrix_coo.row)
            min_col = np.min(matrix_coo.col)
            
            print(f"✓ Row indices range: [{min_row}, {max_row}] (valid: [0, {n_nodes-1}])")
            print(f"✓ Column indices range: [{min_col}, {max_col}] (valid: [0, {n_nodes-1}])")
            
            assert max_row < n_nodes, f"Row index {max_row} >= {n_nodes}"
            assert max_col < n_nodes, f"Column index {max_col} >= {n_nodes}"
            assert min_row >= 0, f"Row index {min_row} < 0"
            assert min_col >= 0, f"Column index {min_col} < 0"
        
        # Clean up
        os.remove(test_file)
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error with bounds checking: {e}")
        if os.path.exists(test_file):
            os.remove(test_file)
        return False


def main():
    """Run the demonstration."""
    print("=" * 70)
    print("Demonstrating the issue and fix for out-of-bounds indices")
    print("=" * 70)
    
    success1 = demonstrate_error_without_bounds_checking()
    success2 = demonstrate_fix_with_bounds_checking()
    
    print("\n" + "=" * 70)
    if success1 and success2:
        print("🎉 Demonstration completed successfully!")
        print("The bounds checking in build_matrix() prevents the ValueError")
        print("and allows the script to continue processing valid edges.")
    else:
        print("❌ Demonstration had issues")
    print("=" * 70)


if __name__ == "__main__":
    main()