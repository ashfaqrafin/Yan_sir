#!/usr/bin/env python3
"""
Test script for incremental_rank_multi_json.py

This script tests the build_matrix function with data containing out-of-bounds indices
to verify that the filtering works correctly.
"""

import sys
import os
import numpy as np

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from incremental_rank_multi_json import build_matrix, load_json_edges, incremental_ranking


def test_build_matrix_with_invalid_indices():
    """Test build_matrix function with out-of-bounds indices."""
    print("Testing build_matrix with invalid indices...")
    
    # Test with n_nodes = 10 (valid indices: 0-9)
    n_nodes = 10
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    citation_file = os.path.join(data_dir, 'citations_2020.json')
    
    # Load edges to see what we're working with
    edges = load_json_edges(citation_file)
    print(f"Original edges: {edges}")
    
    # Build matrix with filtering
    matrix = build_matrix(citation_file, n_nodes)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix nnz (non-zeros): {matrix.nnz}")
    
    # Convert to dense for inspection
    dense_matrix = matrix.toarray()
    print(f"Non-zero positions:")
    nonzero_rows, nonzero_cols = np.nonzero(dense_matrix)
    for i in range(len(nonzero_rows)):
        print(f"  [{nonzero_rows[i]}, {nonzero_cols[i]}] = {dense_matrix[nonzero_rows[i], nonzero_cols[i]]}")
    
    # Verify that all indices are within bounds
    assert np.all(nonzero_rows < n_nodes), "Found row index >= n_nodes"
    assert np.all(nonzero_cols < n_nodes), "Found column index >= n_nodes"
    assert np.all(nonzero_rows >= 0), "Found negative row index"
    assert np.all(nonzero_cols >= 0), "Found negative column index"
    
    print("✓ All indices are within valid bounds")
    return True


def test_empty_file():
    """Test behavior with empty or non-existent files."""
    print("\nTesting with non-existent file...")
    
    n_nodes = 5
    matrix = build_matrix("non_existent_file.json", n_nodes)
    
    assert matrix.shape == (n_nodes, n_nodes), f"Expected shape ({n_nodes}, {n_nodes}), got {matrix.shape}"
    assert matrix.nnz == 0, f"Expected 0 non-zeros, got {matrix.nnz}"
    
    print("✓ Non-existent file handled correctly")


def test_invalid_json_format():
    """Test with invalid JSON format."""
    print("\nTesting with invalid JSON format...")
    
    # Create a temporary file with invalid JSON
    temp_file = "/tmp/invalid.json"
    with open(temp_file, 'w') as f:
        f.write('{"invalid": "json", "missing": "edges"}')
    
    n_nodes = 5
    matrix = build_matrix(temp_file, n_nodes)
    
    assert matrix.shape == (n_nodes, n_nodes), f"Expected shape ({n_nodes}, {n_nodes}), got {matrix.shape}"
    assert matrix.nnz == 0, f"Expected 0 non-zeros, got {matrix.nnz}"
    
    # Clean up
    os.remove(temp_file)
    print("✓ Invalid JSON format handled correctly")


def test_full_pipeline():
    """Test the full incremental ranking pipeline."""
    print("\nTesting full incremental ranking pipeline...")
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    years = [2020]
    n_nodes = 10
    
    # This should not raise any errors despite invalid indices in the data
    try:
        ranking = incremental_ranking(data_dir, years, n_nodes)
        
        assert len(ranking) == n_nodes, f"Expected ranking length {n_nodes}, got {len(ranking)}"
        assert np.all(ranking >= 0), "Found negative ranking values"
        
        print(f"Final ranking shape: {ranking.shape}")
        print(f"Ranking sum: {np.sum(ranking):.6f}")
        print(f"Top 3 researchers: {np.argsort(ranking)[::-1][:3]}")
        
        print("✓ Full pipeline completed without errors")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline failed with error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing incremental_rank_multi_json.py")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    try:
        if test_build_matrix_with_invalid_indices():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    try:
        test_empty_file()
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    try:
        test_invalid_json_format()
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    try:
        if test_full_pipeline():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())