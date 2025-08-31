# Incremental Ranking with JSON Data

This directory contains the implementation for incremental researcher ranking using multi-year JSON citation and collaboration data.

## Files

- `src/incremental_rank_multi_json.py` - Main implementation with bounds-checking for sparse matrix construction
- `data/` - Sample test data files
- `test_incremental_rank.py` - Comprehensive test suite
- `test_error_demo.py` - Demonstration of the bounds checking fix

## Key Features

### Bounds Checking for Sparse Matrix Construction

The `build_matrix` function includes robust bounds checking to prevent `ValueError: row index exceeds matrix dimensions` that can occur when JSON files contain out-of-bounds indices.

**Features:**
- Filters edges where `src >= n_nodes` or `tgt >= n_nodes`
- Filters edges with negative indices
- Logs warnings about filtered edges with statistics
- Handles various JSON formats (edge lists and objects with 'edges' key)
- Graceful error handling for missing or invalid files

### Usage

```python
from src.incremental_rank_multi_json import build_matrix, incremental_ranking

# Build a matrix from JSON file with bounds checking
matrix = build_matrix("data/citations_2020.json", n_nodes=100)

# Run full incremental ranking
ranking = incremental_ranking("data/", years=[2020, 2021], n_nodes=100)
```

### Data Format

JSON files should contain edge lists in one of these formats:

```json
# Format 1: Simple array of [src, tgt] pairs
[
  [0, 1],
  [1, 2],
  [2, 3]
]

# Format 2: Object with 'edges' key
{
  "edges": [
    [0, 1],
    [1, 2]
  ]
}
```

### Testing

Run the test suite:
```bash
python test_incremental_rank.py
```

Demonstrate the bounds checking fix:
```bash
python test_error_demo.py
```

## Dependencies

- numpy
- scipy

Install with: `pip install numpy scipy`