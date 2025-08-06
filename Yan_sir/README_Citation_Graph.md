# Citation Graph Test Dataset

## Overview
This directory contains a test graph representing citation relationships between 20 researchers, where some are Nobel Prize winners. The graph is designed to test various graph algorithms and is represented as an adjacency matrix.

## Dataset Structure

### Graph Properties
- **Total Researchers**: 20 (IDs: 0-19)
- **Nobel Prize Winners**: 5 researchers
- **Total Citations**: 61 directed edges
- **Graph Density**: 0.161 (16.1% of possible edges exist)
- **Representation**: Adjacency Matrix where `matrix[i][j] = 1` means researcher `i` cites researcher `j`

### Nobel Prize Winners in the Dataset
1. **Dr. Carol Davis** (ID 2) - Medicine 2017
2. **Prof. David Wilson** (ID 3) - Physics 2017  
3. **Prof. Henry Moore** (ID 7) - Chemistry 2019
4. **Prof. Jack White** (ID 9) - Medicine 2022
5. **Dr. Maya Patel** (ID 12) - Chemistry 2017

### Generated Files
- `citation_graph_matrix_YYYYMMDD_HHMMSS.json` - Complete graph data with adjacency matrix
- `citation_graph_matrix_YYYYMMDD_HHMMSS.csv` - Adjacency matrix in CSV format
- `citation_graph_researchers_YYYYMMDD_HHMMSS.json` - Researcher details

## Scripts Included

### 1. `simple_citation_graph_generator.py`
**Purpose**: Generates the test citation graph
**Features**:
- Creates 20 researchers with realistic profiles
- Randomly assigns 5 Nobel Prize winners
- Generates citation relationships with bias toward Nobel winners
- Saves data in JSON and CSV formats

**Usage**:
```bash
python simple_citation_graph_generator.py
```

### 2. `graph_algorithm_tester.py`
**Purpose**: Demonstrates various graph algorithms on the citation network
**Algorithms Included**:
- **PageRank**: Identifies most influential researchers
- **Shortest Paths**: Finds citation paths between researchers
- **Strongly Connected Components**: Identifies citation clusters
- **Nobel Winner Analysis**: Analyzes citation patterns of Nobel winners
- **Citation Clusters**: Finds mutual citation groups

**Usage**:
```bash
python graph_algorithm_tester.py
```

### 3. `matrix_visualizer.py`
**Purpose**: Provides clear visualization of the adjacency matrix
**Features**:
- Shows matrix with researcher names
- Highlights Nobel Prize winners with 🏆
- Displays citation patterns and statistics
- Uses visual symbols (● for citations, · for no citation)

**Usage**:
```bash
python matrix_visualizer.py
```

## How to Use This Test Graph

### 1. For Algorithm Development
The adjacency matrix can be used to test various graph algorithms:

```python
import json

# Load the graph data
with open('citation_graph_matrix_YYYYMMDD_HHMMSS.json', 'r') as f:
    data = json.load(f)

adjacency_matrix = data['adjacency_matrix']
researchers = data['researchers']
nobel_winners = set(data['nobel_winners_ids'])

# Your algorithm here
def your_algorithm(adjacency_matrix):
    # Implementation
    pass
```

### 2. For Testing Specific Scenarios

#### Testing Citation Influence
- Nobel winners have higher citation rates (both received and made)
- Test algorithms that identify influential nodes

#### Testing Path Finding
- Find citation paths between researchers
- Analyze reachability in the citation network

#### Testing Community Detection
- Identify research clusters
- Find strongly connected components

#### Testing Centrality Measures
- PageRank, Betweenness Centrality, Closeness Centrality
- Compare results for Nobel winners vs. regular researchers

### 3. Example Test Cases

#### Test Case 1: Most Influential Researcher
Expected: Prof. Jack White (ID 9) should rank highest in most centrality measures

#### Test Case 2: Nobel Winner Bias
Expected: Nobel winners should generally have higher citation counts

#### Test Case 3: Reachability
Expected: Most researchers should be reachable from Nobel winners within 2-3 steps

#### Test Case 4: Clustering
Expected: Some researchers form citation clusters (mutual citations)

## Graph Statistics

### Citation Distribution
- **Most Cited**: Prof. Jack White 🏆 (7 citations)
- **Most Active Citer**: Prof. Frank Miller (6 citations made)
- **Average Citations per Researcher**: 3.05

### Nobel Winner Statistics
- **Citations to Nobel Winners**: 24 (39% of all citations)
- **Citations by Nobel Winners**: 21 (34% of all citations)
- **Nobel-to-Nobel Citations**: Multiple instances of Nobel winners citing each other

### Network Properties
- **Strongly Connected**: Most researchers form one large strongly connected component
- **Citation Clusters**: 4 mutual citation clusters identified
- **Isolated Nodes**: Prof. Rachel Chen (ID 17) has no outgoing citations

## Algorithm Testing Recommendations

### 1. Centrality Algorithms
Test PageRank, Eigenvector Centrality, Betweenness Centrality:
- Verify Nobel winners rank higher
- Compare different centrality measures

### 2. Path Algorithms
Test shortest path, all-pairs shortest paths:
- Analyze citation propagation paths
- Find most "connected" researchers

### 3. Community Detection
Test Louvain, Leiden, or custom clustering:
- Identify research communities
- Analyze interdisciplinary citations

### 4. Influence Propagation
Test influence spread algorithms:
- Model how citations propagate
- Compare influence of Nobel winners

## Data Quality Notes

- **Realistic Bias**: Nobel winners have 2.5x higher probability of being cited
- **Research Areas**: Researchers in same area cite each other more frequently
- **Temporal Consistency**: Award years are realistic (2015-2023)
- **Scale**: 20 researchers provide good test size without being overwhelming

## Extending the Dataset

To modify the test graph:

1. **Change Size**: Modify `num_researchers` parameter in generator
2. **Adjust Nobel Winners**: Modify `num_nobel_winners` parameter  
3. **Change Citation Probability**: Modify `citation_probability` parameter
4. **Adjust Nobel Bias**: Modify `nobel_bias` parameter

## Validation

The generated graph has been validated for:
- ✅ Proper adjacency matrix format
- ✅ No self-citations (diagonal zeros)
- ✅ Realistic citation distribution
- ✅ Nobel winner bias implementation
- ✅ Connected graph structure
- ✅ JSON format correctness

This test dataset provides a realistic, controlled environment for testing graph algorithms on citation networks with known properties and expected behaviors.
