# Author Ranking Algorithm Research Project

This repository contains implementations and research tools for author ranking algorithms, with a focus on citation analysis and Nobel Prize winner identification in academic datasets.

## Overview

The project implements and analyzes algorithms for ranking academic authors based on various metrics including citation patterns, co-authorship networks, and academic achievements. The research is based on "An Algorithm for Ranking Authors" and includes practical implementations for testing and validation.

## Key Features

- **Citation Graph Generation**: Tools to create directed graphs representing citation relationships between researchers
- **Nobel Prize Winner Detection**: Algorithms to identify Nobel laureates in academic databases
- **Author Ranking Algorithms**: Implementation of various ranking metrics and algorithms
- **Data Visualization**: Graph visualization tools for citation networks and ranking results
- **Year-wise Analysis**: Tools for analyzing academic data across different time periods

## Project Structure

```
├── done/                          # Completed implementations
│   ├── array_folder.py           # Array data structure utilities
│   ├── find_nobel_authors_json_output.py  # Nobel author identification
│   ├── nobel_search_sample.py    # Sample Nobel prize search implementation
│   ├── quick_nobel_search.py     # Optimized Nobel search algorithm
│   ├── test_nobel_search.py      # Test suite for Nobel search functionality
│   └── year_wise_extraction.py   # Year-based data extraction tools
├── sample/                        # Sample data and experimental code
│   ├── An_Algorithm_for_Ranking_Authors.pdf  # Research paper reference
│   ├── citation_graph_generator.py           # Citation network generator
│   ├── citation_graph_*.json                 # Generated citation data
│   ├── citation_graph_report_*.html          # Analysis reports
│   ├── citation_graph_report_*.txt           # Text-based reports
│   ├── generate_graph_*.py                   # Graph generation utilities
│   ├── matrix_visualizer.py                  # Matrix visualization tools
│   ├── nobel_authors_found_in_dblp.json     # Identified Nobel authors
│   ├── nobel.json                            # Nobel Prize database
│   └── simple_citation_graph_generator.py    # Simplified graph generator
└── record.txt                     # Data records and statistics
```

## Core Components

### 1. Citation Graph Generation
- **`citation_graph_generator.py`**: Creates directed graphs representing citation relationships between researchers
- **`simple_citation_graph_generator.py`**: Simplified version for testing and experimentation
- **Matrix Representation**: Uses adjacency matrices where `Matrix[i][j] = 1` means researcher i cites researcher j

### 2. Nobel Prize Analysis
- **`nobel.json`**: Comprehensive database of Nobel Prize winners with detailed information
- **Nobel Search Algorithms**: Multiple implementations for identifying Nobel laureates in academic datasets
- **Performance Optimization**: Quick search algorithms for large-scale data processing

### 3. Author Ranking
- Implementation of various ranking algorithms based on citation metrics
- Co-authorship network analysis
- Multi-criteria ranking systems

### 4. Data Visualization
- **HTML Reports**: Interactive visualization of citation networks
- **Matrix Visualization**: Tools for displaying adjacency matrices
- **Graph Generation**: Multiple output formats (HTML, PDF, text)

## Key Algorithms

### Citation-Based Ranking
- Analyzes citation patterns to rank authors
- Considers both direct citations and citation networks
- Implements weighted scoring based on citation quality

### Nobel Winner Identification
- Pattern matching algorithms for identifying Nobel laureates
- Name normalization and fuzzy matching
- Cross-reference validation with official Nobel database

### Network Analysis
- Graph-based metrics for author importance
- Community detection in citation networks
- Centrality measures for ranking

## Data Sources

### Nobel Prize Database
- Complete Nobel Prize winner information from 1901 to present
- Categories: Physics, Chemistry, Medicine, Literature, Peace, Economic Sciences
- Detailed laureate information including motivations and biographical data

### Academic Records
- Year-wise academic publication data
- Citation relationships and co-authorship information
- Temporal analysis capabilities (1800-present)

## Usage Examples

### Generate Citation Graph
```python
from sample.citation_graph_generator import CitationGraphGenerator

# Create a citation graph with 20 researchers, 5 Nobel winners
generator = CitationGraphGenerator(num_researchers=20, num_nobel_winners=5)
generator.generate_researchers()
generator.generate_citation_matrix()
generator.save_data()
```

### Search for Nobel Winners
```python
from done.nobel_search_sample import main

# Run Nobel laureate search on academic data
main()
```

### Visualize Results
```python
from sample.matrix_visualizer import MatrixVisualizer

# Create visualization of citation matrix
visualizer = MatrixVisualizer()
visualizer.generate_report()
```

## Research Applications

- **Academic Impact Assessment**: Evaluate researcher influence and impact
- **Collaboration Network Analysis**: Study co-authorship patterns
- **Trend Analysis**: Identify emerging research areas and influential authors
- **Quality Metrics**: Develop new metrics for academic achievement
- **Predictive Modeling**: Predict future research impact and collaboration patterns

## Technical Requirements

- Python 3.7+
- NumPy for matrix operations
- NetworkX for graph analysis
- Matplotlib for visualization
- JSON for data handling

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Yan_sir

# Install required dependencies
pip install numpy networkx matplotlib
```

## Data Statistics

The project processes extensive academic data:
- Historical records from 1800 to present
- Peak data availability in recent decades
- Comprehensive Nobel Prize database (900+ laureates)
- Citation networks with thousands of relationships

## Research Impact

This project contributes to:
- Development of improved author ranking algorithms
- Better understanding of academic citation patterns
- Tools for identifying influential researchers
- Methods for academic quality assessment

## Future Work

- Integration with larger academic databases (DBLP, Google Scholar, etc.)
- Machine learning approaches for ranking prediction
- Real-time analysis capabilities
- Enhanced visualization tools
- Multi-language support for international datasets

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

## License

This project is for research and educational purposes.

---

*Based on research in "An Algorithm for Ranking Authors" and implementations for academic network analysis.*
