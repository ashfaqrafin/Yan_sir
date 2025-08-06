# Author Ranking Algorithm Research Project

Research tools for ranking academic authors using citation analysis and Nobel Prize winner identification in academic datasets.

## Features

- **Citation Graph Generation**: Create directed graphs of researcher citation relationships
- **Nobel Prize Detection**: Identify Nobel laureates in academic databases  
- **Author Ranking**: Multiple ranking algorithms based on citation metrics
- **Data Visualization**: Generate reports and graphs for citation networks

## Structure

```
├── done/                    # Completed implementations
│   ├── nobel_search_*.py   # Nobel prize search algorithms
│   ├── test_nobel_search.py # Test suite
│   └── year_wise_extraction.py # Data extraction tools
├── sample/                  # Sample data and experiments
│   ├── citation_graph_generator.py # Citation network generator
│   ├── nobel.json          # Nobel Prize database
│   └── *.json, *.html      # Generated data and reports
└── record.txt              # Data statistics
```

## Key Components

### Citation Analysis
- Adjacency matrix representation where `Matrix[i][j] = 1` means researcher i cites researcher j
- Citation-based ranking algorithms
- Network analysis and centrality measures

### Nobel Prize Integration
- Complete Nobel database (1901-present) across all categories
- Pattern matching and name normalization for laureate identification
- Cross-reference validation algorithms

## Usage

```python
# Generate citation graph
from sample.citation_graph_generator import CitationGraphGenerator
generator = CitationGraphGenerator(num_researchers=20, num_nobel_winners=5)
generator.generate_researchers()
generator.save_data()

# Search for Nobel winners  
from done.nobel_search_sample import main
main()
```

## Requirements

- Python 3.7+
- NumPy, NetworkX, Matplotlib

## Installation

```bash
git clone <repository-url>
cd Yan_sir
pip install numpy networkx matplotlib
```

---

*Research implementation based on "An Algorithm for Ranking Authors"*
