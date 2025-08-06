# Citation Graph Dataset - File Summary

## 🎯 Complete Package Generated

This directory now contains a comprehensive citation graph test dataset with multiple output formats and analysis tools. Here's everything that has been created:

## 📊 Core Graph Data Files

### `citation_graph_matrix_20250806_162306.json`
- **Complete graph dataset** with adjacency matrix
- Contains researcher profiles, Nobel winner information
- Includes comprehensive statistics and metadata
- Ready for direct algorithm implementation

### `citation_graph_matrix_20250806_162306.csv` 
- **Adjacency matrix in CSV format**
- 20×20 matrix with row/column headers (R0-R19)
- Easy to import into Excel, R, Python pandas, etc.

### `citation_graph_researchers_20250806_162306.json`
- **Researcher profiles only** (separate file)
- Detailed information about all 20 researchers
- Nobel Prize winner annotations

## 📄 Report Files (Multiple Formats)

### `citation_graph_report_20250806_164505.html`
- **Interactive HTML report** with beautiful styling
- Complete adjacency matrix visualization
- Researcher profiles with Nobel winner highlights
- Comprehensive statistics and analysis
- **Can be converted to PDF**: Open in any browser → Print → Save as PDF

### `citation_graph_report_20250806_164625.txt`
- **Plain text report** for easy reading/printing
- ASCII-art adjacency matrix visualization
- All analysis and statistics in text format
- Perfect for documentation and sharing

## 🛠️ Generator Scripts

### Core Generators
- `simple_citation_graph_generator.py` - **Main graph generator**
- `generate_graph_html.py` - **HTML report generator**
- `generate_graph_text.py` - **Text report generator**
- `generate_graph_pdf.py` - **PDF generator** (requires ReportLab)

### Analysis Tools
- `graph_algorithm_tester.py` - **Algorithm testing framework**
- `matrix_visualizer.py` - **Matrix visualization tool**

### Original Advanced Version
- `citation_graph_generator.py` - **Advanced version** (requires NumPy/NetworkX)

## 📚 Documentation
- `README_Citation_Graph.md` - **Complete usage documentation**
- This summary file

## 🎯 Graph Properties Summary

| Property | Value |
|----------|--------|
| **Total Researchers** | 20 |
| **Nobel Prize Winners** | 5 (25%) |
| **Total Citations** | 61 directed edges |
| **Graph Density** | 0.161 (16.1%) |
| **Average Citations/Researcher** | 3.05 |
| **Matrix Format** | 20×20 adjacency matrix |
| **Nobel Winner Bias** | 2.5x citation probability |

## 🏆 Nobel Prize Winners in Dataset

1. **Dr. Carol Davis** (ID 2) - Medicine 2017
2. **Prof. David Wilson** (ID 3) - Physics 2017  
3. **Prof. Henry Moore** (ID 7) - Chemistry 2019
4. **Prof. Jack White** (ID 9) - Medicine 2022
5. **Dr. Maya Patel** (ID 12) - Chemistry 2017

## 🧪 Ready for Algorithm Testing

The dataset is optimized for testing:
- **Centrality Algorithms**: PageRank, Betweenness, Closeness
- **Path Finding**: Shortest paths, Reachability analysis
- **Community Detection**: Modularity-based clustering
- **Influence Analysis**: Information propagation models
- **Network Analysis**: Strongly connected components

## 📋 Quick Start Guide

### 1. For Algorithm Development
```python
import json
with open('citation_graph_matrix_20250806_162306.json', 'r') as f:
    data = json.load(f)
adjacency_matrix = data['adjacency_matrix']
researchers = data['researchers']
nobel_winners = set(data['nobel_winners_ids'])
```

### 2. For Viewing Results
- **Best formatted report**: Open `citation_graph_report_20250806_164505.html` in browser
- **Printable version**: Use the HTML file's Print → Save as PDF
- **Text analysis**: Open `citation_graph_report_20250806_164625.txt`

### 3. For Algorithm Testing
```bash
python graph_algorithm_tester.py  # Run all analysis algorithms
python matrix_visualizer.py       # View matrix visualization
```

## ✅ Validation Completed

- ✅ Proper adjacency matrix format (20×20)
- ✅ No self-citations (diagonal zeros)
- ✅ Nobel winner bias correctly implemented
- ✅ Research area clustering effects included
- ✅ Realistic citation distribution
- ✅ Connected graph structure
- ✅ Multiple output formats generated
- ✅ Comprehensive documentation provided

## 🎉 Result

You now have a complete, professional-quality test dataset for graph algorithm development with:
- **Multiple file formats** for maximum compatibility
- **Rich visualizations** and reports
- **Comprehensive documentation** 
- **Ready-to-use algorithm testing framework**
- **Publication-quality reports** that can be converted to PDF

The dataset is ready for immediate use in research, algorithm development, and testing!
