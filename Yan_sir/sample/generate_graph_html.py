"""
Citation Graph HTML Report Generator

This script generates a comprehensive HTML report containing all details of the citation graph,
which can be easily converted to PDF using any browser's print-to-PDF functionality.
"""

import json
from datetime import datetime
import os

class CitationGraphHTMLGenerator:
    def __init__(self, graph_file):
        """Initialize with citation graph data"""
        with open(graph_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Handle both sparse and dense matrix formats
        if 'adjacency_matrix_dense' in self.data:
            self.adjacency_matrix = self.data['adjacency_matrix_dense']
            self.sparse_matrix = self.data['sparse_adjacency_matrix']
        elif 'adjacency_matrix' in self.data:
            self.adjacency_matrix = self.data['adjacency_matrix']
            self.sparse_matrix = None
        else:
            raise ValueError("No adjacency matrix found in the data file")
            
        self.researchers = self.data['researchers']
        self.nobel_winners = set(self.data['nobel_winners_ids'])
        self.statistics = self.data['statistics']
        self.metadata = self.data['metadata']
        self.n = len(self.adjacency_matrix)
    
    def generate_css(self):
        """Generate CSS styles for the HTML report"""
        return """
        <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin: 0 0 10px 0;
        }
        
        .header h2 {
            font-size: 1.2em;
            margin: 0;
            opacity: 0.9;
        }
        
        .section {
            background: #f8f9fa;
            padding: 25px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }
        
        .section h3 {
            color: #27ae60;
            margin-top: 25px;
        }
        
        .overview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background: #34495e;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .nobel-winner {
            background-color: #fff3cd !important;
        }
        
        .nobel-badge {
            background: #ffc107;
            color: #000;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .matrix-table {
            font-family: monospace;
            font-size: 12px;
            margin: 20px 0;
        }
        
        .matrix-table th, .matrix-table td {
            padding: 6px;
            text-align: center;
            border: 1px solid #ccc;
            width: 30px;
            height: 30px;
        }
        
        .matrix-cell-cite {
            background-color: #28a745;
            color: white;
            font-weight: bold;
        }
        
        .matrix-cell-no {
            background-color: #f8f9fa;
            color: #6c757d;
        }
        
        .matrix-nobel-header {
            background-color: #ffc107 !important;
            color: #000 !important;
            font-weight: bold;
        }
        
        .citation-flow {
            display: flex;
            align-items: center;
            margin: 5px 0;
            padding: 8px;
            background: white;
            border-radius: 5px;
        }
        
        .arrow {
            margin: 0 15px;
            font-size: 1.2em;
            color: #3498db;
        }
        
        .algorithm-card {
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #9b59b6;
            border-radius: 0 8px 8px 0;
        }
        
        .print-break {
            page-break-before: always;
        }
        
        @media print {
            body { font-size: 12px; }
            .section { break-inside: avoid; }
            .matrix-table { font-size: 8px; }
        }
        </style>
        """
    
    def generate_header(self):
        """Generate HTML header section"""
        gen_date = datetime.fromisoformat(self.metadata['generated_date']).strftime("%B %d, %Y at %I:%M %p")
        
        return f"""
        <div class="header">
            <h1>Citation Graph ({self.statistics['total_researchers']} Researchers)</h1>
            <h2>Comprehensive Analysis Report</h2>
            <p>Generated: {gen_date}</p>
        </div>
        
        <div class="overview-grid">
            <div class="stat-card">
                <div class="stat-value">{self.statistics['total_researchers']}</div>
                <div class="stat-label">Total Researchers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.statistics['nobel_winners']}</div>
                <div class="stat-label">Nobel Prize Winners</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.statistics['total_citations']}</div>
                <div class="stat-label">Total Citations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.statistics['graph_density']:.3f}</div>
                <div class="stat-label">Graph Density</div>
            </div>
        </div>
        """
    
    def generate_researcher_section(self):
        """Generate researcher profiles section"""
        html = """
        <div class="section">
            <h2>Researcher Profiles</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Nobel Prize</th>
                        <th>Research Area</th>
                        <th>H-Index</th>
                        <th>Citations</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for researcher in self.researchers:
            row_class = "nobel-winner" if researcher['is_nobel_winner'] else ""
            nobel_info = ""
            if researcher['is_nobel_winner']:
                nobel_info = f'<span class="nobel-badge">🏆 {researcher["nobel_category"]} {researcher["award_year"]}</span>'
            
            html += f"""
                    <tr class="{row_class}">
                        <td>{researcher['id']}</td>
                        <td><strong>{researcher['name']}</strong></td>
                        <td>{nobel_info}</td>
                        <td>{researcher['research_area']}</td>
                        <td>{researcher['h_index']}</td>
                        <td>{researcher['total_citations']:,}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <h3>Nobel Prize Winners Details</h3>
        """
        
        for nobel_id in sorted(self.nobel_winners):
            researcher = self.researchers[nobel_id]
            citations_received = sum(self.adjacency_matrix[i][nobel_id] for i in range(self.n))
            citations_made = sum(self.adjacency_matrix[nobel_id])
            
            html += f"""
            <div style="background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107;">
                <h4 style="margin-top: 0; color: #2c3e50;">🏆 {researcher['name']} (ID {nobel_id})</h4>
                <p><strong>Nobel Prize:</strong> {researcher['nobel_category']} ({researcher['award_year']})</p>
                <p><strong>Research Area:</strong> {researcher['research_area']}</p>
                <p><strong>Academic Profile:</strong> H-Index: {researcher['h_index']}, Total Citations: {researcher['total_citations']:,}</p>
                <p><strong>In Citation Graph:</strong> Receives {citations_received} citations, Makes {citations_made} citations</p>
            </div>
            """
        
        html += "</div>"
        return html
    
    def generate_matrix_section(self):
        """Generate adjacency matrix section"""
        html = f"""
        <div class="section print-break">
            <h2>Adjacency Matrix Representations</h2>
            <p>The adjacency matrix represents citation relationships where matrix[i][j] = 1 means researcher i cites researcher j. 
               Nobel Prize winners are highlighted in gold.</p>
        """
        
        # Add sparse matrix information if available
        if self.sparse_matrix:
            html += f"""
            <h3>Sparse Matrix Representation</h3>
            <div style="background: white; padding: 20px; border-radius: 5px; margin: 15px 0;">
                <p><strong>Format:</strong> {self.sparse_matrix['format']}</p>
                <p><strong>Storage Efficiency:</strong> {self.metadata.get('storage_efficiency', 'N/A')}</p>
                <p><strong>Number of Edges:</strong> {self.sparse_matrix['num_edges']} out of {self.sparse_matrix['num_nodes']}² = {self.sparse_matrix['num_nodes']**2} possible connections</p>
                
                <h4>Edge List (First 30 edges shown):</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 15px 0;">
            """
            
            # Display first 30 edges with researcher names
            edges_to_show = min(30, len(self.sparse_matrix['edges']))
            for i in range(edges_to_show):
                source, target = self.sparse_matrix['edges'][i]
                source_name = self.researchers[source]['name'].split()[-1]  # Last name
                target_name = self.researchers[target]['name'].split()[-1]  # Last name
                source_badge = '🏆' if source in self.nobel_winners else ''
                target_badge = '🏆' if target in self.nobel_winners else ''
                
                html += f"""
                    <div style="background: #f8f9fa; padding: 8px; border-radius: 4px; font-size: 0.9em;">
                        [{source}→{target}] {source_name}{source_badge} → {target_name}{target_badge}
                    </div>
                """
            
            if len(self.sparse_matrix['edges']) > edges_to_show:
                html += f"<div style='grid-column: 1/-1; text-align: center; font-style: italic; color: #666;'>... and {len(self.sparse_matrix['edges']) - edges_to_show} more edges</div>"
            
            html += """
                </div>
            </div>
            
            <h3>Dense Matrix Visualization</h3>
            """
        
        html += f"""
            <table class="matrix-table">
                <thead>
                    <tr>
                        <th></th>
        """
        
        # Column headers
        for j in range(self.n):
            header_class = "matrix-nobel-header" if j in self.nobel_winners else ""
            html += f'<th class="{header_class}">{j}</th>'
        
        html += """
                    </tr>
                </thead>
                <tbody>
        """
        
        # Matrix rows
        for i in range(self.n):
            row_header_class = "matrix-nobel-header" if i in self.nobel_winners else ""
            html += f'<tr><th class="{row_header_class}">{i}</th>'
            
            for j in range(self.n):
                if self.adjacency_matrix[i][j] == 1:
                    html += '<td class="matrix-cell-cite">●</td>'
                else:
                    html += '<td class="matrix-cell-no">·</td>'
            
            html += "</tr>"
        
        html += """
                </tbody>
            </table>
            
            <div style="background: white; padding: 15px; border-radius: 5px; margin-top: 20px;">
                <h4>Legend:</h4>
                <p><strong>●</strong> = Citation exists (researcher in row cites researcher in column)</p>
                <p><strong>·</strong> = No citation</p>
                <p><span style="background: #ffc107; padding: 2px 8px; border-radius: 3px;">Gold</span> = Nobel Prize winner</p>
                <p>Numbers represent researcher IDs (0-{self.n-1})</p>
        """
        
        if self.sparse_matrix:
            html += f"""
                <p><strong>Sparse Format:</strong> [source→target] notation shows citation direction</p>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def generate_statistics_section(self):
        """Generate statistics section"""
        # Most cited researchers
        citations_received = [(i, self.statistics['in_degrees'][i]) for i in range(self.n)]
        citations_received.sort(key=lambda x: x[1], reverse=True)
        
        # Most active citers
        citations_made = [(i, self.statistics['out_degrees'][i]) for i in range(self.n)]
        citations_made.sort(key=lambda x: x[1], reverse=True)
        
        html = f"""
        <div class="section print-break">
            <h2>Graph Statistics and Analysis</h2>
            
            <h3>Basic Statistics</h3>
            <div style="background: white; padding: 20px; border-radius: 5px;">
                <ul>
                    <li><strong>Total Researchers:</strong> {self.statistics['total_researchers']}</li>
                    <li><strong>Nobel Prize Winners:</strong> {self.statistics['nobel_winners']} ({self.statistics['nobel_winners']/self.statistics['total_researchers']*100:.1f}%)</li>
                    <li><strong>Total Citation Edges:</strong> {self.statistics['total_citations']}</li>
                    <li><strong>Graph Density:</strong> {self.statistics['graph_density']:.3f} ({self.statistics['graph_density']*100:.1f}% of possible edges)</li>
                    <li><strong>Average Citations per Researcher:</strong> {self.statistics['average_citations_per_researcher']:.2f}</li>
                    <li><strong>Maximum Citations Received:</strong> {self.statistics['max_citations_received']}</li>
                    <li><strong>Maximum Citations Made:</strong> {self.statistics['max_citations_made']}</li>
                </ul>
            </div>
            
            <h3>Top 10 Most Cited Researchers</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Researcher</th>
                        <th>Citations Received</th>
                        <th>Nobel Prize</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for rank, (researcher_id, citations) in enumerate(citations_received[:10], 1):
            researcher = self.researchers[researcher_id]
            nobel_status = f'<span class="nobel-badge">🏆</span>' if researcher_id in self.nobel_winners else "No"
            row_class = "nobel-winner" if researcher_id in self.nobel_winners else ""
            
            html += f"""
                    <tr class="{row_class}">
                        <td>{rank}</td>
                        <td><strong>{researcher['name']}</strong></td>
                        <td>{citations}</td>
                        <td>{nobel_status}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <h3>Top 10 Most Active Citers</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Researcher</th>
                        <th>Citations Made</th>
                        <th>Nobel Prize</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for rank, (researcher_id, citations) in enumerate(citations_made[:10], 1):
            researcher = self.researchers[researcher_id]
            nobel_status = f'<span class="nobel-badge">🏆</span>' if researcher_id in self.nobel_winners else "No"
            row_class = "nobel-winner" if researcher_id in self.nobel_winners else ""
            
            html += f"""
                    <tr class="{row_class}">
                        <td>{rank}</td>
                        <td><strong>{researcher['name']}</strong></td>
                        <td>{citations}</td>
                        <td>{nobel_status}</td>
                    </tr>
            """
        
        html += f"""
                </tbody>
            </table>
            
            <h3>Nobel Prize Winner Analysis</h3>
            <div style="background: white; padding: 20px; border-radius: 5px;">
                <ul>
                    <li><strong>Citations to Nobel Winners:</strong> {self.statistics['nobel_winners_citations_received']} ({self.statistics['nobel_winners_citations_received']/self.statistics['total_citations']*100:.1f}% of all citations)</li>
                    <li><strong>Citations by Nobel Winners:</strong> {self.statistics['nobel_winners_citations_made']} ({self.statistics['nobel_winners_citations_made']/self.statistics['total_citations']*100:.1f}% of all citations)</li>
                    <li><strong>Average Citations Received by Nobel Winners:</strong> {self.statistics['nobel_winners_citations_received']/len(self.nobel_winners):.1f}</li>
                    <li><strong>Average Citations Received by Non-Nobel Winners:</strong> {(self.statistics['total_citations'] - self.statistics['nobel_winners_citations_received'])/(self.n - len(self.nobel_winners)):.1f}</li>
                </ul>
            </div>
        </div>
        """
        
        return html
    
    def generate_citation_patterns_section(self):
        """Generate citation patterns section"""
        # Find all citations involving Nobel winners
        nobel_citations = []
        for i in range(self.n):
            for j in range(self.n):
                if self.adjacency_matrix[i][j] == 1 and (i in self.nobel_winners or j in self.nobel_winners):
                    citing = self.researchers[i]['name']
                    cited = self.researchers[j]['name']
                    citing_nobel = i in self.nobel_winners
                    cited_nobel = j in self.nobel_winners
                    nobel_citations.append((i, j, citing, cited, citing_nobel, cited_nobel))
        
        html = f"""
        <div class="section print-break">
            <h2>Citation Patterns</h2>
            
            <h3>Citations Involving Nobel Prize Winners ({len(nobel_citations)} total)</h3>
            <div style="background: white; padding: 15px; border-radius: 5px; max-height: 400px; overflow-y: auto;">
        """
        
        for i, j, citing, cited, citing_nobel, cited_nobel in nobel_citations[:20]:  # Show first 20
            citing_badge = '🏆' if citing_nobel else ''
            cited_badge = '🏆' if cited_nobel else ''
            
            html += f"""
                <div class="citation-flow">
                    <span><strong>{citing}</strong> {citing_badge}</span>
                    <span class="arrow">→</span>
                    <span><strong>{cited}</strong> {cited_badge}</span>
                </div>
            """
        
        if len(nobel_citations) > 20:
            html += f"<p><em>... and {len(nobel_citations) - 20} more citations</em></p>"
        
        html += """
            </div>
            
            <h3>Research Area Distribution</h3>
        """
        
        # Research area analysis
        research_areas = {}
        for researcher in self.researchers:
            area = researcher['research_area']
            if area not in research_areas:
                research_areas[area] = {'total': 0, 'nobel': 0}
            research_areas[area]['total'] += 1
            if researcher['is_nobel_winner']:
                research_areas[area]['nobel'] += 1
        
        html += """
            <table>
                <thead>
                    <tr>
                        <th>Research Area</th>
                        <th>Total Researchers</th>
                        <th>Nobel Winners</th>
                        <th>% Nobel</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for area, counts in sorted(research_areas.items()):
            nobel_pct = (counts['nobel'] / counts['total']) * 100 if counts['total'] > 0 else 0
            html += f"""
                    <tr>
                        <td><strong>{area}</strong></td>
                        <td>{counts['total']}</td>
                        <td>{counts['nobel']}</td>
                        <td>{nobel_pct:.1f}%</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def generate_algorithm_section(self):
        """Generate algorithm testing section"""
        html = """
        <div class="section print-break">
            <h2>Algorithm Testing Guidelines</h2>
            
            <p>This citation graph is designed for testing various graph algorithms. Both sparse (edge list) and dense (adjacency matrix) formats are available for different algorithm requirements and performance testing.</p>
            
            <h3>Data Format Options</h3>
            <div style="background: white; padding: 20px; border-radius: 5px; margin: 15px 0;">
        """
        
        if self.sparse_matrix:
            html += f"""
                <h4>🔹 Sparse Matrix (Recommended for large graphs)</h4>
                <ul>
                    <li><strong>Format:</strong> Edge list with {self.sparse_matrix['num_edges']} edges</li>
                    <li><strong>Memory Usage:</strong> {self.metadata.get('storage_efficiency', 'Efficient storage')}</li>
                    <li><strong>Best For:</strong> Graph traversal, pathfinding, network analysis algorithms</li>
                    <li><strong>Access:</strong> <code>data['sparse_adjacency_matrix']['edges']</code></li>
                </ul>
                
                <h4>🔹 Dense Matrix (Recommended for matrix operations)</h4>
                <ul>
                    <li><strong>Format:</strong> Full {self.n}×{self.n} adjacency matrix</li>
                    <li><strong>Memory Usage:</strong> {self.n**2} elements ({self.n**2 - self.sparse_matrix['num_edges']} zeros)</li>
                    <li><strong>Best For:</strong> Linear algebra operations, centrality measures</li>
                    <li><strong>Access:</strong> <code>data['adjacency_matrix_dense']</code></li>
                </ul>
            """
        else:
            html += """
                <h4>🔹 Dense Matrix Format</h4>
                <ul>
                    <li><strong>Format:</strong> Full adjacency matrix</li>
                    <li><strong>Access:</strong> <code>data['adjacency_matrix']</code></li>
                </ul>
            """
        
        html += """
            </div>
            
            <h3>Recommended Algorithm Categories</h3>
            
            <div class="algorithm-card">
                <h4>Centrality Measures</h4>
                <p>PageRank, Betweenness Centrality, Closeness Centrality, Eigenvector Centrality</p>
                <p><strong>Expected:</strong> Nobel winners should generally score higher, especially Prof. Jack White (ID 9)</p>
                <p><strong>Format:</strong> Use dense matrix for linear algebra-based algorithms</p>
            </div>
            
            <div class="algorithm-card">
                <h4>Path Finding Algorithms</h4>
                <p>Shortest paths, All-pairs shortest paths, Reachability analysis</p>
                <p><strong>Expected:</strong> Average path length ~2-3 steps, most researchers reachable from Nobel winners</p>
                <p><strong>Format:</strong> Use sparse edge list for better performance</p>
            </div>
            
            <div class="algorithm-card">
                <h4>Community Detection</h4>
                <p>Louvain, Leiden, Modularity-based clustering</p>
                <p><strong>Expected:</strong> Research area clustering effects, one main connected component</p>
                <p><strong>Format:</strong> Edge list format typically preferred</p>
            </div>
            
            <div class="algorithm-card">
                <h4>Influence Analysis</h4>
                <p>Information propagation, Influence maximization, Network flow</p>
                <p><strong>Expected:</strong> Nobel winner bias in influence spread models</p>
                <p><strong>Format:</strong> Sparse format ideal for simulation algorithms</p>
            </div>
            
            <div class="algorithm-card">
                <h4>Network Analysis</h4>
                <p>Strongly connected components, Topological sorting, Graph traversal</p>
                <p><strong>Expected:</strong> One large strongly connected component containing most researchers</p>
                <p><strong>Format:</strong> Edge list format for graph traversal algorithms</p>
            </div>
            
            <h3>Data Quality Features</h3>
            <div style="background: white; padding: 20px; border-radius: 5px;">
                <ul>
                    <li><strong>Realistic Bias:</strong> Nobel winners have 2.5x higher probability of being cited</li>
                    <li><strong>Research Clustering:</strong> Researchers in same area cite each other more frequently</li>
                    <li><strong>Temporal Consistency:</strong> Award years are realistic (2015-2023)</li>
                    <li><strong>Scale:</strong> 20 researchers provide good test size without being overwhelming</li>
                    <li><strong>No Self-Citations:</strong> Diagonal elements are all zero</li>
                    <li><strong>Connected Structure:</strong> Most nodes are reachable from each other</li>
        """
        
        if self.sparse_matrix:
            html += f"""
                    <li><strong>Sparse Efficiency:</strong> {self.metadata.get('storage_efficiency', 'Significant space savings')} compared to dense representation</li>
            """
        
        html += """
                </ul>
            </div>
            
            <h3>Usage Instructions</h3>
            <div style="background: #e8f5e8; padding: 20px; border-radius: 5px; border-left: 4px solid #28a745;">
                <p><strong>Loading the Data:</strong></p>
                <pre style="background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;">
import json

# Load the graph data
with open('citation_graph_matrix_YYYYMMDD_HHMMSS.json', 'r') as f:
    data = json.load(f)

# For sparse matrix algorithms
if 'sparse_adjacency_matrix' in data:
    edges = data['sparse_adjacency_matrix']['edges']
    num_nodes = data['sparse_adjacency_matrix']['num_nodes']
    
# For dense matrix algorithms  
if 'adjacency_matrix_dense' in data:
    adjacency_matrix = data['adjacency_matrix_dense']
elif 'adjacency_matrix' in data:
    adjacency_matrix = data['adjacency_matrix']

# Common data
researchers = data['researchers']
nobel_winners = set(data['nobel_winners_ids'])
                </pre>
        """
        
        if self.sparse_matrix:
            html += """
                <p><strong>Converting Between Formats:</strong></p>
                <pre style="background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;">
# Sparse to dense
def sparse_to_dense(edges, num_nodes):
    matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for source, target in edges:
        matrix[source][target] = 1
    return matrix

# Dense to sparse  
def dense_to_sparse(matrix):
    edges = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                edges.append([i, j])
    return edges
                </pre>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def generate_html_report(self, output_filename=None):
        """Generate the complete HTML report"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"citation_graph_report_{timestamp}.html"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Citation Graph ({self.statistics['total_researchers']} Researchers) - Analysis Report</title>
    {self.generate_css()}
</head>
<body>
    {self.generate_header()}
    {self.generate_researcher_section()}
    {self.generate_matrix_section()}
    {self.generate_statistics_section()}
    {self.generate_citation_patterns_section()}
    {self.generate_algorithm_section()}
    
    <footer style="text-align: center; padding: 20px; margin-top: 40px; border-top: 1px solid #ddd; color: #666;">
        <p>Citation Graph Test Dataset Report - Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
        <p>To convert to PDF: Use your browser's Print function and select "Save as PDF"</p>
    </footer>
</body>
</html>
        """
        
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return output_filename
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return None


def main():
    """Main function to generate HTML report"""
    # Find the most recent graph file
    import glob
    
    pattern = "citation_graph_matrix_*.json"
    files = glob.glob(pattern)
    
    if not files:
        print("No citation graph files found. Please run simple_citation_graph_generator.py first.")
        return
    
    # Use the most recent file
    graph_file = max(files, key=os.path.getctime)
    print(f"Using graph file: {graph_file}")
    
    # Generate HTML report
    html_generator = CitationGraphHTMLGenerator(graph_file)
    output_file = html_generator.generate_html_report()
    
    if output_file:
        print(f"\n✅ HTML report generated successfully: {output_file}")
        print("\n📋 The report contains:")
        print("• Title page with overview statistics")
        print("• Complete researcher profiles with Nobel winner highlights")
        print("• Interactive adjacency matrix visualization")
        print("• Comprehensive graph statistics and rankings")
        print("• Citation patterns and research area analysis")
        print("• Algorithm testing guidelines and usage instructions")
        
        print(f"\n🔄 To convert to PDF:")
        print(f"1. Open {output_file} in any web browser")
        print("2. Press Ctrl+P (or Cmd+P on Mac)")
        print("3. Select 'Save as PDF' as the destination")
        print("4. Adjust print settings if needed and save")
        
        print(f"\n📂 Files available:")
        print(f"• HTML Report: {output_file}")
        print(f"• Graph Data: {graph_file}")
        print(f"• CSV Matrix: {graph_file.replace('.json', '.csv')}")
    else:
        print("❌ Failed to generate HTML report.")


if __name__ == "__main__":
    main()
