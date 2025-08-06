"""
JSON to HTML Converter for Researcher Ranking Results

This script converts the researcher ranking results from JSON format
to an interactive HTML representation for better visualization and analysis.

Features:
- Interactive tabs for different ranking methods
- Responsive design for mobile and desktop
- Nobel winner highlighting
- Time-dependent ranking visualization
- Detailed researcher profiles
- Statistical analysis display
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

class ResearcherRankingHTMLGenerator:
    """Generates interactive HTML from researcher ranking JSON results"""
    
    def __init__(self, json_file: str):
        """
        Initialize the HTML generator
        
        Args:
            json_file: Path to the JSON results file
        """
        self.json_file = json_file
        self.data = self._load_json_data()
        
    def _load_json_data(self) -> Dict[str, Any]:
        """Load data from JSON file"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {self.json_file}")
    
    def _generate_css_styles(self) -> str:
        """Generate CSS styles for the HTML page"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 15px;
            font-weight: 300;
        }

        .paper-info {
            background: #3498db;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }

        .paper-info h2 {
            font-size: 1.2em;
            margin-bottom: 10px;
        }

        .equations {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }

        .equation-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
        }

        .parameters {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .param-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }

        .param-value {
            font-size: 1.8em;
            font-weight: bold;
            margin-top: 5px;
        }

        .tabs {
            display: flex;
            justify-content: center;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 5px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }

        .tab {
            background: transparent;
            color: white;
            border: none;
            padding: 12px 24px;
            cursor: pointer;
            border-radius: 8px;
            font-size: 1em;
            transition: all 0.3s ease;
            margin: 2px;
        }

        .tab:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .tab.active {
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .tab-content {
            display: none;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .tab-content.active {
            display: block;
        }

        .ranking-title {
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
        }

        .ranking-table {
            background: #f8f9fa;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }

        .table-header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px;
            font-weight: bold;
            text-align: center;
        }

        .researcher-row {
            display: grid;
            grid-template-columns: 50px 1fr 100px 80px;
            gap: 15px;
            padding: 12px 15px;
            align-items: center;
            border-bottom: 1px solid #e9ecef;
            transition: background 0.2s ease;
        }

        .researcher-row:hover {
            background: #e3f2fd;
        }

        .researcher-row:last-child {
            border-bottom: none;
        }

        .rank {
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            font-size: 1.1em;
        }

        .rank.top-3 {
            background: linear-gradient(135deg, #f39c12, #e67e22);
            color: white;
            border-radius: 50%;
            width: 35px;
            height: 35px;
            line-height: 35px;
            font-size: 1em;
        }

        .researcher-info {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .researcher-name {
            font-weight: 600;
            color: #2c3e50;
            font-size: 1.1em;
        }

        .nobel-badge {
            background: #f39c12;
            color: white;
            font-size: 0.75em;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: bold;
            display: inline-block;
            margin-top: 2px;
        }

        .research-area {
            font-size: 0.85em;
            color: #7f8c8d;
            font-style: italic;
        }

        .score {
            font-weight: bold;
            color: #27ae60;
            text-align: right;
            font-size: 1.1em;
        }

        .h-index {
            color: #7f8c8d;
            text-align: center;
            font-size: 0.9em;
        }

        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }

        .analysis-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #3498db;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .analysis-card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .stat {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #ecf0f1;
        }

        .stat:last-child {
            border-bottom: none;
            margin-bottom: 0;
        }

        .stat-label {
            color: #7f8c8d;
            font-size: 0.95em;
        }

        .stat-value {
            font-weight: bold;
            color: #2c3e50;
        }

        .time-period-selector {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .period-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9em;
        }

        .period-btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }

        .period-btn.active {
            background: #e74c3c;
            transform: scale(1.05);
        }

        .researcher-details {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .researcher-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }

        .researcher-card:hover {
            transform: translateY(-5px);
        }

        .researcher-card.nobel {
            border-left: 5px solid #f39c12;
        }

        .researcher-card h4 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            font-size: 0.9em;
        }

        .info-item {
            display: flex;
            justify-content: space-between;
            padding: 2px 0;
        }

        .comparison-section {
            margin-top: 30px;
            padding: 20px;
            background: #f1f2f6;
            border-radius: 10px;
        }

        .summary-stats {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }

        .summary-stat {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 10px;
            min-width: 120px;
        }

        .summary-stat .number {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }

        .summary-stat .label {
            font-size: 0.9em;
            color: #7f8c8d;
        }

        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header h1 { font-size: 1.8em; }
            .researcher-row {
                grid-template-columns: 40px 1fr;
                gap: 10px;
            }
            .score, .h-index {
                grid-column: span 2;
                text-align: left;
                margin-top: 5px;
            }
            .parameters {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        .highlight {
            background: linear-gradient(135deg, #ff6b6b, #ee5a52) !important;
            color: white !important;
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 30px;
        }
        """
    
    def _generate_javascript(self) -> str:
        """Generate JavaScript for interactive functionality"""
        return """
        // Current time period for temporal rankings
        let currentTimePeriod = 0;

        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {
            showTab('basic');
        });

        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });

            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });

            // Show selected tab content
            document.getElementById(tabName).classList.add('active');

            // Add active class to clicked tab
            document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
        }

        function showTimePeriod(period) {
            currentTimePeriod = period;
            
            // Update button states
            document.querySelectorAll('.period-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.period-btn')[period].classList.add('active');

            const periodNames = ['Early Career', 'Mid Career', 'Late Career'];
            document.getElementById('time-header').textContent = `Period ${period} - ${periodNames[period]} Rankings`;

            // Hide all period tables and show selected one
            document.querySelectorAll('.time-period-table').forEach(table => {
                table.style.display = 'none';
            });
            document.getElementById(`period-${period}`).style.display = 'block';
        }

        // Add click highlighting functionality
        document.addEventListener('click', function(e) {
            if (e.target.closest('.researcher-row')) {
                e.target.closest('.researcher-row').classList.toggle('highlight');
            }
        });
        """
    
    def _create_ranking_table(self, rankings: Dict[str, float], title: str, table_id: str = "") -> str:
        """Create HTML table for rankings"""
        # Sort rankings by score (descending)
        sorted_rankings = sorted(
            [(int(rid), score) for rid, score in rankings.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        html = f"""
        <div class="ranking-table" {f'id="{table_id}"' if table_id else ''}>
            <div class="table-header">{title}</div>
        """
        
        for i, (researcher_id, score) in enumerate(sorted_rankings[:15], 1):
            researcher = self.data['researchers'][str(researcher_id)]
            
            rank_class = "top-3" if i <= 3 else ""
            nobel_badge = ""
            if researcher['is_nobel_winner']:
                nobel_badge = f'<div class="nobel-badge">{researcher["nobel_category"]} {researcher["award_year"]}</div>'
            
            html += f"""
            <div class="researcher-row">
                <div class="rank {rank_class}">{i}</div>
                <div class="researcher-info">
                    <div class="researcher-name">
                        {'🏆 ' if researcher['is_nobel_winner'] else ''}{researcher['name']}
                    </div>
                    {nobel_badge}
                    <div class="research-area">{researcher['research_area']}</div>
                </div>
                <div class="score">{score:.2f}</div>
                <div class="h-index">H: {researcher['h_index']}</div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_header(self) -> str:
        """Generate the header section"""
        algorithm_info = self.data.get('algorithm_info', {})
        dataset_info = self.data.get('dataset_info', {})
        params = algorithm_info.get('parameters', {})
        
        equations_html = ""
        for eq in algorithm_info.get('equations_implemented', []):
            equations_html += f'<span class="equation-badge">{eq}</span>'
        
        return f"""
        <div class="header">
            <h1>🏆 Researcher Collaboration Network Ranking</h1>
            
            <div class="paper-info">
                <h2>📄 "{algorithm_info.get('paper_title', 'An Algorithm for Ranking Authors')}"</h2>
                <div class="equations">
                    {equations_html}
                </div>
                
                <div class="parameters">
                    <div class="param-card">
                        <div>Citation Weight (α)</div>
                        <div class="param-value">{params.get('alpha_citation_weight', 0.6)}</div>
                    </div>
                    <div class="param-card">
                        <div>Collaboration Weight (β)</div>
                        <div class="param-value">{params.get('beta_collaboration_weight', 0.4)}</div>
                    </div>
                    <div class="param-card">
                        <div>Total Researchers</div>
                        <div class="param-value">{dataset_info.get('num_researchers', 0)}</div>
                    </div>
                    <div class="param-card">
                        <div>Nobel Winners</div>
                        <div class="param-value">{dataset_info.get('num_awards', 0)}</div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_basic_ranking_tab(self) -> str:
        """Generate the basic ranking tab content"""
        basic_rankings = self.data['rankings']['basic_ranking']
        
        return f"""
        <div id="basic" class="tab-content active">
            <h2 class="ranking-title">📊 Basic Ranking (Equation 1)</h2>
            <p style="margin-bottom: 20px; color: #7f8c8d;">
                Basic ranking using awards as dummy researchers: <strong>rank<sub>σ</sub> = w<sub>σ</sub> = Σ(w<sub>j</sub> × c<sub>ρj</sub> / c<sub>ρ</sub>)</strong>
            </p>
            
            {self._create_ranking_table(basic_rankings, "Top Researchers - Basic Ranking")}
        </div>
        """
    
    def _generate_collaboration_ranking_tab(self) -> str:
        """Generate the collaboration ranking tab content"""
        collab_rankings = self.data['rankings']['collaboration_ranking']
        
        return f"""
        <div id="collaboration" class="tab-content">
            <h2 class="ranking-title">🤝 Collaboration Ranking (Equation 5)</h2>
            <p style="margin-bottom: 20px; color: #7f8c8d;">
                Combined citation and collaboration ranking: <strong>rank<sub>i</sub> = ΣΣ w<sub>j</sub> (α × c<sub>j,τ</sub>/c<sub>jτ</sub> + β × κ<sub>j,τ</sub>/κ<sub>jτ</sub>)</strong>
            </p>
            
            {self._create_ranking_table(collab_rankings, "Top Researchers - Collaboration Ranking")}
        </div>
        """
    
    def _generate_temporal_ranking_tab(self) -> str:
        """Generate the time-dependent ranking tab content"""
        time_rankings = self.data['rankings'].get('time_dependent_ranking', {})
        
        if not time_rankings:
            return """
            <div id="temporal" class="tab-content">
                <h2 class="ranking-title">⏰ Time-Dependent Ranking</h2>
                <p>No time-dependent ranking data available.</p>
            </div>
            """
        
        html = """
        <div id="temporal" class="tab-content">
            <h2 class="ranking-title">⏰ Time-Dependent Ranking (Equation 6)</h2>
            <p style="margin-bottom: 20px; color: #7f8c8d;">
                Time-dependent ranking with evolving researcher influence: <strong>rank<sub>i</sub> = w<sub>ti</sub> = ΣΣ w<sub>j,τ</sub> (α × c<sub>j,τ</sub>/c<sub>jτ</sub> + β × κ<sub>j,τ</sub>/κ<sub>jτ</sub>)</strong>
            </p>
            
            <div class="time-period-selector">
                <button class="period-btn active" onclick="showTimePeriod(0)">Period 0 (Early Career)</button>
                <button class="period-btn" onclick="showTimePeriod(1)">Period 1 (Mid Career)</button>
                <button class="period-btn" onclick="showTimePeriod(2)">Period 2 (Late Career)</button>
            </div>
            
            <div class="table-header" id="time-header">Period 0 - Early Career Rankings</div>
        """
        
        period_names = ['Early Career', 'Mid Career', 'Late Career']
        for i, (period_key, rankings) in enumerate(time_rankings.items()):
            display_style = 'block' if i == 0 else 'none'
            period_name = period_names[i] if i < len(period_names) else f'Period {i}'
            
            html += f"""
            <div class="time-period-table" id="period-{i}" style="display: {display_style};">
                {self._create_ranking_table(rankings, f"Period {i} - {period_name} Rankings", f"period-{i}-table")}
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_analysis_tab(self) -> str:
        """Generate the analysis tab content"""
        basic_analysis = self.data['analysis']['basic_ranking']
        collab_analysis = self.data['analysis']['collaboration_ranking']
        dataset_info = self.data['dataset_info']
        
        # Calculate additional statistics
        nobel_winners = [r for r in self.data['researchers'].values() if r['is_nobel_winner']]
        avg_h_index_nobel = sum(r['h_index'] for r in nobel_winners) / len(nobel_winners) if nobel_winners else 0
        
        # Count Nobel winners by category
        nobel_categories = {}
        for winner in nobel_winners:
            category = winner['nobel_category']
            year = winner['award_year']
            key = f"{category} ({year})"
            nobel_categories[key] = nobel_categories.get(key, 0) + 1
        
        return f"""
        <div id="analysis" class="tab-content">
            <h2 class="ranking-title">📈 Ranking Analysis & Comparison</h2>
            
            <div class="summary-stats">
                <div class="summary-stat">
                    <div class="number">{basic_analysis['total_researchers']}</div>
                    <div class="label">Total Researchers</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{basic_analysis['nobel_winners_total']}</div>
                    <div class="label">Nobel Winners</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{dataset_info.get('num_awards', 0)}</div>
                    <div class="label">Award Categories</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{avg_h_index_nobel:.1f}</div>
                    <div class="label">Avg Nobel H-Index</div>
                </div>
            </div>
            
            <div class="analysis-grid">
                <div class="analysis-card">
                    <h3>🏆 Basic Ranking Statistics</h3>
                    <div class="stat">
                        <span class="stat-label">Nobel Winners in Top 10</span>
                        <span class="stat-value">{basic_analysis['nobel_in_top_10']}/{basic_analysis['nobel_winners_total']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Score Range</span>
                        <span class="stat-value">{basic_analysis['score_range'][0]:.1f} - {basic_analysis['score_range'][1]:.1f}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Mean Score</span>
                        <span class="stat-value">{basic_analysis['mean_score']:.2f}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Algorithm</span>
                        <span class="stat-value">Equation 1</span>
                    </div>
                </div>

                <div class="analysis-card">
                    <h3>🤝 Collaboration Ranking Statistics</h3>
                    <div class="stat">
                        <span class="stat-label">Nobel Winners in Top 10</span>
                        <span class="stat-value">{collab_analysis['nobel_in_top_10']}/{collab_analysis['nobel_winners_total']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Score Range</span>
                        <span class="stat-value">{collab_analysis['score_range'][0]:.1f} - {collab_analysis['score_range'][1]:.1f}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Mean Score</span>
                        <span class="stat-value">{collab_analysis['mean_score']:.2f}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Algorithm</span>
                        <span class="stat-value">Equation 5</span>
                    </div>
                </div>

                <div class="analysis-card">
                    <h3>🔬 Dataset Information</h3>
                    <div class="stat">
                        <span class="stat-label">Citation Matrix Size</span>
                        <span class="stat-value">{dataset_info.get('citation_matrix_size', 'N/A')}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Extended Matrix Size</span>
                        <span class="stat-value">{dataset_info.get('extended_matrix_size', 'N/A')}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Algorithm Complexity</span>
                        <span class="stat-value">O(N²)</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Matrix Type</span>
                        <span class="stat-value">Dense + Awards</span>
                    </div>
                </div>

                <div class="analysis-card">
                    <h3>🏅 Nobel Prize Distribution</h3>
                    {''.join(f'<div class="stat"><span class="stat-label">{category}</span><span class="stat-value">{count}</span></div>' for category, count in nobel_categories.items())}
                </div>
            </div>
            
            <div class="comparison-section">
                <h3>📊 Ranking Method Comparison</h3>
                <p style="color: #7f8c8d; margin-bottom: 15px;">
                    Comparison of how different ranking methods perform in identifying top researchers:
                </p>
                <div class="info-grid">
                    <div class="info-item">
                        <span>Basic Ranking Nobel Coverage:</span>
                        <strong>{(basic_analysis['nobel_in_top_10']/basic_analysis['nobel_winners_total']*100):.1f}%</strong>
                    </div>
                    <div class="info-item">
                        <span>Collaboration Ranking Nobel Coverage:</span>
                        <strong>{(collab_analysis['nobel_in_top_10']/collab_analysis['nobel_winners_total']*100):.1f}%</strong>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_researchers_tab(self) -> str:
        """Generate the researchers tab content"""
        researchers = list(self.data['researchers'].values())
        # Sort by h-index descending
        researchers.sort(key=lambda r: r['h_index'], reverse=True)
        
        html = """
        <div id="researchers" class="tab-content">
            <h2 class="ranking-title">👨‍🔬 Researcher Profiles</h2>
            <p style="margin-bottom: 20px; color: #7f8c8d;">
                Detailed information about all researchers in the network
            </p>
            
            <div class="researcher-details">
        """
        
        for researcher in researchers:
            nobel_class = "nobel" if researcher['is_nobel_winner'] else ""
            nobel_badge = ""
            if researcher['is_nobel_winner']:
                nobel_badge = f'<div class="nobel-badge">{researcher["nobel_category"]} Nobel {researcher["award_year"]}</div>'
            
            html += f"""
            <div class="researcher-card {nobel_class}">
                <div class="researcher-info">
                    <h4>{'🏆 ' if researcher['is_nobel_winner'] else ''}{researcher['name']}</h4>
                    {nobel_badge}
                </div>
                <div class="info-grid">
                    <div class="info-item">
                        <span>Research Area:</span>
                        <strong>{researcher['research_area']}</strong>
                    </div>
                    <div class="info-item">
                        <span>H-Index:</span>
                        <strong>{researcher['h_index']}</strong>
                    </div>
                    <div class="info-item">
                        <span>Total Citations:</span>
                        <strong>{researcher['total_citations']:,}</strong>
                    </div>
                    <div class="info-item">
                        <span>Researcher ID:</span>
                        <strong>#{researcher['id']}</strong>
                    </div>
                </div>
            </div>
            """
        
        html += """
            </div>
        </div>
        """
        return html
    
    def generate_html(self, output_file: str = None) -> str:
        """
        Generate complete HTML representation
        
        Args:
            output_file: Optional output file path. If not provided, uses input filename with .html extension
            
        Returns:
            str: Complete HTML content
        """
        if output_file is None:
            base_name = os.path.splitext(self.json_file)[0]
            output_file = f"{base_name}.html"
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Researcher Collaboration Network Ranking Results</title>
    <style>
        {self._generate_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header()}

        <div class="tabs">
            <button class="tab active" onclick="showTab('basic')">📊 Basic Ranking</button>
            <button class="tab" onclick="showTab('collaboration')">🤝 Collaboration Ranking</button>
            <button class="tab" onclick="showTab('temporal')">⏰ Time-Dependent</button>
            <button class="tab" onclick="showTab('analysis')">📈 Analysis</button>
            <button class="tab" onclick="showTab('researchers')">👨‍🔬 Researchers</button>
        </div>

        {self._generate_basic_ranking_tab()}
        {self._generate_collaboration_ranking_tab()}
        {self._generate_temporal_ranking_tab()}
        {self._generate_analysis_tab()}
        {self._generate_researchers_tab()}
        
        <div class="footer">
            <p>📊 Generated from: {os.path.basename(self.json_file)} | 🕒 {current_time}</p>
            <p>🔬 Algorithm: "An Algorithm for Ranking Authors" | 💻 Interactive HTML Report</p>
        </div>
    </div>

    <script>
        {self._generate_javascript()}
    </script>
</body>
</html>"""
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report generated successfully: {output_file}")
        print(f"📊 Report includes:")
        print(f"   • Basic Ranking (Equation 1)")
        print(f"   • Collaboration Ranking (Equation 5)")
        print(f"   • Time-Dependent Ranking (Equation 6)")
        print(f"   • Statistical Analysis")
        print(f"   • Researcher Profiles")
        print(f"   • Interactive Tabs and Filtering")
        
        return html_content

def main():
    """Main function to demonstrate usage"""
    import sys
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python json_to_html_converter.py <json_file> [output_file]")
        print("\nExample:")
        print("python json_to_html_converter.py researcher_ranking_results.json")
        print("python json_to_html_converter.py results.json custom_output.html")
        return
    
    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Create HTML generator
        generator = ResearcherRankingHTMLGenerator(json_file)
        
        # Generate HTML
        html_content = generator.generate_html(output_file)
        
        print(f"\n🌐 Open the HTML file in your browser to view the interactive report:")
        final_output = output_file if output_file else f"{os.path.splitext(json_file)[0]}.html"
        print(f"   file://{os.path.abspath(final_output)}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    # For direct usage with the provided JSON file
    json_file = "researcher_ranking_results.json"
    
    if os.path.exists(json_file):
        print(f"🔄 Converting {json_file} to HTML...")
        generator = ResearcherRankingHTMLGenerator(json_file)
        generator.generate_html()
    else:
        print(f"📁 Running in command-line mode...")
        main()
