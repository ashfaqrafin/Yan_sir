"""
Citation Graph PDF Report Generator

This script generates a comprehensive PDF report containing all details of the citation graph,
including the adjacency matrix, researcher profiles, statistics, and analysis results.
"""

import json
from datetime import datetime
import os

# Try to import required libraries, install if needed
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus.flowables import KeepTogether
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ReportLab not found. Installing...")

class CitationGraphPDFGenerator:
    def __init__(self, graph_file):
        """Initialize with citation graph data"""
        with open(graph_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.adjacency_matrix = self.data['adjacency_matrix']
        self.researchers = self.data['researchers']
        self.nobel_winners = set(self.data['nobel_winners_ids'])
        self.statistics = self.data['statistics']
        self.metadata = self.data['metadata']
        self.n = len(self.adjacency_matrix)
        
        # Setup styles
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        self.subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.darkgreen
        )
        
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6
        )
        
        self.code_style = ParagraphStyle(
            'CodeStyle',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Courier',
            spaceAfter=6,
            leftIndent=20
        )
    
    def create_title_page(self):
        """Create the title page"""
        story = []
        
        # Title
        story.append(Paragraph("Citation Graph Test Dataset", self.title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        story.append(Paragraph("Comprehensive Analysis Report", self.styles['Heading2']))
        story.append(Spacer(1, 0.3*inch))
        
        # Generation info
        gen_date = datetime.fromisoformat(self.metadata['generated_date']).strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"Generated: {gen_date}", self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Overview table
        overview_data = [
            ['Graph Property', 'Value'],
            ['Total Researchers', str(self.statistics['total_researchers'])],
            ['Nobel Prize Winners', str(self.statistics['nobel_winners'])],
            ['Total Citations', str(self.statistics['total_citations'])],
            ['Graph Density', f"{self.statistics['graph_density']:.3f}"],
            ['Average Citations per Researcher', f"{self.statistics['average_citations_per_researcher']:.2f}"]
        ]
        
        overview_table = Table(overview_data, colWidths=[3*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
        ]))
        
        story.append(Spacer(1, 0.5*inch))
        story.append(overview_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Description
        description = """
        This report presents a comprehensive analysis of a test citation graph containing 20 researchers,
        including 5 Nobel Prize winners. The graph is represented as a directed adjacency matrix where
        each edge represents a citation relationship. This dataset is designed for testing graph algorithms
        and analyzing citation patterns in academic networks.
        """
        story.append(Paragraph(description, self.styles['Normal']))
        
        story.append(PageBreak())
        return story
    
    def create_researcher_profiles(self):
        """Create researcher profiles section"""
        story = []
        
        story.append(Paragraph("Researcher Profiles", self.heading_style))
        
        # Create table for all researchers
        researcher_data = [['ID', 'Name', 'Nobel Prize', 'Research Area', 'H-Index', 'Citations']]
        
        for researcher in self.researchers:
            nobel_info = ""
            if researcher['is_nobel_winner']:
                nobel_info = f"🏆 {researcher['nobel_category']} {researcher['award_year']}"
            
            researcher_data.append([
                str(researcher['id']),
                researcher['name'],
                nobel_info,
                researcher['research_area'],
                str(researcher['h_index']),
                str(researcher['total_citations'])
            ])
        
        researcher_table = Table(researcher_data, colWidths=[0.5*inch, 2*inch, 1.5*inch, 1.5*inch, 0.8*inch, 1*inch])
        researcher_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        # Highlight Nobel winners
        for i, researcher in enumerate(self.researchers):
            if researcher['is_nobel_winner']:
                researcher_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, i+1), (-1, i+1), colors.gold),
                ]))
        
        story.append(researcher_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Nobel Prize winners details
        story.append(Paragraph("Nobel Prize Winners Details", self.subheading_style))
        
        for nobel_id in sorted(self.nobel_winners):
            researcher = self.researchers[nobel_id]
            citations_received = sum(self.adjacency_matrix[i][nobel_id] for i in range(self.n))
            citations_made = sum(self.adjacency_matrix[nobel_id])
            
            nobel_text = f"""
            <b>{researcher['name']}</b> (ID {nobel_id})<br/>
            🏆 Nobel Prize in {researcher['nobel_category']} ({researcher['award_year']})<br/>
            Research Area: {researcher['research_area']}<br/>
            H-Index: {researcher['h_index']} | Total Citations: {researcher['total_citations']}<br/>
            In Graph - Citations Received: {citations_received} | Citations Made: {citations_made}
            """
            story.append(Paragraph(nobel_text, self.normal_style))
            story.append(Spacer(1, 0.1*inch))
        
        story.append(PageBreak())
        return story
    
    def create_adjacency_matrix_section(self):
        """Create adjacency matrix visualization"""
        story = []
        
        story.append(Paragraph("Adjacency Matrix", self.heading_style))
        
        # Matrix interpretation
        interpretation = """
        The adjacency matrix represents citation relationships where matrix[i][j] = 1 means 
        researcher i cites researcher j. The matrix is 20×20 corresponding to the 20 researchers 
        in the dataset. Nobel Prize winners are highlighted in the matrix.
        """
        story.append(Paragraph(interpretation, self.normal_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Create matrix table
        # Header row with researcher IDs
        matrix_data = [['']]  # Empty top-left cell
        for j in range(self.n):
            if j in self.nobel_winners:
                matrix_data[0].append(f"{j}🏆")
            else:
                matrix_data[0].append(str(j))
        
        # Matrix rows
        for i in range(self.n):
            row = []
            if i in self.nobel_winners:
                row.append(f"{i}🏆")
            else:
                row.append(str(i))
            
            for j in range(self.n):
                if self.adjacency_matrix[i][j] == 1:
                    row.append("●")
                else:
                    row.append("·")
            matrix_data.append(row)
        
        # Create table with smaller font for matrix
        col_widths = [0.4*inch] + [0.25*inch] * self.n
        matrix_table = Table(matrix_data, colWidths=col_widths)
        
        matrix_style = [
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),  # Header row
            ('BACKGROUND', (0, 1), (0, -1), colors.lightblue),  # Header column
        ]
        
        # Highlight Nobel winners in matrix
        for i in range(self.n):
            if i in self.nobel_winners:
                matrix_style.append(('BACKGROUND', (0, i+1), (0, i+1), colors.gold))
                matrix_style.append(('BACKGROUND', (i+1, 0), (i+1, 0), colors.gold))
        
        matrix_table.setStyle(TableStyle(matrix_style))
        
        story.append(matrix_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Legend
        legend_text = """
        <b>Legend:</b><br/>
        ● = Citation exists (researcher in row cites researcher in column)<br/>
        · = No citation<br/>
        🏆 = Nobel Prize winner<br/>
        Numbers represent researcher IDs
        """
        story.append(Paragraph(legend_text, self.normal_style))
        
        story.append(PageBreak())
        return story
    
    def create_statistics_section(self):
        """Create statistics and analysis section"""
        story = []
        
        story.append(Paragraph("Graph Statistics and Analysis", self.heading_style))
        
        # Basic statistics
        story.append(Paragraph("Basic Graph Statistics", self.subheading_style))
        
        basic_stats = f"""
        • Total Researchers: {self.statistics['total_researchers']}<br/>
        • Nobel Prize Winners: {self.statistics['nobel_winners']} ({self.statistics['nobel_winners']/self.statistics['total_researchers']*100:.1f}%)<br/>
        • Total Citation Edges: {self.statistics['total_citations']}<br/>
        • Graph Density: {self.statistics['graph_density']:.3f} ({self.statistics['graph_density']*100:.1f}% of possible edges)<br/>
        • Average Citations per Researcher: {self.statistics['average_citations_per_researcher']:.2f}<br/>
        • Maximum Citations Received: {self.statistics['max_citations_received']}<br/>
        • Maximum Citations Made: {self.statistics['max_citations_made']}
        """
        story.append(Paragraph(basic_stats, self.normal_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Citation distribution
        story.append(Paragraph("Citation Distribution", self.subheading_style))
        
        # Most cited researchers
        citations_received = [(i, self.statistics['in_degrees'][i]) for i in range(self.n)]
        citations_received.sort(key=lambda x: x[1], reverse=True)
        
        most_cited_data = [['Rank', 'Researcher', 'Citations Received', 'Nobel Prize']]
        for rank, (researcher_id, citations) in enumerate(citations_received[:10], 1):
            researcher = self.researchers[researcher_id]
            nobel_status = "🏆 Yes" if researcher_id in self.nobel_winners else "No"
            most_cited_data.append([
                str(rank),
                researcher['name'],
                str(citations),
                nobel_status
            ])
        
        most_cited_table = Table(most_cited_data, colWidths=[0.8*inch, 2.5*inch, 1.2*inch, 1*inch])
        most_cited_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(Paragraph("Top 10 Most Cited Researchers:", self.normal_style))
        story.append(most_cited_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Most active citers
        citations_made = [(i, self.statistics['out_degrees'][i]) for i in range(self.n)]
        citations_made.sort(key=lambda x: x[1], reverse=True)
        
        most_active_data = [['Rank', 'Researcher', 'Citations Made', 'Nobel Prize']]
        for rank, (researcher_id, citations) in enumerate(citations_made[:10], 1):
            researcher = self.researchers[researcher_id]
            nobel_status = "🏆 Yes" if researcher_id in self.nobel_winners else "No"
            most_active_data.append([
                str(rank),
                researcher['name'],
                str(citations),
                nobel_status
            ])
        
        most_active_table = Table(most_active_data, colWidths=[0.8*inch, 2.5*inch, 1.2*inch, 1*inch])
        most_active_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkorange),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(Paragraph("Top 10 Most Active Citers:", self.normal_style))
        story.append(most_active_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Nobel winner analysis
        story.append(Paragraph("Nobel Prize Winner Analysis", self.subheading_style))
        
        nobel_stats = f"""
        • Citations to Nobel Winners: {self.statistics['nobel_winners_citations_received']} ({self.statistics['nobel_winners_citations_received']/self.statistics['total_citations']*100:.1f}% of all citations)<br/>
        • Citations by Nobel Winners: {self.statistics['nobel_winners_citations_made']} ({self.statistics['nobel_winners_citations_made']/self.statistics['total_citations']*100:.1f}% of all citations)<br/>
        • Average Citations Received by Nobel Winners: {self.statistics['nobel_winners_citations_received']/len(self.nobel_winners):.1f}<br/>
        • Average Citations Received by Non-Nobel Winners: {(self.statistics['total_citations'] - self.statistics['nobel_winners_citations_received'])/(self.n - len(self.nobel_winners)):.1f}
        """
        story.append(Paragraph(nobel_stats, self.normal_style))
        
        story.append(PageBreak())
        return story
    
    def create_citation_patterns_section(self):
        """Create citation patterns section"""
        story = []
        
        story.append(Paragraph("Citation Patterns", self.heading_style))
        
        # Find all citations involving Nobel winners
        nobel_citations = []
        for i in range(self.n):
            for j in range(self.n):
                if self.adjacency_matrix[i][j] == 1 and (i in self.nobel_winners or j in self.nobel_winners):
                    citing = self.researchers[i]['name']
                    cited = self.researchers[j]['name']
                    citing_nobel = "🏆" if i in self.nobel_winners else ""
                    cited_nobel = "🏆" if j in self.nobel_winners else ""
                    nobel_citations.append((i, j, citing, cited, citing_nobel, cited_nobel))
        
        story.append(Paragraph("Citations Involving Nobel Prize Winners", self.subheading_style))
        story.append(Paragraph(f"Total: {len(nobel_citations)} citations", self.normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Create citation patterns table
        citation_data = [['Citing Researcher', '', 'Cited Researcher', '']]
        for i, j, citing, cited, citing_nobel, cited_nobel in nobel_citations[:15]:  # Limit to first 15
            citation_data.append([citing, citing_nobel, cited, cited_nobel])
        
        if len(nobel_citations) > 15:
            citation_data.append(['...', '', f'({len(nobel_citations) - 15} more citations)', ''])
        
        citation_table = Table(citation_data, colWidths=[2.5*inch, 0.5*inch, 2.5*inch, 0.5*inch])
        citation_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(citation_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Research area analysis
        story.append(Paragraph("Research Area Distribution", self.subheading_style))
        
        research_areas = {}
        for researcher in self.researchers:
            area = researcher['research_area']
            if area not in research_areas:
                research_areas[area] = {'total': 0, 'nobel': 0}
            research_areas[area]['total'] += 1
            if researcher['is_nobel_winner']:
                research_areas[area]['nobel'] += 1
        
        area_data = [['Research Area', 'Total Researchers', 'Nobel Winners', '% Nobel']]
        for area, counts in sorted(research_areas.items()):
            nobel_pct = (counts['nobel'] / counts['total']) * 100 if counts['total'] > 0 else 0
            area_data.append([
                area,
                str(counts['total']),
                str(counts['nobel']),
                f"{nobel_pct:.1f}%"
            ])
        
        area_table = Table(area_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
        area_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(area_table)
        
        story.append(PageBreak())
        return story
    
    def create_algorithm_testing_section(self):
        """Create algorithm testing guidelines section"""
        story = []
        
        story.append(Paragraph("Algorithm Testing Guidelines", self.heading_style))
        
        # Usage instructions
        usage_text = """
        This citation graph is designed for testing various graph algorithms. The adjacency matrix 
        format makes it suitable for testing centrality measures, path-finding algorithms, 
        community detection, and influence analysis.
        """
        story.append(Paragraph(usage_text, self.normal_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Algorithm categories
        story.append(Paragraph("Recommended Algorithm Categories", self.subheading_style))
        
        algorithms = [
            ("Centrality Measures", "PageRank, Betweenness, Closeness, Eigenvector Centrality"),
            ("Path Finding", "Shortest paths, All-pairs shortest paths, Reachability analysis"),
            ("Community Detection", "Louvain, Leiden, Modularity-based clustering"),
            ("Influence Analysis", "Information propagation, Influence maximization"),
            ("Network Analysis", "Strongly connected components, Topological sorting"),
        ]
        
        algo_data = [['Algorithm Category', 'Specific Algorithms']]
        for category, algorithms_list in algorithms:
            algo_data.append([category, algorithms_list])
        
        algo_table = Table(algo_data, colWidths=[2*inch, 4*inch])
        algo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(algo_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Expected results
        story.append(Paragraph("Expected Test Results", self.subheading_style))
        
        expected_results = """
        Based on the graph structure, algorithms should show:
        • Higher centrality scores for Nobel Prize winners (especially Prof. Jack White, ID 9)
        • Short citation paths between researchers (average path length ~2-3)
        • One main strongly connected component containing most researchers
        • Research area clustering effects in community detection
        • Nobel winner bias in influence propagation models
        """
        story.append(Paragraph(expected_results, self.normal_style))
        
        return story
    
    def generate_pdf(self, output_filename=None):
        """Generate the complete PDF report"""
        if not REPORTLAB_AVAILABLE:
            print("Error: ReportLab library is not available. Please install it using:")
            print("pip install reportlab")
            return False
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"citation_graph_report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(output_filename, pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Build the story
        story = []
        
        # Add all sections
        story.extend(self.create_title_page())
        story.extend(self.create_researcher_profiles())
        story.extend(self.create_adjacency_matrix_section())
        story.extend(self.create_statistics_section())
        story.extend(self.create_citation_patterns_section())
        story.extend(self.create_algorithm_testing_section())
        
        # Build PDF
        try:
            doc.build(story)
            print(f"PDF report generated successfully: {output_filename}")
            return True
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return False


def install_reportlab():
    """Install ReportLab if not available"""
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
        print("ReportLab installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install ReportLab: {e}")
        return False


def main():
    """Main function to generate PDF report"""
    global REPORTLAB_AVAILABLE
    
    # Install ReportLab if not available
    if not REPORTLAB_AVAILABLE:
        print("Installing ReportLab library...")
        if install_reportlab():
            # Try importing again
            try:
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
                from reportlab.platypus.flowables import KeepTogether
                REPORTLAB_AVAILABLE = True
                print("ReportLab successfully imported!")
            except ImportError as e:
                print(f"Still cannot import ReportLab after installation: {e}")
                return
        else:
            print("Cannot proceed without ReportLab. Please install manually:")
            print("pip install reportlab")
            return
    
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
    
    # Generate PDF report
    pdf_generator = CitationGraphPDFGenerator(graph_file)
    success = pdf_generator.generate_pdf()
    
    if success:
        print("\nPDF report generated successfully!")
        print("The report contains:")
        print("• Title page with overview")
        print("• Researcher profiles and Nobel winner details")
        print("• Complete adjacency matrix visualization")
        print("• Graph statistics and analysis")
        print("• Citation patterns and research area distribution")
        print("• Algorithm testing guidelines")
    else:
        print("Failed to generate PDF report.")


if __name__ == "__main__":
    main()
