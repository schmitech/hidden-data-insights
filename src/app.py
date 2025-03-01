import os
import json
import pandas as pd
import numpy as np
from io import StringIO
from dotenv import load_dotenv
import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import re

from utils.data_generator import DataGenerator
from models.llm_analyzer import LLMAnalyzer
from visualization.visualizer import DataVisualizer

# Load environment variables
load_dotenv()

# Initialize components
data_generator = DataGenerator()
visualizer = DataVisualizer()

# Check if API key is available
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    llm_analyzer = LLMAnalyzer(api_key=api_key)
else:
    llm_analyzer = None

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)

server = app.server
app.title = "Hidden Data Insights"

# Helper function to convert text to markdown
def format_text_as_markdown(text):
    """
    Format text with markdown enhancements:
    - Convert numbered lists to proper markdown
    - Add bold to key terms
    - Format section headers
    """
    if not text:
        return text
    
    # Convert numbered lists (e.g., "1. Item" to proper markdown)
    # This regex ensures there's a space after the period in numbered lists
    text = re.sub(r'(\d+)\.(?!\s)', r'\1. ', text)
    
    # Add bold to key terms like "correlation", "pattern", etc.
    # But only when they appear as standalone concepts, not as part of every sentence
    key_terms = [
        r'\bkey correlation\b', r'\bstrong pattern\b', r'\bsignificant relationship\b', 
        r'\bimportant trend\b', r'\bcritical insight\b', r'\bhighly significant\b', 
        r'\bkey finding\b'
    ]
    
    for term in key_terms:
        text = re.sub(term, f'**{term.replace(r"\b", "")}**', text, flags=re.IGNORECASE)
    
    # Format potential section headers
    text = re.sub(r'^([A-Z][A-Za-z\s]+:)', r'### \1', text, flags=re.MULTILINE)
    
    # Ensure proper list formatting
    # Convert dash or bullet lists to proper markdown lists
    text = re.sub(r'^\s*[-•]\s+', '- ', text, flags=re.MULTILINE)
    
    # Ensure proper paragraph breaks
    # Add double line breaks between paragraphs if they don't already exist
    text = re.sub(r'(\n)(?!\n)', r'\n\n', text)
    
    return text

# Helper function to ensure proper markdown list formatting
def format_list_items(items):
    """
    Format a list of items as a proper markdown list with appropriate spacing.
    Each item will be on its own line with proper indentation.
    """
    if not items:
        return ""
    
    # If we have a single item that contains multiple points, split it
    if len(items) == 1 and len(items[0]) > 200:  # Long text likely containing multiple points
        text = items[0]
        
        # Try to split on common patterns
        split_items = []
        
        # First check if there are numbered points (e.g., "1.", "2.", etc.)
        numbered_pattern = re.compile(r'(\d+\.\s+)')
        numbered_splits = numbered_pattern.split(text)
        
        if len(numbered_splits) > 1:
            # We have numbered points
            current_item = ""
            for i, part in enumerate(numbered_splits):
                if numbered_pattern.match(part):
                    # This is a number marker
                    if current_item:
                        split_items.append(current_item.strip())
                    current_item = part
                else:
                    current_item += part
            if current_item:
                split_items.append(current_item.strip())
        else:
            # Try to split on bullet points or dashes
            bullet_pattern = re.compile(r'(\s*[-•]\s+)')
            bullet_splits = bullet_pattern.split(text)
            
            if len(bullet_splits) > 1:
                # We have bullet points
                current_item = ""
                for i, part in enumerate(bullet_splits):
                    if bullet_pattern.match(part):
                        # This is a bullet marker
                        if current_item:
                            split_items.append(current_item.strip())
                        current_item = part
                    else:
                        current_item += part
                if current_item:
                    split_items.append(current_item.strip())
            else:
                # Try to split on section headers
                section_pattern = re.compile(r'(\d+\.\s+[A-Z][A-Za-z\s]+:)')
                section_splits = section_pattern.split(text)
                
                if len(section_splits) > 1:
                    # We have section headers
                    current_item = ""
                    for i, part in enumerate(section_splits):
                        if section_pattern.match(part):
                            # This is a section header
                            if current_item:
                                split_items.append(current_item.strip())
                            current_item = part
                        else:
                            current_item += part
                    if current_item:
                        split_items.append(current_item.strip())
                else:
                    # Last resort: split on double newlines or periods followed by space
                    split_items = re.split(r'\n\n+|\.\s+', text)
        
        # If we successfully split the text, use those items
        if split_items:
            items = split_items
    
    # Format each item and ensure it starts with a markdown list marker
    formatted_items = []
    for item in items:
        # Skip empty items
        if not item.strip():
            continue
            
        # Format the item text but don't apply the general markdown formatting
        # that might over-bold everything
        formatted_item = item.strip()
        
        # Only apply specific formatting for headers
        formatted_item = re.sub(r'^([A-Z][A-Za-z\s]+:)', r'### \1', formatted_item, flags=re.MULTILINE)
        
        # Ensure it starts with a list marker if it doesn't already
        if not formatted_item.strip().startswith('-') and not re.match(r'^\d+\.', formatted_item.strip()):
            formatted_item = f"- {formatted_item}"
            
        # Add to the list
        formatted_items.append(formatted_item)
    
    # Join with double newlines to ensure proper spacing between list items
    return "\n\n".join(formatted_items)

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Hidden Data Insights", className="display-4 text-primary mb-4"),
            html.P(
                "Discover hidden patterns in your data using OpenAI's GPT-3.5 model. "
                "No data analysts required!",
                className="lead"
            ),
        ], width=12)
    ], className="mt-4 mb-5"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("1. Select Dataset Type", className="text-primary")),
                dbc.CardBody([
                    dbc.RadioItems(
                        id="dataset-type",
                        options=[
                            {"label": "E-commerce Data", "value": "ecommerce"},
                            {"label": "Financial Data", "value": "financial"},
                            {"label": "Healthcare Data", "value": "healthcare"},
                            {"label": "Upload Custom Data (CSV)", "value": "custom"}
                        ],
                        value="ecommerce",
                        inline=True,
                        className="mb-3"
                    ),
                    dbc.Collapse(
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select a CSV File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            }
                        ),
                        id="upload-collapse",
                        is_open=False,
                    ),
                    html.Div(id="dataset-info", className="mt-3"),
                    dbc.Button(
                        "Generate Dataset",
                        id="generate-btn",
                        color="primary",
                        className="mt-3"
                    )
                ])
            ], className="mb-4"),
            
            # Data Preview Card
            dbc.Card([
                dbc.CardHeader(html.H4("Data Preview", className="text-primary")),
                dbc.CardBody([
                    dbc.Spinner(id="data-preview-spinner", children=[
                        html.Div(id="data-preview-content")
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Download CSV",
                                id="download-btn",
                                color="secondary",
                                className="mt-3",
                                disabled=True
                            ),
                            dcc.Download(id="download-dataframe-csv"),
                        ]),
                        dbc.Col([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Rows to display:", className="me-2"),
                                ], width="auto"),
                                dbc.Col([
                                    dbc.Select(
                                        id="rows-to-display",
                                        options=[
                                            {"label": "10 rows", "value": "10"},
                                            {"label": "25 rows", "value": "25"},
                                            {"label": "50 rows", "value": "50"},
                                            {"label": "100 rows", "value": "100"}
                                        ],
                                        value="10",
                                    )
                                ])
                            ], className="d-flex align-items-center")
                        ], width=3)
                    ], className="mt-3")
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader(html.H4("2. Analyze Data", className="text-primary")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Specific Questions (Optional)"),
                            dbc.Textarea(
                                id="specific-questions",
                                placeholder="Enter specific questions about the data, one per line...",
                                style={"height": "100px"}
                            )
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Analyze with GPT-3.5",
                                id="analyze-btn",
                                color="success",
                                className="mt-3",
                                disabled=not bool(api_key)
                            ),
                        ]),
                        dbc.Col([
                            html.Div([
                                dbc.Spinner(color="success", size="lg", spinner_style={"width": "3rem", "height": "3rem"}),
                                html.Span("Analyzing data...", className="ms-2")
                            ], id="analysis-loading-indicator", style={"display": "none"}, className="mt-3")
                        ])
                    ]),
                    html.Div(
                        "OpenAI API key not found. Please add it to your .env file.",
                        id="api-key-warning",
                        className="text-danger mt-2",
                        style={"display": "none" if api_key else "block"}
                    )
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader(html.H4("3. Results", className="text-primary")),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(
                            dbc.Spinner(html.Div(id="summary-tab-content")),
                            label="Summary",
                            tab_id="summary-tab"
                        ),
                        dbc.Tab(
                            dbc.Spinner(html.Div(id="specific-questions-tab-content")),
                            label="Specific Questions",
                            tab_id="specific-questions-tab"
                        ),
                        dbc.Tab(
                            dbc.Spinner(html.Div(id="patterns-tab-content")),
                            label="Hidden Patterns",
                            tab_id="patterns-tab"
                        ),
                        dbc.Tab(
                            dbc.Spinner(html.Div(id="visualizations-tab-content")),
                            label="Visualizations",
                            tab_id="visualizations-tab"
                        ),
                        dbc.Tab(
                            dbc.Spinner(html.Div(id="raw-tab-content")),
                            label="Raw Analysis",
                            tab_id="raw-tab"
                        ),
                    ], id="result-tabs", active_tab="summary-tab")
                ])
            ])
        ], width=12)
    ]),
    
    # Store components for the data
    dcc.Store(id="dataset-store"),
    dcc.Store(id="analysis-store"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(
                "Hidden Data Insights - Powered by OpenAI GPT-3.5",
                className="text-center text-muted"
            )
        ], width=12)
    ], className="mt-5")
    
], fluid=True)

# Callbacks
@app.callback(
    Output("upload-collapse", "is_open"),
    Input("dataset-type", "value")
)
def toggle_upload_collapse(dataset_type):
    return dataset_type == "custom"

@app.callback(
    [
        Output("dataset-store", "data"),
        Output("dataset-info", "children"),
        Output("download-btn", "disabled")
    ],
    [
        Input("generate-btn", "n_clicks"),
        Input("upload-data", "contents")
    ],
    [
        State("dataset-type", "value"),
        State("upload-data", "filename")
    ],
    prevent_initial_call=True
)
def generate_or_upload_dataset(n_clicks, contents, dataset_type, filename):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    df = None
    info_text = ""
    
    if trigger_id == "generate-btn" and dataset_type != "custom":
        # Generate synthetic dataset
        if dataset_type == "ecommerce":
            df = data_generator.generate_ecommerce_data(num_records=1000)
            info_text = "Generated E-commerce dataset with 1000 records"
        elif dataset_type == "financial":
            df = data_generator.generate_financial_data(num_records=1000)
            info_text = "Generated Financial dataset with 1000 records"
        elif dataset_type == "healthcare":
            df = data_generator.generate_healthcare_data(num_records=1000)
            info_text = "Generated Healthcare dataset with 1000 records"
    
    elif trigger_id == "upload-data" and contents:
        # Parse uploaded file
        import base64
        import io
        
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
            
            info_text = f"Uploaded {filename} with {len(df)} records and {len(df.columns)} columns"
        except Exception as e:
            info_text = f"Error processing file: {str(e)}"
    
    if df is not None:
        # Convert datetime columns to strings for JSON serialization
        for col in df.columns:
            if pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        # Create dataset info card
        info_card = dbc.Card([
            dbc.CardBody([
                html.H5("Dataset Information", className="card-title"),
                html.P(f"Records: {len(df)}", className="card-text"),
                html.P(f"Columns: {len(df.columns)}", className="card-text"),
                html.P(info_text, className="card-text text-success")
            ])
        ])
        
        return df.to_json(date_format='iso', orient='split'), info_card, False
    
    return dash.no_update, html.Div(info_text, className="text-danger"), True

@app.callback(
    Output("data-preview-content", "children"),
    [Input("dataset-store", "data"), Input("rows-to-display", "value")],
    prevent_initial_call=True
)
def update_data_preview(dataset_json, rows_to_display):
    if not dataset_json:
        return html.Div("No data available. Please generate or upload a dataset.")
    
    # Convert JSON back to DataFrame
    df = pd.read_json(StringIO(dataset_json), orient='split')
    
    # Limit to specified number of rows
    rows = int(rows_to_display) if rows_to_display else 10
    preview_df = df.head(rows)
    
    # Create data table
    table = dash_table.DataTable(
        data=preview_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in preview_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'padding': '10px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'padding': '10px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        page_size=rows,
        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in preview_df.to_dict('records')
        ],
        tooltip_duration=None,
        sort_action='native',
        filter_action='native',
    )
    
    return [
        html.Div([
            html.P(f"Showing {len(preview_df)} of {len(df)} rows", className="text-muted"),
            table
        ])
    ]

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-btn", "n_clicks"),
    State("dataset-store", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, dataset_json):
    if not dataset_json:
        return dash.no_update
    
    # Convert JSON back to DataFrame
    df = pd.read_json(StringIO(dataset_json), orient='split')
    
    return dcc.send_data_frame(df.to_csv, "hidden_data_insights_dataset.csv", index=False)

@app.callback(
    [Output("analysis-store", "data"), Output("analysis-loading-indicator", "children")],
    Input("analyze-btn", "n_clicks"),
    [
        State("dataset-store", "data"),
        State("dataset-type", "value"),
        State("specific-questions", "value")
    ],
    prevent_initial_call=True
)
def analyze_dataset(n_clicks, dataset_json, dataset_type, specific_questions):
    if not dataset_json or not llm_analyzer:
        return None, html.Div()
    
    # Convert JSON back to DataFrame
    df = pd.read_json(StringIO(dataset_json), orient='split')
    
    # Parse specific questions if provided
    questions_list = None
    if specific_questions:
        questions_list = [q.strip() for q in specific_questions.split('\n') if q.strip()]
    
    # Run analysis
    analysis_results = llm_analyzer.analyze_dataset(
        df, 
        domain=dataset_type if dataset_type != "custom" else None,
        specific_questions=questions_list
    )
    
    # Return both the analysis results and a success message
    return analysis_results, html.Div("Analysis complete!", className="text-success")

# Add a client-side callback to show the spinner immediately when the button is clicked
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            return [
                {"display": "none"}, 
                {"display": "block"}
            ];
        }
        return [
            {"display": "block"}, 
            {"display": "none"}
        ];
    }
    """,
    [
        Output("analyze-btn", "style"),
        Output("analysis-loading-indicator", "style")
    ],
    Input("analyze-btn", "n_clicks"),
    prevent_initial_call=True
)

@app.callback(
    Output("summary-tab-content", "children"),
    Input("analysis-store", "data"),
    prevent_initial_call=True
)
def update_summary_tab(analysis_results):
    if not analysis_results:
        return html.Div("No analysis results available. Please analyze the data first.")
    
    summary = analysis_results.get("summary", "No summary available.")
    
    # Apply the markdown formatting function
    formatted_summary = format_text_as_markdown(summary)
    
    return dbc.Card([
        dbc.CardBody([
            html.H4("Data Analysis Summary", className="card-title"),
            dcc.Markdown(
                formatted_summary,
                className="card-text",
                style={"padding": "10px", "white-space": "pre-wrap"},
                dangerously_allow_html=True
            )
        ])
    ])

@app.callback(
    Output("specific-questions-tab-content", "children"),
    [Input("analysis-store", "data"), Input("specific-questions", "value")],
    prevent_initial_call=True
)
def update_specific_questions_tab(analysis_results, specific_questions_text):
    if not analysis_results:
        return html.Div("No analysis results available. Please analyze the data first.")
    
    specific_answers = analysis_results.get("specific_answers", [])
    
    if not specific_answers:
        if not specific_questions_text:
            return html.Div("No specific questions were asked. Enter questions in the 'Analyze Data' section.")
        else:
            return html.Div("No specific answers were found in the analysis. Try rephrasing your questions.")
    
    # Create a card for each question and answer
    answer_cards = []
    
    for i, answer in enumerate(specific_answers):
        # Split into question and answer parts
        parts = answer.split('\nA: ')
        if len(parts) == 2:
            question = parts[0].replace('Q: ', '')
            answer_text = parts[1]
            
            # For concise display, don't apply full markdown formatting
            # Just ensure basic formatting is preserved
            
            answer_cards.append(
                dbc.Card([
                    dbc.CardHeader(html.H5(f"Question {i+1}: {question}", className="text-primary")),
                    dbc.CardBody([
                        html.Div([
                            html.Strong("Answer: "),
                            html.Span(answer_text)
                        ], style={"fontSize": "1.1rem"})
                    ])
                ], className="mb-3")
            )
    
    if not answer_cards:
        return html.Div("No specific answers were found in the analysis. Try rephrasing your questions.")
    
    return html.Div([
        html.H4("Answers to Your Specific Questions", className="mb-4"),
        html.Div(answer_cards)
    ])

@app.callback(
    Output("patterns-tab-content", "children"),
    Input("analysis-store", "data"),
    prevent_initial_call=True
)
def update_patterns_tab(analysis_results):
    if not analysis_results:
        return html.Div("No analysis results available. Please analyze the data first.")
    
    hidden_patterns = analysis_results.get("hidden_patterns", [])
    unusual_correlations = analysis_results.get("unusual_correlations", [])
    causal_relationships = analysis_results.get("causal_relationships", [])
    recommendations = analysis_results.get("recommendations", [])
    
    pattern_cards = []
    
    # Hidden Patterns
    if hidden_patterns:
        # Format patterns with proper spacing
        pattern_markdown = format_list_items(hidden_patterns)
        
        pattern_cards.append(
            dbc.Card([
                dbc.CardHeader(html.H5("Hidden Patterns", className="text-primary")),
                dbc.CardBody([
                    dcc.Markdown(
                        pattern_markdown, 
                        style={"padding": "10px", "white-space": "pre-wrap"},
                        dangerously_allow_html=True
                    )
                ])
            ], className="mb-3")
        )
    
    # Unusual Correlations
    if unusual_correlations:
        # Format correlations with proper spacing
        correlation_markdown = format_list_items(unusual_correlations)
        
        pattern_cards.append(
            dbc.Card([
                dbc.CardHeader(html.H5("Unusual Correlations", className="text-primary")),
                dbc.CardBody([
                    dcc.Markdown(
                        correlation_markdown, 
                        style={"padding": "10px", "white-space": "pre-wrap"},
                        dangerously_allow_html=True
                    )
                ])
            ], className="mb-3")
        )
    
    # Causal Relationships
    if causal_relationships:
        # Format relationships with proper spacing
        relationship_markdown = format_list_items(causal_relationships)
        
        pattern_cards.append(
            dbc.Card([
                dbc.CardHeader(html.H5("Potential Causal Relationships", className="text-primary")),
                dbc.CardBody([
                    dcc.Markdown(
                        relationship_markdown, 
                        style={"padding": "10px", "white-space": "pre-wrap"},
                        dangerously_allow_html=True
                    )
                ])
            ], className="mb-3")
        )
    
    # Recommendations
    if recommendations:
        # Format recommendations with proper spacing
        recommendation_markdown = format_list_items(recommendations)
        
        pattern_cards.append(
            dbc.Card([
                dbc.CardHeader(html.H5("Recommendations", className="text-primary")),
                dbc.CardBody([
                    dcc.Markdown(
                        recommendation_markdown, 
                        style={"padding": "10px", "white-space": "pre-wrap"},
                        dangerously_allow_html=True
                    )
                ])
            ], className="mb-3")
        )
    
    if not pattern_cards:
        return html.Div("No patterns identified in the analysis.")
    
    return html.Div(pattern_cards)

@app.callback(
    Output("visualizations-tab-content", "children"),
    [Input("analysis-store", "data"), Input("dataset-store", "data")],
    prevent_initial_call=True
)
def update_visualizations_tab(analysis_results, dataset_json):
    if not analysis_results or not dataset_json:
        return html.Div("No data available for visualization. Please analyze the data first.")
    
    # Convert JSON back to DataFrame
    df = pd.read_json(StringIO(dataset_json), orient='split')
    
    # Create overview visualization
    overview_fig = visualizer.create_overview_dashboard(df, "Dataset Overview")
    
    # Create correlation heatmap
    corr_fig = visualizer.create_correlation_heatmap(df)
    
    # Create visualizations for hidden patterns
    hidden_patterns = analysis_results.get("hidden_patterns", [])
    pattern_figs = []
    
    for i, pattern in enumerate(hidden_patterns):
        # Extract columns mentioned in the pattern
        mentioned_cols = [col for col in df.columns if col.lower() in pattern.lower()]
        
        if len(mentioned_cols) >= 2:
            x_col = mentioned_cols[0]
            y_col = mentioned_cols[1]
            color_col = mentioned_cols[2] if len(mentioned_cols) > 2 else None
            
            try:
                fig = visualizer.create_pattern_visualization(
                    df, 
                    f"Pattern {i+1}: {pattern}", 
                    x_col, 
                    y_col, 
                    color_col
                )
                pattern_figs.append(dcc.Graph(figure=fig, className="mb-4"))
            except Exception as e:
                pattern_figs.append(html.Div(f"Error creating visualization for pattern {i+1}: {str(e)}"))
    
    # Combine all visualizations
    visualization_content = [
        html.H4("Dataset Overview", className="mt-4 mb-3"),
        dcc.Graph(figure=overview_fig, className="mb-4"),
        
        html.H4("Correlation Heatmap", className="mt-4 mb-3"),
        dcc.Graph(figure=corr_fig, className="mb-4"),
    ]
    
    if pattern_figs:
        visualization_content.extend([
            html.H4("Pattern Visualizations", className="mt-4 mb-3"),
            *pattern_figs
        ])
    
    return html.Div(visualization_content)

@app.callback(
    Output("raw-tab-content", "children"),
    Input("analysis-store", "data"),
    prevent_initial_call=True
)
def update_raw_tab(analysis_results):
    if not analysis_results:
        return html.Div("No analysis results available. Please analyze the data first.")
    
    raw_analysis = analysis_results.get("raw_analysis", "No raw analysis available.")
    
    # Apply the markdown formatting function
    formatted_analysis = format_text_as_markdown(raw_analysis)
    
    return dbc.Card([
        dbc.CardBody([
            html.H4("Raw Analysis Output", className="card-title"),
            dcc.Markdown(
                formatted_analysis,
                style={
                    "background-color": "#f8f9fa",
                    "padding": "15px",
                    "border-radius": "5px",
                    "max-height": "600px",
                    "overflow-y": "auto",
                    "white-space": "pre-wrap"
                },
                dangerously_allow_html=True
            )
        ])
    ])

# Add a callback to automatically select the Specific Questions tab when specific questions are asked
@app.callback(
    Output("result-tabs", "active_tab"),
    [Input("analysis-store", "data"), Input("specific-questions", "value")],
    prevent_initial_call=True
)
def set_active_tab(analysis_results, specific_questions_text):
    if not analysis_results:
        return "summary-tab"
    
    # If there are specific questions and answers, select the specific questions tab
    if specific_questions_text and analysis_results.get("specific_answers"):
        return "specific-questions-tab"
    
    # Otherwise, default to summary tab
    return "summary-tab"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True) 