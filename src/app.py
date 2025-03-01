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
import markdown

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
# Get model name from environment variables with fallback
model_name = os.getenv("OPENAI_MODEL", "gpt-4.5-preview")

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

# Add custom CSS for markdown rendering
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom CSS for markdown rendering */
            .markdown-content strong {
                font-weight: bold;
            }
            .markdown-content h3 {
                margin-top: 1.5rem;
                margin-bottom: 1rem;
                font-weight: 600;
            }
            .markdown-content ul, .markdown-content ol {
                margin-bottom: 1rem;
                padding-left: 2rem;
            }
            .markdown-content li {
                margin-bottom: 0.5rem;
            }
            .markdown-content p {
                margin-bottom: 1rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

server = app.server
app.title = "Hidden Data Insights"

# Helper function to enhance markdown for better readability
def enhance_markdown(text):
    """
    Enhance markdown text for better readability:
    - Ensure proper list formatting
    - Add emphasis to key metrics and terms
    - Ensure proper spacing
    """
    if not text:
        return text
    
    # Normalize line endings
    text = text.replace('\r\n', '\n')
    
    # First, preserve any existing markdown formatting
    # Save code blocks to prevent modifying them
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"CODE_BLOCK_{len(code_blocks)-1}"
    
    # Save existing markdown code blocks
    text = re.sub(r'```[\s\S]+?```', save_code_block, text)
    
    # Handle section headers - convert "Title:" format to markdown headers
    text = re.sub(r'^([A-Z][A-Za-z\s]+):(\s*)', r'### \1\2', text, flags=re.MULTILINE)
    
    # Handle dash-separated items (common in summaries)
    # Convert "- Item:" to HTML bold for better rendering
    text = re.sub(r'\s*-\s+([^:]+):', r'\n\n<strong>\1:</strong>', text)
    
    # Ensure numbered lists have proper spacing and formatting
    # Add a space after numbers if missing
    text = re.sub(r'^(\d+)\.(?!\s)', r'\1. ', text, flags=re.MULTILINE)
    
    # Ensure bullet points have proper spacing
    text = re.sub(r'^\s*[-•]\s*', r'- ', text, flags=re.MULTILINE)
    
    # Bold percentages for emphasis using HTML tags
    text = re.sub(r'(\d+\.?\d*\s*%)', r'<strong>\1</strong>', text)
    
    # Bold key terms for emphasis using HTML tags
    key_terms = [
        r'\bkey\b', r'\bsignificant\b', r'\bimportant\b', r'\bcritical\b', 
        r'\bhighly\b', r'\bnotable\b', r'\bsubstantial\b', r'\bmajor\b',
        r'\bstrong\b', r'\bclear\b', r'\bprimary\b', r'\bexceptional\b'
    ]
    
    for term in key_terms:
        text = re.sub(term, f'<strong>{term.replace(r"\\b", "")}</strong>', text, flags=re.IGNORECASE)
    
    # Special handling for summary text that often has dash-separated items
    # Look for patterns like "1. Title - Detail: More details"
    text = re.sub(r'(\d+\.\s+[^-]+)\s*-\s*([^:]+):', r'\1\n\n<strong>\2:</strong>', text)
    
    # Convert any remaining markdown-style bold (**text**) to HTML bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    
    # Improve paragraph and list formatting
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for i, line in enumerate(lines):
        # Check if this line is a list item
        is_list_item = bool(line.strip().startswith('- ') or re.match(r'^\d+\.\s', line.strip()))
        
        # If we're transitioning into a list, add a blank line before
        if is_list_item and not in_list and i > 0 and formatted_lines[-1].strip():
            formatted_lines.append('')
        
        # Add the current line
        formatted_lines.append(line)
        
        # If we're in a list and this is the last item or the next item is not a list item,
        # add a blank line after
        if is_list_item and i < len(lines) - 1:
            next_line = lines[i+1]
            next_is_list = bool(next_line.strip().startswith('- ') or re.match(r'^\d+\.\s', next_line.strip()))
            
            if not next_is_list and next_line.strip():
                formatted_lines.append('')
        
        # Update list state
        in_list = is_list_item
    
    # Join lines back together
    text = '\n'.join(formatted_lines)
    
    # Ensure proper paragraph breaks by adding double line breaks between paragraphs
    # but avoid adding too many breaks
    text = re.sub(r'([^\n])\n([^\n-])', r'\1\n\n\2', text)
    
    # Add line breaks after colons in certain contexts (common in summaries)
    text = re.sub(r':\s+([A-Z][a-z])', r':\n\n\1', text)
    
    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', r'\n\n', text)
    
    # Restore code blocks
    for i, block in enumerate(code_blocks):
        text = text.replace(f"CODE_BLOCK_{i}", block)
    
    return text

# Helper function to format list items as markdown
def format_list_as_markdown(items):
    """
    Format a list of items as proper markdown list
    """
    if not items:
        return ""
    
    # If we have a single item that might contain multiple points
    if len(items) == 1 and len(items[0]) > 200:
        # Try to split it into multiple items based on common patterns
        text = items[0]
        
        # Check for numbered lists
        if re.search(r'^\d+\.\s', text, re.MULTILINE):
            # Already has numbered list formatting, just enhance it
            return enhance_markdown(text)
            
        # Check for bullet points
        if re.search(r'^[-•]\s', text, re.MULTILINE):
            # Already has bullet list formatting, just enhance it
            return enhance_markdown(text)
            
        # Try to split on numbered patterns like "1." or "1)"
        if re.search(r'\d+[\.\)]\s', text):
            # Split the text on these patterns
            parts = re.split(r'(\d+[\.\)]\s+)', text)
            if len(parts) > 2:  # We have at least one match plus text before and after
                formatted_text = ""
                current_item = ""
                
                for i, part in enumerate(parts):
                    if re.match(r'\d+[\.\)]\s+', part):
                        # This is a number marker
                        if current_item:
                            formatted_text += current_item + "\n\n"
                        current_item = part
                    else:
                        current_item += part
                
                if current_item:
                    formatted_text += current_item
                
                return enhance_markdown(formatted_text)
            
        # Split on periods followed by space as a last resort
        if '.' in text:
            items = []
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    # For the first item, don't add a number
                    if i == 0:
                        items.append(sentence.strip())
                    else:
                        # For subsequent items, add a number
                        items.append(f"{i}. {sentence.strip()}")
    
    # Format each item as a markdown list item
    markdown_items = []
    for i, item in enumerate(items):
        if not item.strip():
            continue
            
        # Clean up the item
        clean_item = item.strip()
        
        # If it's already a list item, use it as is
        if clean_item.startswith('- ') or re.match(r'^\d+[\.\)]\s', clean_item):
            # Ensure proper spacing after the number
            if re.match(r'^\d+[\.\)]\s', clean_item):
                clean_item = re.sub(r'^(\d+[\.\)])\s*', r'\1 ', clean_item)
            markdown_items.append(clean_item)
        else:
            # Make it a numbered list item if it looks like it should be numbered
            if re.match(r'^\d+\s', clean_item):
                # It starts with a number but is missing the period
                clean_item = re.sub(r'^(\d+)\s+', r'\1. ', clean_item)
                markdown_items.append(clean_item)
            else:
                # Make it a bullet point
                markdown_items.append(f"- {clean_item}")
    
    # Join with blank lines between items for proper markdown rendering
    return enhance_markdown('\n\n'.join(markdown_items))

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Hidden Data Insights", className="display-4 text-primary mb-4"),
            html.P(
                f"Discover hidden patterns in your data using OpenAI's {model_name}.",
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
                                f"Analyze with {model_name}",
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
                f"Hidden Data Insights - Powered by OpenAI {model_name}",
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
    [Output("analysis-store", "data"), Output("analysis-loading-indicator", "style")],
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
        return None, {"display": "none"}
    
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
    
    # If there were specific questions but no specific answers in the results,
    # create placeholder answers to ensure they appear in the UI
    if questions_list and not analysis_results.get("specific_answers"):
        analysis_results["specific_answers"] = []
        for question in questions_list:
            # Create a placeholder that will trigger our extraction logic
            analysis_results["specific_answers"].append(f"Q: {question}\nA: See analysis for details.")
    
    # Return the analysis results and hide the loading indicator
    return analysis_results, {"display": "none"}

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
        Output("analysis-loading-indicator", "style", allow_duplicate=True)
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
    formatted_summary = enhance_markdown(summary)
    
    return dbc.Card([
        dbc.CardBody([
            html.H4("Data Analysis Summary", className="card-title mb-4"),
            html.Div([
                dcc.Markdown(
                    formatted_summary,
                    className="markdown-content",
                    style={
                        "white-space": "pre-wrap",
                        "line-height": "1.8",
                        "font-size": "1.1rem",
                        "overflow-wrap": "break-word",
                        "word-break": "normal"
                    },
                    dangerously_allow_html=True
                )
            ], className="card-text p-3")
        ], className="p-4")
    ], className="shadow-sm")

@app.callback(
    Output("specific-questions-tab-content", "children"),
    [Input("analysis-store", "data"), Input("specific-questions", "value")],
    prevent_initial_call=True
)
def update_specific_questions_tab(analysis_results, specific_questions_text):
    if not analysis_results:
        return html.Div("No analysis results available. Please analyze the data first.")
    
    # Check if specific questions were asked
    if not specific_questions_text or not specific_questions_text.strip():
        return html.Div([
            html.H4("No Specific Questions", className="mb-4"),
            html.P("You didn't enter any specific questions before analysis. To get targeted insights, add questions in the 'Analyze Data' section and run the analysis again.", className="lead")
        ])
    
    # Get specific answers from analysis results
    specific_answers = analysis_results.get("specific_answers", [])
    
    # If no specific answers were found but questions were asked
    if not specific_answers:
        # Try to extract answers from the raw analysis
        raw_analysis = analysis_results.get("raw_analysis", "")
        if not raw_analysis:
            raw_analysis = analysis_results.get("summary", "") + "\n\n" + "\n\n".join([
                "\n".join(analysis_results.get("hidden_patterns", [])),
                "\n".join(analysis_results.get("unusual_correlations", [])),
                "\n".join(analysis_results.get("causal_relationships", [])),
                "\n".join(analysis_results.get("recommendations", []))
            ])
        
        # Look for question-answer patterns in the raw analysis
        extracted_answers = []
        questions = [q.strip() for q in specific_questions_text.split('\n') if q.strip()]
        
        for question in questions:
            # Try to find a direct answer in the raw analysis
            # Look for the question text or keywords
            keywords = [word for word in question.lower().split() if len(word) > 3]
            
            # Find relevant sections
            relevant_sections = []
            for keyword in keywords:
                if keyword in raw_analysis.lower():
                    paragraphs = raw_analysis.split('\n\n')
                    for para in paragraphs:
                        if keyword in para.lower() and para not in relevant_sections:
                            relevant_sections.append(para)
            
            if relevant_sections:
                answer = "Based on the analysis: " + "\n\n".join(relevant_sections)
            else:
                answer = "See analysis for details."
                
            extracted_answers.append({"question": question, "answer": answer})
            
        specific_answers = extracted_answers
    
    # Format and display the answers
    answer_cards = []
    
    for i, qa_pair in enumerate(specific_answers):
        if isinstance(qa_pair, dict):
            question = qa_pair.get("question", f"Question {i+1}")
            answer_text = qa_pair.get("answer", "No answer provided.")
        else:
            # Handle the case where specific_answers might be a list of strings
            question = f"Question {i+1}"
            answer_text = qa_pair
            
        # If the answer is just a placeholder, try to extract from raw analysis
        if answer_text == "See analysis for details.":
            raw_analysis = analysis_results.get("raw_analysis", "")
            if not raw_analysis:
                raw_analysis = analysis_results.get("summary", "") + "\n\n" + "\n\n".join([
                    "\n".join(analysis_results.get("hidden_patterns", [])),
                    "\n".join(analysis_results.get("unusual_correlations", [])),
                    "\n".join(analysis_results.get("causal_relationships", [])),
                    "\n".join(analysis_results.get("recommendations", []))
                ])
            
            # Extract keywords from the question
            question_keywords = [word for word in question.lower().split() if len(word) > 3]
            
            # Find relevant sections
            relevant_sections = []
            for keyword in question_keywords:
                if keyword in raw_analysis.lower():
                    paragraphs = raw_analysis.split('\n\n')
                    for para in paragraphs:
                        if keyword in para.lower() and para not in relevant_sections:
                            relevant_sections.append(para)
            
            if relevant_sections:
                answer_text = "Based on the analysis: " + "\n\n".join(relevant_sections)
            else:
                answer_text = "No specific answer found in the analysis. Please check the Summary and Hidden Patterns tabs for relevant information."
        
        # Format the answer text with enhanced markdown
        formatted_answer = enhance_markdown(answer_text)
        
        answer_cards.append(
            dbc.Card([
                dbc.CardHeader(html.H5(f"Question {i+1}: {question}", className="text-primary")),
                dbc.CardBody([
                    html.Div([
                        html.Strong("Answer: ", className="me-2"),
                        html.Div([
                            dcc.Markdown(
                                formatted_answer,
                                className="markdown-content",
                                style={
                                    "white-space": "pre-wrap",
                                    "line-height": "1.8",
                                    "font-size": "1.1rem",
                                    "overflow-wrap": "break-word",
                                    "word-break": "normal"
                                },
                                dangerously_allow_html=True
                            )
                        ])
                    ])
                ], className="p-4")
            ], className="mb-4 shadow-sm")
        )
    
    if not answer_cards:
        return html.Div([
            html.H4("Specific Questions Analysis", className="mb-4"),
            dbc.Alert([
                html.H5("Processing Error", className="alert-heading"),
                html.P("There was an issue processing the answers to your specific questions."),
                html.P("Please try analyzing the data again or check the 'Raw Analysis' tab for insights.")
            ], color="danger", className="mb-4")
        ])
    
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
        # Format patterns with proper spacing and numbering
        # Check if we need to join the patterns or if they're already formatted
        if len(hidden_patterns) == 1 and len(hidden_patterns[0]) > 200 and re.search(r'\d+[\.\)]', hidden_patterns[0]):
            # This is likely a pre-formatted list with numbers
            pattern_markdown = enhance_markdown(hidden_patterns[0])
        else:
            # Format as a numbered list
            numbered_patterns = []
            for i, pattern in enumerate(hidden_patterns):
                numbered_patterns.append(f"{i+1}. {pattern}")
            pattern_markdown = enhance_markdown("\n\n".join(numbered_patterns))
        
        pattern_cards.append(
            dbc.Card([
                dbc.CardHeader(html.H5("Hidden Patterns", className="text-primary")),
                dbc.CardBody([
                    html.Div([
                        dcc.Markdown(
                            pattern_markdown, 
                            className="markdown-content",
                            style={
                                "white-space": "pre-wrap",
                                "line-height": "1.8",
                                "font-size": "1.1rem",
                                "overflow-wrap": "break-word",
                                "word-break": "normal"
                            },
                            dangerously_allow_html=True
                        )
                    ], className="p-3")
                ], className="p-4")
            ], className="mb-4 shadow-sm")
        )
    
    # Unusual Correlations
    if unusual_correlations:
        # Format correlations with proper spacing and numbering
        if len(unusual_correlations) == 1 and len(unusual_correlations[0]) > 200 and re.search(r'\d+[\.\)]', unusual_correlations[0]):
            # This is likely a pre-formatted list with numbers
            correlation_markdown = enhance_markdown(unusual_correlations[0])
        else:
            # Format as a numbered list
            numbered_correlations = []
            for i, correlation in enumerate(unusual_correlations):
                numbered_correlations.append(f"{i+1}. {correlation}")
            correlation_markdown = enhance_markdown("\n\n".join(numbered_correlations))
        
        pattern_cards.append(
            dbc.Card([
                dbc.CardHeader(html.H5("Unusual Correlations", className="text-primary")),
                dbc.CardBody([
                    html.Div([
                        dcc.Markdown(
                            correlation_markdown, 
                            className="markdown-content",
                            style={
                                "white-space": "pre-wrap",
                                "line-height": "1.8",
                                "font-size": "1.1rem",
                                "overflow-wrap": "break-word",
                                "word-break": "normal"
                            },
                            dangerously_allow_html=True
                        )
                    ], className="p-3")
                ], className="p-4")
            ], className="mb-4 shadow-sm")
        )
    
    # Causal Relationships
    if causal_relationships:
        # Format relationships with proper spacing and numbering
        if len(causal_relationships) == 1 and len(causal_relationships[0]) > 200 and re.search(r'\d+[\.\)]', causal_relationships[0]):
            # This is likely a pre-formatted list with numbers
            relationship_markdown = enhance_markdown(causal_relationships[0])
        else:
            # Format as a numbered list
            numbered_relationships = []
            for i, relationship in enumerate(causal_relationships):
                numbered_relationships.append(f"{i+1}. {relationship}")
            relationship_markdown = enhance_markdown("\n\n".join(numbered_relationships))
        
        pattern_cards.append(
            dbc.Card([
                dbc.CardHeader(html.H5("Potential Causal Relationships", className="text-primary")),
                dbc.CardBody([
                    html.Div([
                        dcc.Markdown(
                            relationship_markdown, 
                            className="markdown-content",
                            style={
                                "white-space": "pre-wrap",
                                "line-height": "1.8",
                                "font-size": "1.1rem",
                                "overflow-wrap": "break-word",
                                "word-break": "normal"
                            },
                            dangerously_allow_html=True
                        )
                    ], className="p-3")
                ], className="p-4")
            ], className="mb-4 shadow-sm")
        )
    
    # Recommendations
    if recommendations:
        # Format recommendations with proper spacing and numbering
        if len(recommendations) == 1 and len(recommendations[0]) > 200 and re.search(r'\d+[\.\)]', recommendations[0]):
            # This is likely a pre-formatted list with numbers
            recommendation_markdown = enhance_markdown(recommendations[0])
        else:
            # Format as a numbered list
            numbered_recommendations = []
            for i, recommendation in enumerate(recommendations):
                numbered_recommendations.append(f"{i+1}. {recommendation}")
            recommendation_markdown = enhance_markdown("\n\n".join(numbered_recommendations))
        
        pattern_cards.append(
            dbc.Card([
                dbc.CardHeader(html.H5("Recommendations", className="text-primary")),
                dbc.CardBody([
                    html.Div([
                        dcc.Markdown(
                            recommendation_markdown, 
                            className="markdown-content",
                            style={
                                "white-space": "pre-wrap",
                                "line-height": "1.8",
                                "font-size": "1.1rem",
                                "overflow-wrap": "break-word",
                                "word-break": "normal"
                            },
                            dangerously_allow_html=True
                        )
                    ], className="p-3")
                ], className="p-4")
            ], className="mb-4 shadow-sm")
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
    formatted_analysis = enhance_markdown(raw_analysis)
    
    return dbc.Card([
        dbc.CardBody([
            html.H4("Raw Analysis Output", className="card-title mb-4"),
            html.Div([
                dcc.Markdown(
                    formatted_analysis,
                    className="markdown-content",
                    style={
                        "background-color": "#f8f9fa",
                        "padding": "20px",
                        "border-radius": "5px",
                        "max-height": "600px",
                        "overflow-y": "auto",
                        "white-space": "pre-wrap",
                        "line-height": "1.8",
                        "font-size": "1.05rem",
                        "overflow-wrap": "break-word",
                        "word-break": "normal"
                    },
                    dangerously_allow_html=True
                )
            ], className="p-2")
        ], className="p-4")
    ], className="shadow-sm")

# Add a callback to automatically select the Specific Questions tab when specific questions are asked
@app.callback(
    Output("result-tabs", "active_tab"),
    [Input("analysis-store", "data"), Input("specific-questions", "value")],
    prevent_initial_call=True
)
def set_active_tab(analysis_results, specific_questions_text):
    if not analysis_results:
        return "summary-tab"
    
    # If there are specific questions, select the specific questions tab
    # regardless of whether there are direct answers or not
    if specific_questions_text and specific_questions_text.strip():
        return "specific-questions-tab"
    
    # Otherwise, default to summary tab
    return "summary-tab"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True) 