import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataVisualizer:
    """
    Creates visualizations for datasets and analysis results.
    """
    
    def __init__(self, theme='plotly_white'):
        """
        Initialize the visualizer with a theme.
        
        Args:
            theme (str): Plotly theme to use for visualizations
        """
        self.theme = theme
    
    def create_overview_dashboard(self, df, title="Dataset Overview"):
        """
        Create an overview dashboard for a dataset.
        
        Args:
            df (pd.DataFrame): The dataset to visualize
            title (str): Dashboard title
            
        Returns:
            plotly.graph_objects.Figure: A plotly figure with multiple subplots
        """
        # Determine number of numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number, 'datetime64']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Limit to first 10 of each type to avoid overcrowding
        numeric_cols = numeric_cols[:10]
        cat_cols = cat_cols[:10]
        date_cols = date_cols[:5]
        
        # Calculate number of subplots needed
        n_numeric = len(numeric_cols)
        n_cat = len(cat_cols)
        n_date = len(date_cols)
        
        # Create subplot grid
        n_rows = n_numeric + n_cat + n_date
        fig = make_subplots(
            rows=n_rows, 
            cols=1,
            subplot_titles=[f"{col} Distribution" for col in numeric_cols + cat_cols + date_cols],
            vertical_spacing=0.05
        )
        
        # Add numeric distributions
        for i, col in enumerate(numeric_cols, 1):
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    name=col,
                    marker_color='royalblue',
                    opacity=0.7
                ),
                row=i, col=1
            )
            
            # Add box plot on secondary y-axis
            fig.add_trace(
                go.Box(
                    x=df[col],
                    name=col,
                    marker_color='indianred',
                    opacity=0.7,
                    orientation='h'
                ),
                row=i, col=1
            )
        
        # Add categorical distributions
        for i, col in enumerate(cat_cols, n_numeric + 1):
            # Get top 10 categories by frequency
            value_counts = df[col].value_counts().head(10)
            
            fig.add_trace(
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    name=col,
                    marker_color='lightseagreen'
                ),
                row=i, col=1
            )
        
        # Add date distributions if available
        for i, col in enumerate(date_cols, n_numeric + n_cat + 1):
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(df[col]):
                try:
                    date_series = pd.to_datetime(df[col])
                except:
                    continue
            else:
                date_series = df[col]
            
            # Group by month or day depending on date range
            date_range = (date_series.max() - date_series.min()).days
            
            if date_range > 365:
                # Group by month for longer periods
                date_counts = date_series.dt.to_period('M').value_counts().sort_index()
                date_labels = [str(d) for d in date_counts.index]
            else:
                # Group by day for shorter periods
                date_counts = date_series.dt.date.value_counts().sort_index()
                date_labels = [str(d) for d in date_counts.index]
            
            fig.add_trace(
                go.Scatter(
                    x=date_labels,
                    y=date_counts.values,
                    mode='lines+markers',
                    name=col,
                    marker_color='darkorange'
                ),
                row=i, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=300 * n_rows,
            width=1000,
            showlegend=False,
            template=self.theme
        )
        
        return fig
    
    def create_correlation_heatmap(self, df, title="Correlation Matrix"):
        """
        Create a correlation heatmap for numeric columns in the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to visualize
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: A plotly figure with the correlation heatmap
        """
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title=title
        )
        
        fig.update_layout(
            width=800,
            height=800,
            template=self.theme
        )
        
        return fig
    
    def create_pattern_visualization(self, df, pattern_description, x_col, y_col, color_col=None, size_col=None, facet_col=None):
        """
        Create a visualization for a specific pattern in the data.
        
        Args:
            df (pd.DataFrame): The dataset to visualize
            pattern_description (str): Description of the pattern
            x_col (str): Column to use for x-axis
            y_col (str): Column to use for y-axis
            color_col (str, optional): Column to use for color
            size_col (str, optional): Column to use for point size
            facet_col (str, optional): Column to use for faceting
            
        Returns:
            plotly.graph_objects.Figure: A plotly figure visualizing the pattern
        """
        # Determine the best type of plot based on column types
        x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col].dtype)
        y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col].dtype)
        
        if x_is_numeric and y_is_numeric:
            # Scatter plot for two numeric columns
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                size=size_col,
                facet_col=facet_col,
                title=pattern_description,
                trendline="ols" if len(df) < 5000 else None,  # Add trendline for smaller datasets
                opacity=0.7
            )
            
        elif x_is_numeric and not y_is_numeric:
            # Box plot for numeric vs categorical
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                facet_col=facet_col,
                title=pattern_description
            )
            
        elif not x_is_numeric and y_is_numeric:
            # Bar plot for categorical vs numeric
            fig = px.bar(
                df.groupby(x_col)[y_col].mean().reset_index(),
                x=x_col,
                y=y_col,
                color=color_col,
                title=pattern_description
            )
            
        else:
            # Heatmap for two categorical columns
            heatmap_data = pd.crosstab(df[y_col], df[x_col], normalize='all')
            fig = px.imshow(
                heatmap_data,
                title=pattern_description,
                color_continuous_scale='Viridis'
            )
        
        fig.update_layout(
            width=900,
            height=600,
            template=self.theme
        )
        
        return fig
    
    def create_insight_dashboard(self, df, analysis_results):
        """
        Create a dashboard visualizing the insights from the analysis.
        
        Args:
            df (pd.DataFrame): The dataset
            analysis_results (dict): Results from the LLM analysis
            
        Returns:
            list: List of plotly figures visualizing the insights
        """
        figures = []
        
        # Add correlation heatmap
        figures.append(self.create_correlation_heatmap(df))
        
        # Create visualizations for each hidden pattern
        for i, pattern in enumerate(analysis_results.get('hidden_patterns', [])):
            # Extract key columns from the pattern description
            # This is a simplified approach - in a real application, you might want to use
            # NLP to extract the relevant columns from the pattern description
            
            # Get all column names
            all_cols = df.columns.tolist()
            
            # Find columns mentioned in the pattern description
            mentioned_cols = [col for col in all_cols if col.lower() in pattern.lower()]
            
            # If we found at least two columns, create a visualization
            if len(mentioned_cols) >= 2:
                x_col = mentioned_cols[0]
                y_col = mentioned_cols[1]
                
                # Use a third column for color if available
                color_col = mentioned_cols[2] if len(mentioned_cols) > 2 else None
                
                figures.append(
                    self.create_pattern_visualization(
                        df, 
                        f"Pattern {i+1}: {pattern}", 
                        x_col, 
                        y_col, 
                        color_col
                    )
                )
        
        return figures
    
    def create_comparison_visualization(self, actual_patterns, discovered_patterns, evaluation):
        """
        Create a visualization comparing actual patterns with discovered patterns.
        
        Args:
            actual_patterns (list): List of actual patterns
            discovered_patterns (list): List of discovered patterns
            evaluation (str): Evaluation text from the LLM
            
        Returns:
            plotly.graph_objects.Figure: A plotly figure visualizing the comparison
        """
        # Create a figure with two subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Actual Patterns", "Discovered Patterns"),
            specs=[[{"type": "table"}, {"type": "table"}]]
        )
        
        # Add actual patterns table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["#", "Actual Pattern"],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        list(range(1, len(actual_patterns) + 1)),
                        actual_patterns
                    ],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=1, col=1
        )
        
        # Add discovered patterns table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["#", "Discovered Pattern"],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        list(range(1, len(discovered_patterns) + 1)),
                        discovered_patterns
                    ],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Comparison of Actual vs. Discovered Patterns",
            height=400,
            width=1200,
            template=self.theme
        )
        
        return fig 