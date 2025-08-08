import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import warnings
import plotly.express as px
from utils import clean_dataframe

def auto_analyze_dataset(dataframe):
    """
    Automatically analyze a dataset without user prompting,
    generating summary statistics, visualizations, and insights.
    
    Args:
        dataframe (pd.DataFrame): The dataset to analyze
        
    Returns:
        dict: A dictionary containing summary stats, visualizations, and insights
    """
    results = {
        "summary": {},
        "visualizations": [],
        "insights": "",
        "recommendations": [],
        "error": None
    }
    
    try:
        # Create a copy of the dataframe to avoid any issues
        df = dataframe.copy()
        
        # Store original shape and missing values before cleaning
        original_shape = df.shape
        original_missing_values = df.isna().sum().to_dict()
        original_missing_percentage = (df.isna().sum() / len(df) * 100).to_dict()
        total_original_missing = df.isna().sum().sum()
        
        # Clean the dataset automatically
        df = clean_dataframe(
            df,
            drop_empty_rows=True,
            drop_empty_cols=True,
            drop_any_nan_rows=False  # Keep rows with some data
        )
        
        # Additional cleaning steps
        # Remove duplicate rows
        duplicates_before = len(df)
        df = df.drop_duplicates()
        duplicates_removed = duplicates_before - len(df)
        
        # Convert object columns to appropriate types where possible
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert to numeric if possible
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        # Fill missing values using forward fill and backward fill
        missing_before = df.isna().sum().sum()
        
        # For numeric columns: use forward fill, then backward fill
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # For categorical/text columns: use forward fill, then backward fill, then mode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # Forward fill and backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # If still missing, fill with mode (most frequent value)
            if df[col].isna().any():
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    # If no mode, fill with a placeholder
                    df[col] = df[col].fillna('Unknown')
        
        # For datetime columns: use forward fill, then backward fill
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            df[datetime_cols] = df[datetime_cols].fillna(method='ffill').fillna(method='bfill')
        
        missing_after = df.isna().sum().sum()
        missing_filled = missing_before - missing_after
        
        # Store cleaning statistics
        results["summary"]["duplicates_removed"] = duplicates_removed
        results["summary"]["missing_values_filled"] = missing_filled
        
        # Store original missing values info (before cleaning)
        results["summary"]["original_missing_values"] = original_missing_values
        results["summary"]["original_missing_percentage"] = original_missing_percentage
        results["summary"]["total_original_missing"] = total_original_missing
        
        # 1. Basic dataset summary (after cleaning)
        results["summary"]["original_shape"] = original_shape
        results["summary"]["shape"] = df.shape
        results["summary"]["rows_removed"] = original_shape[0] - df.shape[0]
        results["summary"]["columns_removed"] = original_shape[1] - df.shape[1]
        results["summary"]["columns"] = list(df.columns)
        results["summary"]["dtypes"] = {col: str(df[col].dtype) for col in df.columns}
        results["summary"]["missing_values"] = df.isna().sum().to_dict()
        results["summary"]["missing_percentage"] = (df.isna().sum() / len(df) * 100).to_dict()
        
        # 2. Numeric columns analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        results["summary"]["numeric_columns"] = numeric_cols
        
        if numeric_cols:
            # Calculate summary statistics for numeric columns
            results["summary"]["numeric_stats"] = df[numeric_cols].describe().to_dict()
            
            # Create distribution plots for numeric columns (up to 4)
            modern_colors = ['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981']
            for i, col in enumerate(numeric_cols[:4]):
                color = modern_colors[i % len(modern_colors)]
                fig = px.histogram(df, x=col, title=f"Distribution of {col}", marginal="rug",
                                   template="plotly_white", color_discrete_sequence=[color])
                fig.update_layout(
                    bargap=0.1,
                    title_font_size=16,
                    title_font_color='#1F2937',
                    font_family="Inter, system-ui, sans-serif",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='white',
                    margin=dict(t=60, b=40, l=40, r=40)
                )
                fig.update_xaxes(gridcolor='#F3F4F6', title_font_color='#4B5563')
                fig.update_yaxes(gridcolor='#F3F4F6', title_font_color='#4B5563')
                results["visualizations"].append({
                    "title": f"Distribution of {col}",
                    "type": "histogram",
                    "figure": fig,
                    "column": col
                })
        
        # 3. Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        results["summary"]["categorical_columns"] = categorical_cols
        
        if categorical_cols:
            # Create bar charts for categorical columns (up to 4)
            for i, col in enumerate(categorical_cols[:4]):
                # Only show the top 10 categories to avoid overcrowding
                value_counts = df[col].value_counts().head(10)
                color = modern_colors[i % len(modern_colors)]
                
                fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Top 10 values in {col}",
                             labels={'x': col, 'y': 'Count'}, template="plotly_white",
                             color_discrete_sequence=[color])
                fig.update_layout(
                    xaxis_tickangle=-45,
                    title_font_size=16,
                    title_font_color='#1F2937',
                    font_family="Inter, system-ui, sans-serif",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='white',
                    margin=dict(t=60, b=40, l=40, r=40)
                )
                fig.update_xaxes(gridcolor='#F3F4F6', title_font_color='#4B5563')
                fig.update_yaxes(gridcolor='#F3F4F6', title_font_color='#4B5563')
                fig.update_traces(marker_line_width=0, opacity=0.8)
                results["visualizations"].append({
                    "title": f"Top values in {col}",
                    "type": "bar",
                    "figure": fig,
                    "column": col
                })
        
        # 4. Correlation analysis (if at least 2 numeric columns)
        if len(numeric_cols) >= 2:
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Plot correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax,
                       cbar_kws={'shrink': 0.8}, square=True, linewidths=0.5,
                       annot_kws={'size': 10, 'weight': 'bold'})
            ax.set_title("Correlation Matrix", fontsize=16, fontweight='bold', color='#1F2937', pad=20)
            ax.set_facecolor('#FAFAFA')
            fig.patch.set_facecolor('white')
            plt.xticks(rotation=45, ha='right', fontsize=10, color='#4B5563')
            plt.yticks(rotation=0, fontsize=10, color='#4B5563')
            plt.tight_layout()
            results["visualizations"].append({
                "title": "Correlation Matrix",
                "type": "heatmap",
                "figure": fig,
                "columns": numeric_cols
            })
            
            # Find highest correlations
            corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    corr = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr) and abs(corr) > 0.5:  # Only strong correlations
                        corr_pairs.append((col1, col2, corr))
            
            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Create scatter plots for top correlated pairs (up to 3)
            for i, (col1, col2, corr) in enumerate(corr_pairs[:3]):
                color = modern_colors[i % len(modern_colors)]
                fig = px.scatter(df, x=col1, y=col2, title=f"Scatter plot of {col1} vs {col2} (corr: {corr:.2f})",
                                 trendline="ols", template="plotly_white",
                                 color_discrete_sequence=[color])
                fig.update_layout(
                    title_font_size=16,
                    title_font_color='#1F2937',
                    font_family="Inter, system-ui, sans-serif",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='white',
                    margin=dict(t=60, b=40, l=40, r=40)
                )
                fig.update_xaxes(gridcolor='#F3F4F6', title_font_color='#4B5563')
                fig.update_yaxes(gridcolor='#F3F4F6', title_font_color='#4B5563')
                fig.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0)))
                results["visualizations"].append({
                    "title": f"Relationship: {col1} vs {col2}",
                    "type": "scatter",
                    "figure": fig,
                    "columns": [col1, col2],
                    "correlation": corr
                })
        
        # 5. Time series detection and visualization
        date_cols = []
        for col in df.columns:
            try:
                if pd.api.types.is_object_dtype(df[col]):
                    # Try to convert to datetime with explicit format where possible
                    # Suppress the warning by using try-except
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        test_date = pd.to_datetime(df[col], errors='coerce')
                        # If at least 90% of values converted successfully, consider it a date column
                        if test_date.notna().sum() / len(test_date) > 0.9:
                            date_cols.append(col)
            except:
                pass
        
        results["summary"]["date_columns"] = date_cols
        
        # If we have date columns and numeric columns, create time series plots
        if date_cols and numeric_cols:
            date_col = date_cols[0]  # Use the first date column found
            try:
                # Convert to datetime
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Select top 3 numeric columns
                for i, col in enumerate(numeric_cols[:3]):
                    color = modern_colors[i % len(modern_colors)]
                    # Create a time series plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Group by month and take mean if dataset is large
                    if len(df) > 100:
                        time_data = df.set_index(date_col)[col].resample('M').mean()
                        time_data.plot(ax=ax, color=color, linewidth=2.5)
                        title = f"Monthly trend of {col}"
                    else:
                        df.plot(x=date_col, y=col, ax=ax, color=color, linewidth=2.5)
                        title = f"Trend of {col} over time"
                    
                    ax.set_title(title, fontsize=16, fontweight='bold', color='#1F2937', pad=20)
                    ax.set_xlabel(date_col, fontsize=12, color='#4B5563')
                    ax.set_ylabel(col, fontsize=12, color='#4B5563')
                    ax.grid(True, alpha=0.3, color='#F3F4F6')
                    ax.set_facecolor('#FAFAFA')
                    fig.patch.set_facecolor('white')
                    plt.tight_layout()
                    results["visualizations"].append({
                        "title": title,
                        "type": "time_series",
                        "figure": fig,
                        "columns": [date_col, col]
                    })
            except Exception as e:
                # If time series visualization fails, add error to results but continue
                results["summary"]["time_series_error"] = str(e)
        
        # 6. Generate insights using Gemini
        insights = generate_dataset_insights(df, results["summary"], results["visualizations"])
        results["insights"] = insights
        
        # 7. Generate analysis recommendations based on data
        recommendations = generate_recommendations(df, results["summary"])
        results["recommendations"] = recommendations
        
    except Exception as e:
        # If analysis fails, return the error
        results["error"] = str(e)
        
    return results

def generate_dataset_insights(dataframe, summary, visualizations):
    """
    Use Gemini to generate insights about the dataset
    
    Args:
        dataframe (pd.DataFrame): The dataset
        summary (dict): Summary statistics
        visualizations (list): List of visualizations created
        
    Returns:
        str: Insights about the dataset
    """
    # Create visualization descriptions for the prompt
    viz_descriptions = []
    for viz in visualizations:
        viz_descriptions.append(f"- {viz['title']} ({viz['type']})")
    
    # Format summary statistics
    data_summary = {
        "rows": summary["shape"][0],
        "columns": summary["shape"][1],
        "missing_values": sum(summary["missing_values"].values()),
        "numeric_columns": len(summary.get("numeric_columns", [])),
        "categorical_columns": len(summary.get("categorical_columns", [])),
        "date_columns": len(summary.get("date_columns", []))
    }
    
    # Generate sample data
    sample_data = dataframe.head(3).to_dict(orient="records")
    
    prompt = f"""
    You are a data analysis expert. You are given a dataset with the following information:
    
    Basic Info:
    - Rows: {data_summary['rows']}
    - Columns: {data_summary['columns']}
    - Missing Values: {data_summary['missing_values']}
    - Numeric Columns: {data_summary['numeric_columns']}
    - Categorical Columns: {data_summary['categorical_columns']}
    - Date Columns: {data_summary['date_columns']}
    
    Sample Data:
    {sample_data}
    
    Visualizations created:
    {chr(10).join(viz_descriptions)}
    
    Based on this information, provide 3-5 concise, valuable insights about this dataset. Focus on:
    1. Data quality issues (like missing values)
    2. Notable patterns or distributions
    3. Interesting relationships between variables
    4. Potential business implications of the data
    
    Format your response as bullet points, with each insight being 1-2 sentences maximum.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # If Gemini fails, return a basic insight
        return f"Automatic insights generation failed: {str(e)}"

def generate_recommendations(dataframe, summary):
    """
    Generate recommendations for further analysis based on the dataset
    
    Args:
        dataframe (pd.DataFrame): The dataset
        summary (dict): Summary statistics
        
    Returns:
        list: List of analysis recommendations
    """
    recommendations = []
    
    # Recommendation based on missing values
    missing_cols = [col for col, count in summary["missing_values"].items() if count > 0]
    if missing_cols:
        high_missing_cols = [col for col, pct in summary["missing_percentage"].items() if pct > 20]
        if high_missing_cols:
            recommendations.append(f"Consider handling missing data in columns with >20% missing values: {', '.join(high_missing_cols[:3])}")
    
    # Recommendation based on numeric columns
    if len(summary.get("numeric_columns", [])) >= 2:
        recommendations.append("Explore relationships between numeric variables using scatter plots or regression analysis")
    
    # Recommendation based on categorical columns
    if len(summary.get("categorical_columns", [])) > 0:
        recommendations.append("Analyze the distribution of categorical variables and their relationship with numeric variables")
    
    # Recommendation based on date columns
    if len(summary.get("date_columns", [])) > 0:
        recommendations.append("Perform time series analysis to identify trends, seasonality, or anomalies")
    
    # Recommendation for outlier detection if we have numeric columns
    if len(summary.get("numeric_columns", [])) > 0:
        recommendations.append("Identify and analyze outliers in numeric variables to improve data quality")
    
    # Add general recommendations if we have few specific ones
    if len(recommendations) < 3:
        recommendations.append("Consider feature engineering to create new variables that might better capture the underlying patterns")
        recommendations.append("Test different visualization types to better understand the data distribution")
    
    return recommendations 