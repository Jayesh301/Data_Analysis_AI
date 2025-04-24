import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import warnings

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
        
        # 1. Basic dataset summary
        results["summary"]["shape"] = df.shape
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
            for i, col in enumerate(numeric_cols[:4]):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                plt.tight_layout()
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
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                ax.set_title(f"Top 10 values in {col}")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
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
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Correlation Matrix")
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
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
                ax.set_title(f"Scatter plot of {col1} vs {col2} (corr: {corr:.2f})")
                plt.tight_layout()
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
                for col in numeric_cols[:3]:
                    # Create a time series plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Group by month and take mean if dataset is large
                    if len(df) > 100:
                        time_data = df.set_index(date_col)[col].resample('M').mean()
                        time_data.plot(ax=ax)
                        title = f"Monthly trend of {col}"
                    else:
                        df.plot(x=date_col, y=col, ax=ax)
                        title = f"Trend of {col} over time"
                    
                    ax.set_title(title)
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