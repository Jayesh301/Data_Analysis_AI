import pandas as pd
import matplotlib.pyplot as plt
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr

def execute_query(query_string, dataframe):
    """
    Safely execute a pandas query on a DataFrame
    
    Args:
        query_string (str): The pandas query to execute
        dataframe (pd.DataFrame): The DataFrame to query
        
    Returns:
        dict: Results of query execution
    """
    # Create a copy of the dataframe to avoid modifications to the original
    df = dataframe.copy()
    
    # Prepare variables to capture outputs
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = {
        "figure": None,
        "result_data": None,
        "error": None,
        "description": "Analysis completed successfully."
    }
    
    try:
        # Execute the query with redirected output
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Clean up the query if needed
            query = query_string.strip()
            
            # Check if the query needs to generate a visualization
            needs_viz = any(viz_term in query.lower() for viz_term in 
                           ['plot(', '.plot.', '.hist(', '.boxplot(', '.scatter(', '.bar(', '.barh(', '.pie('])
            
            # Execute the query and capture the result
            query_result = eval(query)
            result["result_data"] = query_result
            
            # Create visualization if needed based on query result type
            if needs_viz:
                # The query likely created a plot already
                result["figure"] = plt.gcf()
            elif isinstance(query_result, pd.DataFrame) and not query_result.empty:
                # Create a visualization from the result
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Determine the best chart type based on data
                if query_result.shape[0] <= 20:  # Small enough for a bar chart
                    if query_result.shape[1] >= 2 and all(pd.api.types.is_numeric_dtype(query_result[col]) 
                                                        for col in query_result.columns[1:]):
                        # Get the first column for labels and numeric columns for values
                        query_result.set_index(query_result.columns[0]).plot(
                            kind='bar', ax=ax, rot=45)
                    else:
                        # Simple bar chart
                        query_result.plot(kind='bar', ax=ax, rot=45)
                elif query_result.shape[1] >= 2 and all(pd.api.types.is_numeric_dtype(query_result[col]) 
                                                      for col in query_result.columns):
                    # Line chart for multi-column numeric data
                    query_result.plot(ax=ax)
                else:
                    # Table for complex data
                    ax.axis('off')
                    ax.table(cellText=query_result.head(10).values, 
                             colLabels=query_result.columns, 
                             loc='center',
                             cellLoc='center')
                    plt.title(f"Table of Results (First 10 of {query_result.shape[0]} Rows)")
                
                plt.tight_layout()
                result["figure"] = fig
                
                # Use a more descriptive summary
                result["description"] = f"Analysis returned {query_result.shape[0]} rows of data."
            elif isinstance(query_result, pd.Series):
                fig, ax = plt.subplots(figsize=(10, 6))
                if len(query_result) <= 20:
                    query_result.plot(kind='bar', ax=ax)
                else:
                    if pd.api.types.is_numeric_dtype(query_result):
                        query_result.plot(kind='line', ax=ax)
                    else:
                        query_result.value_counts().head(10).plot(kind='bar', ax=ax)
                plt.tight_layout()
                result["figure"] = fig
            
            # If no figure was created but we have data, create a text summary
            if result["figure"] is None and result["result_data"] is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f"Query executed successfully.\nResult: {str(result['result_data'])}", 
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
                result["figure"] = fig
            
            # Capture any printed output
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            if stdout:
                result["description"] += f"\n\nOutput: {stdout}"
                
            if stderr:
                result["description"] += f"\n\nWarnings: {stderr}"
                
    except Exception as e:
        # Handle any exceptions
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        
        # Create a simple error figure
        error_fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error in query:\n{error_msg}", 
                ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        
        result["figure"] = error_fig
        result["error"] = f"{error_msg}\n\n{traceback_str}"
        result["description"] = f"Error in query: {error_msg}"
    
    return result