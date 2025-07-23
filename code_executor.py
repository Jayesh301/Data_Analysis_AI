import pandas as pd
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from utils import make_plot

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
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            query = query_string.strip()
            needs_viz = any(viz_term in query.lower() for viz_term in 
                           ['plot(', '.plot.', '.hist(', '.boxplot(', '.scatter(', '.bar(', '.barh(', '.pie('])
            query_result = eval(query)
            result["result_data"] = query_result
            # Use make_plot for DataFrame/Series visualizations
            if needs_viz:
                import matplotlib.pyplot as plt
                result["figure"] = plt.gcf()
            elif isinstance(query_result, pd.DataFrame) and not query_result.empty:
                cols = list(query_result.columns)
                fig = None
                if len(cols) >= 2:
                    fig = make_plot(query_result, cols[0], cols[1], title=f"{cols[0]} vs {cols[1]}")
                elif len(cols) == 1:
                    fig = make_plot(query_result, cols[0], title=f"Distribution of {cols[0]}")
                else:
                    fig = make_plot(query_result, None, title="Table of Results")
                result["figure"] = fig
                result["description"] = f"Analysis returned {query_result.shape[0]} rows of data."
            elif isinstance(query_result, pd.Series):
                name = query_result.name if hasattr(query_result, 'name') and query_result.name else "Result"
                df = query_result.to_frame()
                fig = make_plot(df, name, title=f"Distribution of {name}")
                result["figure"] = fig
            if result["figure"] is None and result["result_data"] is not None:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f"Query executed successfully.\nResult: {str(result['result_data'])}", 
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
                result["figure"] = fig
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            if stdout:
                result["description"] += f"\n\nOutput: {stdout}"
            if stderr:
                result["description"] += f"\n\nWarnings: {stderr}"
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        import matplotlib.pyplot as plt
        error_fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error in query:\n{error_msg}", 
                ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        result["figure"] = error_fig
        result["error"] = f"{error_msg}\n\n{traceback_str}"
        result["description"] = f"Error in query: {error_msg}"
    return result