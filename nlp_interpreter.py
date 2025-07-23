import google.generativeai as genai
import pandas as pd
import traceback
from code_executor import execute_query
from utils import classify_columns, make_plot

def process_query(question, dataframe):
    debug_info = {
        "question": question,
        "dataframe_shape": dataframe.shape,
        "dataframe_columns": list(dataframe.columns),
        "steps": []
    }

    try:
        debug_info["steps"].append("Started query generation")
        column_meta = classify_columns(dataframe)
        pandas_query = generate_pandas_query(question, dataframe, column_meta)
        debug_info["pandas_query"] = pandas_query
        debug_info["steps"].append("Completed query generation")

        debug_info["steps"].append("Started query execution")
        execution_result = execute_query(pandas_query, dataframe)
        debug_info["steps"].append("Completed query execution")

        if "error" in execution_result and execution_result["error"]:
            debug_info["execution_error"] = execution_result["error"]
            debug_info["steps"].append("Query execution failed, attempting simplified analysis")
            return simplified_analysis(question, dataframe, debug_info)

        debug_info["steps"].append("Started insight generation")
        insight = generate_insight(question, execution_result, dataframe, column_meta)
        debug_info["insight"] = insight
        debug_info["steps"].append("Completed insight generation")

        execution_result["debug_info"] = debug_info
        execution_result["insight"] = insight

        return execution_result

    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        debug_info["error"] = error_msg
        debug_info["traceback"] = error_traceback
        debug_info["steps"].append(f"Error occurred: {error_msg}")
        return simplified_analysis(question, dataframe, debug_info)

def simplified_analysis(question, dataframe, debug_info):
    """
    Perform a simplified analysis when the main pipeline fails
    
    Args:
        question (str): The user's question
        dataframe (pd.DataFrame): The dataframe to analyze
        debug_info (dict): Debug information collected so far
    
    Returns:
        dict: A simple result with basic insights and visualization
    """
    debug_info["steps"].append("Started simplified analysis")
    
    try:
        meta = classify_columns(dataframe)
        fig = None
        if len(dataframe.columns) >= 2:
            numeric_cols = [col for col, typ in meta.items() if typ == "numeric"]
            if len(numeric_cols) >= 2:
                fig = make_plot(dataframe, numeric_cols[0], numeric_cols[1], title=f"Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}")
                debug_info["simplified_plot"] = f"Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}"
            elif len(numeric_cols) == 1:
                fig = make_plot(dataframe, numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
                debug_info["simplified_plot"] = f"Histogram of {numeric_cols[0]}"
            else:
                cat_cols = [col for col, typ in meta.items() if typ == "categorical"]
                if cat_cols:
                    fig = make_plot(dataframe, cat_cols[0], title=f"Top values in {cat_cols[0]}")
                    debug_info["simplified_plot"] = f"Bar chart of {cat_cols[0]}"
                else:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.text(0.5, 0.5, "Unable to create visualization - no suitable columns found", ha='center', va='center')
                    debug_info["simplified_plot"] = "No visualization possible"
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Not enough columns for visualization", ha='center', va='center')
            debug_info["simplified_plot"] = "Not enough columns"
        insight = f"I couldn't perform the specific analysis you asked for: '{question}' due to technical issues. "
        insight += "Here's a basic summary of your data:\n\n"
        insight += f"- Your dataset has {dataframe.shape[0]} rows and {dataframe.shape[1]} columns.\n"
        missing_values = dataframe.isna().sum()
        if missing_values.sum() > 0:
            insight += f"- There are {missing_values.sum()} missing values in the dataset.\n"
            top_missing = missing_values[missing_values > 0].sort_values(ascending=False).head(3)
            if not top_missing.empty:
                insight += "- Columns with most missing values: "
                insight += ", ".join([f"{col} ({val})" for col, val in top_missing.items()])
                insight += "\n"
        numeric_cols = [col for col, typ in meta.items() if typ == "numeric"]
        if numeric_cols:
            insight += f"- There are {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:5])}"
            if len(numeric_cols) > 5:
                insight += f" and {len(numeric_cols) - 5} more."
            insight += "\n"
        debug_info["steps"].append("Completed simplified analysis")
        return {
            "insight": insight,
            "figure": fig,
            "debug_info": debug_info
        }
    except Exception as e:
        import matplotlib.pyplot as plt
        error_msg = str(e)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Unable to analyze data: {error_msg}", ha='center', va='center')
        ax.axis('off')
        debug_info["simplified_analysis_error"] = error_msg
        debug_info["steps"].append(f"Simplified analysis failed: {error_msg}")
        return {
            "insight": f"I couldn't analyze your question: '{question}' due to technical issues: {error_msg}",
            "figure": fig,
            "debug_info": debug_info
        }

def generate_pandas_query(question, dataframe, column_meta=None):
    """
    Use Gemini to generate a pandas query string based on the user's question
    
    Args:
        question (str): The user's question about the data
        dataframe (pd.DataFrame): The DataFrame to query
        
    Returns:
        str: A pandas query that answers the question
    """
    data_info = {
        "columns": list(dataframe.columns),
        "dtypes": {col: str(dataframe[col].dtype) for col in dataframe.columns},
        "sample_data": dataframe.head(3).to_dict(orient="records"),
        "missing_values": dataframe.isna().sum().to_dict(),
        "shape": dataframe.shape,
        "column_meta": column_meta if column_meta else {}
    }
    prompt = f"""
    You are a pandas expert who generates precise query strings for data analysis.
    
    I have a pandas DataFrame with the following information:
    Columns: {data_info['columns']}
    Data Types: {data_info['dtypes']}
    Column Types: {data_info['column_meta']}
    Sample Data: {data_info['sample_data']}
    Missing Values: {data_info['missing_values']}
    Shape: {data_info['shape']}
    
    The user asks: "{question}"
    
    Generate a pandas query STRING that answers this question. The query should:
    1. Be a valid pandas operation on a DataFrame called "df"
    2. Handle any necessary data cleaning (like handling missing values or data type conversions)
    3. Perform the required analysis (filtering, grouping, aggregating, etc.)
    4. Format the results appropriately (such as sorting or limiting results)
    
    Return ONLY the query string without any explanations or code blocks. 
    The query should be a valid pandas expression that can be evaluated directly.
    
    Examples:
    - For "What are the top 5 sales regions?": df.groupby('Region')['Sales'].sum().sort_values(ascending=False).head(5)
    - For "Average age by gender": df.groupby('Gender')['Age'].mean()
    - For "Sales trend by month": df.groupby(pd.Grouper(key='Date', freq='M'))['Sales'].sum()
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    # Clean the response
    query = response.text.strip()
    if query.startswith('```python'):
        query = query[10:]
    if query.startswith('```'):
        query = query[3:]
    if query.endswith('```'):
        query = query[:-3]
    
    return query.strip()

def generate_insight(question, execution_result, dataframe, column_meta=None):
    """
    Use Gemini to generate insights based on the analysis results
    """
    # Create a description of the result for Gemini
    result_description = "The analysis was completed successfully."
    if "description" in execution_result:
        result_description = execution_result["description"]
    
    # Include numerical results if available
    result_data = "No numerical results were generated."
    if "result_data" in execution_result and execution_result["result_data"] is not None:
        if hasattr(execution_result["result_data"], "to_dict"):
            result_data = str(execution_result["result_data"].head(10).to_dict())
        else:
            result_data = str(execution_result["result_data"])
    
    # Include error information if there was an error
    error_info = ""
    if "error" in execution_result and execution_result["error"]:
        error_info = f"There was an error during analysis: {execution_result['error']}"
    
    prompt = f"""
    You are a data analysis expert who explains insights clearly to non-technical users.
    
    The user asked: "{question}"
    
    Based on the data analysis:
    
    Query result: {result_data}
    Description: {result_description}
    {error_info}
    Column Types: {column_meta}
    
    Provide a concise, insightful explanation of what the results mean in 2-3 sentences. 
    Focus on the most important patterns or findings. Use simple language that a business user would understand.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    return response.text.strip()
