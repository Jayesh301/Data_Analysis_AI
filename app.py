import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import google.generativeai as genai
import traceback
from dotenv import load_dotenv
from auto_analyzer import auto_analyze_dataset

# Set up Streamlit page - MUST be the first Streamlit command
st.set_page_config(page_title="Data Analysis AI", layout="wide")

# Load environment variables
load_dotenv()

# Set up Gemini API - handle both local and Streamlit Cloud deployment
try:
    # First try to get from Streamlit secrets (for cloud deployment)
    api_key = st.secrets["api_keys"]["GEMINI_API_KEY"]
except:
    # If not in cloud, try to get from environment variables (local development)
    api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("No API key found. Please add your GEMINI_API_KEY to .env file or Streamlit secrets.")

st.title("Data Analysis AI")

# Create session state to store data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'debug' not in st.session_state:
    st.session_state.debug = False
if 'auto_analysis_results' not in st.session_state:
    st.session_state.auto_analysis_results = None
if 'current_viz_index' not in st.session_state:
    st.session_state.current_viz_index = 0

# Add debugging toggle
with st.sidebar:
    st.session_state.debug = st.checkbox("Enable debugging", value=st.session_state.debug)

# File upload section
st.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Check if this is a new file upload
        is_new_upload = False
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            is_new_upload = True
            st.session_state.last_uploaded_file = uploaded_file.name
        
        # Load the data
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        
        st.success(f"Dataset loaded with {st.session_state.df.shape[0]} rows and {st.session_state.df.shape[1]} columns.")
        
        # Show data preview
        with st.expander("Data Preview", expanded=True):
            st.dataframe(st.session_state.df.head())
            
            # Display data info
            if st.session_state.debug:
                st.subheader("Dataset Information")
                buffer = io.StringIO()
                st.session_state.df.info(buf=buffer)
                st.text(buffer.getvalue())
                
                st.subheader("Missing Values")
                st.write(st.session_state.df.isna().sum())
        
        # Auto analyze on new file upload
        if is_new_upload:
            with st.spinner("Automatically analyzing your dataset..."):
                try:
                    st.session_state.auto_analysis_results = auto_analyze_dataset(st.session_state.df)
                    st.session_state.current_viz_index = 0
                except Exception as e:
                    st.error(f"Error during auto-analysis: {str(e)}")
                    if st.session_state.debug:
                        st.error(traceback.format_exc())
        
        # Display auto analysis results
        if st.session_state.auto_analysis_results:
            auto_results = st.session_state.auto_analysis_results
            
            # Display summary stats
            with st.expander("Dataset Summary", expanded=True):
                st.write(f"**Rows:** {auto_results['summary']['shape'][0]}, **Columns:** {auto_results['summary']['shape'][1]}")
                
                # Display column types 
                if 'numeric_columns' in auto_results['summary']:
                    st.write(f"**Numeric columns ({len(auto_results['summary']['numeric_columns'])}):** {', '.join(auto_results['summary']['numeric_columns'])}")
                
                if 'categorical_columns' in auto_results['summary']:
                    st.write(f"**Categorical columns ({len(auto_results['summary']['categorical_columns'])}):** {', '.join(auto_results['summary']['categorical_columns'])}")
                
                if 'date_columns' in auto_results['summary']:
                    st.write(f"**Date columns ({len(auto_results['summary']['date_columns'])}):** {', '.join(auto_results['summary']['date_columns'])}")
                
                # Show missing values
                if sum(auto_results['summary']['missing_values'].values()) > 0:
                    st.write("**Missing Values:**")
                    missing_df = pd.DataFrame({
                        'Column': list(auto_results['summary']['missing_values'].keys()),
                        'Missing Count': list(auto_results['summary']['missing_values'].values()),
                        'Missing Percentage': [f"{val:.2f}%" for val in auto_results['summary']['missing_percentage'].values()]
                    })
                    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                    st.dataframe(missing_df)
            
            # Display insights
            st.header("Automatic Insights")
            if auto_results['insights']:
                st.write(auto_results['insights'])
            
            # Display recommendations
            if auto_results['recommendations']:
                st.subheader("Recommendations for Further Analysis")
                for rec in auto_results['recommendations']:
                    st.write(f"- {rec}")
            
            # Display visualizations with navigation
            if auto_results['visualizations']:
                st.header("Automatic Visualizations")
                
                # Create navigation buttons
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("← Previous"):
                        st.session_state.current_viz_index = (st.session_state.current_viz_index - 1) % len(auto_results['visualizations'])
                
                with col3:
                    if st.button("Next →"):
                        st.session_state.current_viz_index = (st.session_state.current_viz_index + 1) % len(auto_results['visualizations'])
                
                # Display current visualization
                current_viz = auto_results['visualizations'][st.session_state.current_viz_index]
                st.subheader(f"{st.session_state.current_viz_index + 1}/{len(auto_results['visualizations'])}: {current_viz['title']}")
                st.pyplot(current_viz['figure'])
                
                # Show visualization details
                viz_type = current_viz['type']
                if viz_type == 'scatter' and 'correlation' in current_viz:
                    st.write(f"Correlation coefficient: {current_viz['correlation']:.2f}")
                
                # Allow downloading the current figure
                buf = io.BytesIO()
                current_viz['figure'].savefig(buf, format='png')
                buf.seek(0)
                st.download_button(
                    label="Download this visualization",
                    data=buf,
                    file_name=f"{current_viz['title'].replace(' ', '_')}.png",
                    mime="image/png"
                )
            
            # Show error if any
            if auto_results['error']:
                st.error(f"Error during automatic analysis: {auto_results['error']}")
                
    except Exception as e:
        st.error(f"Error loading file: {e}")
        if st.session_state.debug:
            st.error(traceback.format_exc())

# Data analysis section for user questions
if st.session_state.df is not None:
    st.header("Ask Questions About Your Data")
    user_question = st.text_input("Enter your question about the data:")
    
    if st.button("Analyze"):
        if user_question:
            with st.spinner("Analyzing your data..."):
                try:
                    # Save the question to history
                    st.session_state.query_history.append(user_question)
                    
                    # Process the question
                    from nlp_interpreter import process_query
                    result = process_query(user_question, st.session_state.df)
                    
                    # Display results
                    st.subheader("Results")
                    st.write(result["insight"])
                    
                    if "figure" in result and result["figure"] is not None:
                        st.pyplot(result["figure"])
                    else:
                        st.warning("No visualization was generated.")
                    
                    # Show the query if in debug mode
                    if st.session_state.debug:
                        st.subheader("Generated Pandas Query")
                        if "pandas_query" in result.get("debug_info", {}):
                            st.code(result["debug_info"]["pandas_query"], language="python")
                        
                        # Show result data if available
                        if "result_data" in result and result["result_data"] is not None:
                            st.subheader("Query Result")
                            if hasattr(result["result_data"], "shape"):  # DataFrame or Series
                                st.dataframe(result["result_data"])
                            else:
                                st.write(result["result_data"])
                        
                        # Show debugging info if enabled
                        if "debug_info" in result:
                            st.subheader("Debug Information")
                            st.json(result["debug_info"])
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    if st.session_state.debug:
                        st.error(traceback.format_exc())
                        
                        # Create a simple fallback visualization
                        fig, ax = plt.subplots()
                        ax.text(0.5, 0.5, f"Error in analysis: {str(e)}", 
                                ha='center', va='center', fontsize=12, color='red')
                        ax.axis('off')
                        st.pyplot(fig)
        else:
            st.warning("Please enter a question to analyze your data.")

# Display history
if st.session_state.query_history:
    st.sidebar.header("Query History")
    for i, query in enumerate(st.session_state.query_history):
        st.sidebar.text(f"{i+1}. {query}") 