import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
            st.header("📊 Dataset Summary")
            
            # Show cleaning information
            if 'original_shape' in auto_results['summary']:
                original_rows, original_cols = auto_results['summary']['original_shape']
                current_rows, current_cols = auto_results['summary']['shape']
                rows_removed = auto_results['summary'].get('rows_removed', 0)
                cols_removed = auto_results['summary'].get('columns_removed', 0)
                
                st.write("### 🧹 Data Cleaning Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Size", f"{original_rows:,} rows × {original_cols} cols")
                with col2:
                    st.metric("Final Size", f"{current_rows:,} rows × {current_cols} cols")
                
                # Show detailed cleaning information
                duplicates_removed = auto_results['summary'].get('duplicates_removed', 0)
                missing_filled = auto_results['summary'].get('missing_values_filled', 0)
                cleaning_actions = []
                
                if rows_removed > 0:
                    cleaning_actions.append(f"🗑️ Removed {rows_removed:,} empty rows")
                if cols_removed > 0:
                    cleaning_actions.append(f"🗑️ Removed {cols_removed} empty columns")
                if duplicates_removed > 0:
                    cleaning_actions.append(f"🔄 Removed {duplicates_removed:,} duplicate rows")
                if missing_filled > 0:
                    cleaning_actions.append(f"🔧 Filled {missing_filled:,} missing values (forward/backward fill + mode)")
                
                if cleaning_actions:
                    st.success("✅ **Cleaning Actions Performed:**")
                    for action in cleaning_actions:
                        st.write(f"  {action}")
                else:
                    st.info("ℹ️ No cleaning needed - dataset was already clean!")
                

            
            # Current dataset info
            st.write("### 📊 Current Dataset Info")
            st.write(f"**Rows:** {auto_results['summary']['shape'][0]:,}, **Columns:** {auto_results['summary']['shape'][1]}")
            
            # Display column types 
            if 'numeric_columns' in auto_results['summary']:
                st.write(f"**Numeric columns ({len(auto_results['summary']['numeric_columns'])}):** {', '.join(auto_results['summary']['numeric_columns'])}")
            
            if 'categorical_columns' in auto_results['summary']:
                st.write(f"**Categorical columns ({len(auto_results['summary']['categorical_columns'])}):** {', '.join(auto_results['summary']['categorical_columns'])}")
            
            if 'date_columns' in auto_results['summary']:
                st.write(f"**Date columns ({len(auto_results['summary']['date_columns'])}):** {', '.join(auto_results['summary']['date_columns'])}")
            
            # Check for any remaining missing values (should be none after cleaning)
            total_missing = sum(auto_results['summary']['missing_values'].values())
            if total_missing > 0:
                st.write("### ⚠️ Data Quality Check")
                missing_df = pd.DataFrame({
                    'Column': list(auto_results['summary']['missing_values'].keys()),
                    'Missing Count': list(auto_results['summary']['missing_values'].values()),
                    'Missing Percentage': [f"{val:.2f}%" for val in auto_results['summary']['missing_percentage'].values()]
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                if len(missing_df) > 0:
                    st.dataframe(missing_df)
                    st.warning("⚠️ Some missing values could not be filled automatically. This may affect analysis quality.")
            else:
                st.success("✅ Dataset is completely clean - no missing values!")
            
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
                fig = current_viz['figure']
                if isinstance(fig, go.Figure):
                    st.plotly_chart(fig, use_container_width=True)
                elif isinstance(fig, plt.Figure):
                    st.pyplot(fig)
                else:
                    st.warning(f"Unsupported figure type: {type(fig)}")
                
                # Show visualization details
                viz_type = current_viz['type']
                if viz_type == 'scatter' and 'correlation' in current_viz:
                    st.write(f"Correlation coefficient: {current_viz['correlation']:.2f}")
                
                # Allow downloading the current figure
                fig_to_download = current_viz['figure']
                buf = io.BytesIO()
                
                if isinstance(fig_to_download, go.Figure):
                    fig_to_download.write_image(buf, format="png")
                elif isinstance(fig_to_download, plt.Figure):
                    fig_to_download.savefig(buf, format="png")
                
                if buf.tell() > 0:
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

# Custom Analysis Chat Interface
if st.session_state.df is not None:
    st.header("💬 Custom Analysis Chat")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input("Ask for custom charts or analysis:", 
                                    placeholder="e.g., 'Show correlation between price and rating', 'Create a box plot of age by gender'")
    with col2:
        analyze_button = st.button("🔍 Analyze", type="primary")
    
    # Display chat history
    if st.session_state.chat_history:
        st.write("### 📝 Chat History")
        for i, (question, result) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:50]}...", expanded=False):
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {result['insight']}")
                
                if "figure" in result and result["figure"] is not None:
                    fig = result["figure"]
                    if isinstance(fig, go.Figure):
                        st.plotly_chart(fig, use_container_width=True)
                    elif isinstance(fig, plt.Figure):
                        st.pyplot(fig)
                    else:
                        st.warning(f"Unsupported figure type: {type(fig)}")
    
    # Process new question
    if analyze_button and user_question:
        with st.spinner("Creating your custom analysis..."):
            try:
                # Process the question
                from nlp_interpreter import process_query
                result = process_query(user_question, st.session_state.df)
                
                # Add to chat history
                st.session_state.chat_history.append((user_question, result))
                
                # Display current result
                st.success("✅ Analysis completed!")
                st.write(f"**Your Question:** {user_question}")
                st.write(f"**Answer:** {result['insight']}")
                
                if "figure" in result and result["figure"] is not None:
                    fig = result["figure"]
                    if isinstance(fig, go.Figure):
                        st.plotly_chart(fig, use_container_width=True)
                    elif isinstance(fig, plt.Figure):
                        st.pyplot(fig)
                    else:
                        st.warning(f"Unsupported figure type: {type(fig)}")
                else:
                    st.info("No visualization was generated for this query.")
                
                # Show debug info if enabled
                if st.session_state.debug:
                    with st.expander("🔧 Debug Information", expanded=False):
                        if "pandas_query" in result.get("debug_info", {}):
                            st.code(result["debug_info"]["pandas_query"], language="python")
                        
                        if "result_data" in result and result["result_data"] is not None:
                            st.subheader("Query Result Data")
                            if hasattr(result["result_data"], "shape"):
                                st.dataframe(result["result_data"])
                            else:
                                st.write(result["result_data"])
                        
                        if "debug_info" in result:
                            st.json(result["debug_info"])
                        
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                if st.session_state.debug:
                    st.error(traceback.format_exc())
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Display history
if st.session_state.query_history:
    st.sidebar.header("Query History")
    for i, query in enumerate(st.session_state.query_history):
        st.sidebar.text(f"{i+1}. {query}") 