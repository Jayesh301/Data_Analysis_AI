# Data Analysis AI

An autonomous data analysis and visualization agent powered by Gemini 1.5 Flash.

## Features

- Automatic dataset analysis upon upload (no need to ask questions first!)
- Smart visualization selection based on data types and patterns
- Automatic insights generation and recommendations
- Upload CSV or Excel files
- Ask questions about your data in natural language
- Get instant visualizations and insights
- Track your query history

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                      DATA ANALYSIS AI ARCHITECTURE                     │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                         │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐    │
│  │ File Uploader│      │ Auto-Analysis │      │ Query Interface  │    │
│  │ (CSV/Excel)  │─────▶│   Results     │      │                  │    │
│  └──────────────┘      └──────────────┘      └──────────────────┘    │
│                                │                       │              │
│                                ▼                       ▼              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐    │
│  │Data Preview  │      │Visualizations│      │Custom Analysis    │    │
│  │              │      │& Navigation  │      │Results            │    │
│  └──────────────┘      └──────────────┘      └──────────────────┘    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                           ANALYSIS LAYER                               │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────┐          ┌─────────────────────────┐    │
│  │   AUTOMATIC ANALYSIS    │          │    NLP INTERPRETER      │    │
│  │  (auto_analyzer.py)     │          │  (nlp_interpreter.py)   │    │
│  ├─────────────────────────┤          ├─────────────────────────┤    │
│  │ - Summary Statistics    │          │ - Natural Language      │    │
│  │ - Data Type Detection   │          │   Query Processing      │    │
│  │ - Visualization Creation │          │ - Query to Pandas Code  │    │
│  │ - Correlation Analysis  │          │   Translation           │    │
│  │ - Time Series Detection │          │ - Fallback Analysis     │    │
│  │ - Insight Generation    │          │                         │    │
│  └─────────────────────────┘          └─────────────────────────┘    │
│                  │                                   │                │
│                  ▼                                   ▼                │
│  ┌─────────────────────────┐                                          │
│  │    CODE EXECUTOR        │                                          │
│  │   (code_executor.py)    │                                          │
│  ├─────────────────────────┤                                          │
│  │ - Safe Query Execution  │                                          │
│  │ - Results Formatting    │                                          │
│  │ - Error Handling        │                                          │
│  └─────────────────────────┘                                          │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                           AI SERVICES LAYER                            │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────┐          ┌─────────────────────────┐    │
│  │  GEMINI AI (LLM)        │          │  DATA VISUALIZATION     │    │
│  │                         │          │                         │    │
│  ├─────────────────────────┤          ├─────────────────────────┤    │
│  │ - Query Translation     │          │ - Matplotlib            │    │
│  │ - Insight Generation    │          │ - Seaborn               │    │
│  │ - Recommendations       │          │ - Interactive Charts     │    │
│  └─────────────────────────┘          └─────────────────────────┘    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT LAYER                             │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────┐          ┌─────────────────────────┐    │
│  │  STREAMLIT CLOUD        │          │  ALTERNATIVE OPTIONS    │    │
│  │                         │          │                         │    │
│  ├─────────────────────────┤          ├─────────────────────────┤    │
│  │ - GitHub Integration    │          │ - Heroku                │    │
│  │ - Environment Setup     │          │ - Render                │    │
│  │ - API Key Security      │          │ - Railway               │    │
│  │ - Free Tier Hosting     │          │ - Local Deployment      │    │
│  └─────────────────────────┘          └─────────────────────────┘    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. User Interface Layer

- **File Uploader**: Accepts CSV/Excel files from users
- **Auto-Analysis Results**: Displays automatic insights and visualizations
- **Query Interface**: Allows users to ask natural language questions

### 2. Analysis Layer

- **Automatic Analysis** (auto_analyzer.py): Performs data type detection, generates visualizations
- **NLP Interpreter** (nlp_interpreter.py): Processes natural language queries
- **Code Executor** (code_executor.py): Safely executes generated code

### 3. AI Services Layer

- **Gemini AI (LLM)**: Natural language understanding and insight generation
- **Data Visualization**: Chart generation using Matplotlib/Seaborn

### 4. Deployment Layer

- **Streamlit Cloud**: Primary deployment platform with GitHub integration
- **Alternative Options**: Additional deployment paths

## Setup for Local Development

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Deployment on Streamlit Community Cloud

Follow these steps to deploy the application for free:

1. Fork or clone this repository to your GitHub account
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch (main), and the main file path (app.py)
6. Expand "Advanced settings"
7. Add your Gemini API key as a secret:
   - Add a key: `api_keys.GEMINI_API_KEY`
   - Set the value to your actual API key
8. Click "Deploy"
9. Your app will be deployed with a shareable URL

## Requirements

- Python 3.9+
- Gemini API key (get one at https://ai.google.dev/)

## Usage

1. Upload your dataset (CSV or Excel)
2. The system will automatically analyze your data and present:
   - Dataset summary statistics
   - Data quality information
   - Automatically selected visualizations
   - Key insights and analysis recommendations
3. Ask follow-up questions to customize or further explore your data
4. Navigate through the auto-generated visualizations

## Example Questions

- "Show the monthly sales trend"
- "Which region has the highest revenue?"
- "What's the correlation between price and quantity sold?"
- "Show me the top 5 products by revenue"
- "Create a scatter plot of price vs rating"
- "Show distribution of ages by gender"

## Note on Date Format Warnings

When running the application, you may see warnings like:

```
UserWarning: Could not infer format, so each element will be parsed individually,
falling back to `dateutil`. To ensure parsing is consistent and as-expected,
please specify a format.
```

These warnings are expected when the application attempts to detect date columns in the dataset. They don't affect functionality and can be safely ignored.
