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
