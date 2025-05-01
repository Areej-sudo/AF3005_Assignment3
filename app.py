import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
from datetime import datetime
import time

# Configure page
st.set_page_config(
    page_title="Financial ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = 'Close'

def show_welcome():
    st.title("💰 Financial Machine Learning Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://media.giphy.com/media/3ohhwM5Z5IHQkXwZNK/giphy.gif", 
                caption="Stock Market Analysis")
    
    with col2:
        st.markdown("""
        ### 📌 How to Use This App
        1. Upload your dataset OR fetch stock data
        2. Follow the ML pipeline steps
        3. Analyze results
        """)

def debug_data_fetch(ticker, start_date, end_date):
    """Debug function to test yfinance connection"""
    try:
        st.write("⌛ Attempting to fetch data...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error("Data fetched but empty. Possible reasons:")
            st.write("- Market was closed on these dates")
            st.write("- Invalid ticker symbol")
            st.write("- Yahoo Finance has no data for this period")
            return False
        
        st.success("✅ Data successfully fetched!")
        st.write("First 5 rows:")
        st.write(data.head())
        return True
        
    except Exception as e:
        st.error(f"❌ Fetch failed: {str(e)}")
        st.write("Possible solutions:")
        st.write("- Check your internet connection")
        st.write("- Try a different ticker (e.g., 'AAPL')")
        st.write("- Try different dates (market open days)")
        return False

def load_data():
    st.sidebar.header("📥 Data Input Options")
    
    # Option 1: File Upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV/Excel File",
        type=["csv", "xlsx"],
        help="Supported formats: CSV, Excel (XLSX)"
    )
    
    # Option 2: Yahoo Finance
    st.sidebar.subheader("OR Fetch Live Data")
    ticker = st.sidebar.text_input("Stock Symbol (e.g., AAPL)", "AAPL").strip().upper()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2023, 12, 31))
    
    if st.sidebar.button("🚀 Fetch Stock Data", help="Click to download market data"):
        if not ticker:
            st.error("Please enter a stock symbol")
            return
            
        with st.spinner(f"Fetching {ticker} data from {start_date} to {end_date}..."):
            try:
                # Debug mode - uncomment to test
                # debug_data_fetch(ticker, start_date, end_date)
                # return
                
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    st.error(f"No data returned for {ticker}. Possible reasons:")
                    st.write("- Market was closed on these dates")
                    st.write("- Invalid ticker symbol")
                    st.write("- Try different dates (e.g., weekdays when market was open)")
                    return
                
                data = data.reset_index()
                data['Date'] = pd.to_datetime(data['Date'])
                data = data.set_index('Date')
                
                # Validate we got actual market data
                if 'Close' not in data.columns:
                    st.error("Unexpected data format received. Columns found:")
                    st.write(data.columns.tolist())
                    return
                
                st.session_state.data = data
                st.success(f"✅ Successfully loaded {ticker} data!")
                st.write(f"Data from {data.index.min().date()} to {data.index.max().date()}")
                st.dataframe(data.head(3))
                
            except Exception as e:
                st.error(f"Failed to fetch data: {str(e)}")
                st.write("Try these solutions:")
                st.write("- Check your internet connection")
                st.write("- Try a popular ticker like 'AAPL' or 'MSFT'")
                st.write("- Ensure dates are valid trading days")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            if 'Date' not in df.columns:
                st.warning("No 'Date' column found - using index as date")
            else:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                st.warning(f"Missing typical market data columns: {', '.join(missing)}")
            
            st.session_state.data = df
            st.success("✅ Dataset loaded successfully!")
            st.dataframe(df.head(3))
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.write("Ensure your file is a valid CSV or Excel file with financial data")

# Rest of your functions (preprocess_data, feature_engineering, etc.) remain the same...

# Main app flow
show_welcome()
load_data()

if st.session_state.data is not None:
    st.write("## Data Preview")
    st.write(f"Data range: {st.session_state.data.index.min()} to {st.session_state.data.index.max()}")
    st.line_chart(st.session_state.data['Close'])
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Preprocessing", 
        "Feature Engineering", 
        "Train-Test Split", 
        "Model Training"
    ])
    
    with tab1:
        preprocess_data()
    
    with tab2:
        feature_engineering()
    
    with tab3:
        perform_train_test_split()
    
    with tab4:
        train_model()
else:
    st.warning("Please load data to begin analysis")
    st.info("""
    Troubleshooting tips:
    1. For Yahoo Finance data:
       - Try popular symbols like AAPL, MSFT, TSLA
       - Use dates between 2010-01-01 and 2023-12-31
       - Ensure dates are weekdays (markets closed weekends)
    2. For file upload:
       - Ensure CSV/Excel format
       - Should contain price data (Open, High, Low, Close)
    """)
