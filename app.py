import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Financial ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

def show_welcome():
    st.title("💰 Financial Machine Learning Dashboard")
    st.markdown("---")
    
    # Columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://media.giphy.com/media/3ohhwM5Z5IHQkXwZNK/giphy.gif", 
                caption="Stock Market Analysis")
    
    with col2:
        st.markdown("""
        ### 📌 How to Use This App
        1. Upload your dataset **OR** fetch stock data
        2. Follow the ML pipeline steps
        3. Analyze results
        """)
        
    st.success("Ready to begin! Use the sidebar to load data.")

show_welcome()

def show_welcome():
    st.title("💰 Financial Machine Learning Dashboard")
    st.markdown("---")
    
    # Columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://media.giphy.com/media/3ohhwM5Z5IHQkXwZNK/giphy.gif", 
                caption="Stock Market Analysis")
    
    with col2:
        st.markdown("""
        ### 📌 How to Use This App
        1. Upload your dataset **OR** fetch stock data
        2. Follow the ML pipeline steps
        3. Analyze results
        """)
        
    st.success("Ready to begin! Use the sidebar to load data.")

show_welcome()
def load_data():
    st.sidebar.header("📥 Data Input Options")
    
   
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV/Excel File",
        type=["csv", "xlsx"],
        help="Supports Kragle datasets"
    )
    

    st.sidebar.subheader("OR Fetch Live Data")
    ticker = st.sidebar.text_input("Stock Symbol (e.g., AAPL)", "AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
    
    if st.sidebar.button("Fetch Stock Data"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            st.session_state.data = data.reset_index()  # Convert index to column
            st.success(f"Successfully loaded {ticker} data!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    return uploaded_file

uploaded_file = load_data()
def preprocess_data():
    st.header("🧹 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("No data loaded yet!")
        return
    
    df = st.session_state.data
    
  
    with st.expander("View Raw Data"):
        st.dataframe(df.head())
    

    st.subheader("Missing Values Analysis")
    missing = df.isnull().sum()
    st.bar_chart(missing)

    if st.checkbox("Auto-fill missing values?"):
        df.fillna(method='ffill', inplace=True)
        st.session_state.data = df
        st.success("Missing values filled!")
   
    st.subheader("Outlier Detection")
    if st.button("Detect Outliers (IQR Method)"):
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR))]
            st.write(f"Outliers in {col}: {len(outliers)}")
    
    st.session_state.data = df
    def feature_engineering():
    st.header("⚙️ Feature Engineering")
    
    if 'data' not in st.session_state:
        st.warning("Please load data first!")
        return

    df = st.session_state.data
    
    # Example: Add moving averages
    if st.checkbox("Add Moving Averages"):
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        st.success("Added 10-day & 50-day Moving Averages!")

    # Feature selection interface
    st.subheader("Select Features")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = st.multiselect(
        "Choose features for modeling",
        numeric_cols,
        default=['Open', 'High', 'Low', 'Volume']
    )
    
    # Save to session state
    st.session_state.features = selected_features
    st.session_state.target = 'Close'  # Default target
    st.session_state.data = df
    
    st.dataframe(df.head())
    def train_test_split():
    st.header("✂️ Train-Test Split")
    
    if 'features' not in st.session_state:
        st.warning("Do feature engineering first!")
        return

    test_size = st.slider(
        "Test Set Size (%)", 
        min_value=10, 
        max_value=40, 
        value=20
    ) / 100  # Convert to decimal

    X = st.session_state.data[st.session_state.features]
    y = st.session_state.data[st.session_state.target]

    # Perform split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42
    )
    
    # Save to session state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    
    # Visualize
    fig = px.pie(
        names=['Train', 'Test'],
        values=[len(X_train), len(X_test)],
        title="Data Split Ratio"
    )
    st.plotly_chart(fig)
