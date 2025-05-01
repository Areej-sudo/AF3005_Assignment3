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

def load_data():
    st.sidebar.header("📥 Data Input Options")
    
    # Option 1: File Upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV/Excel File",
        type=["csv", "xlsx"]
    )
    
    # Option 2: Yahoo Finance
    st.sidebar.subheader("OR Fetch Live Data")
    ticker = st.sidebar.text_input("Stock Symbol (e.g., AAPL)", "AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
    
    if st.sidebar.button("Fetch Stock Data"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                data = data.reset_index()
                data['Date'] = pd.to_datetime(data['Date'])
                st.session_state.data = data.set_index('Date')
                st.success(f"Successfully loaded {ticker} data!")
            else:
                st.error("No data returned for this ticker/date range")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            st.session_state.data = df
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def preprocess_data():
    st.header("🧹 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("No data loaded yet!")
        return
    
    df = st.session_state.data.copy()
    
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

def feature_engineering():
    st.header("⚙️ Feature Engineering")
    
    if st.session_state.data is None:
        st.warning("Please load data first!")
        return
    
    df = st.session_state.data.copy()
    
    st.subheader("Technical Indicators")
    if st.checkbox("Add Moving Averages"):
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        st.success("Added 10-day & 50-day Moving Averages!")
    
    st.subheader("Feature Selection")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = st.multiselect(
        "Choose features for modeling",
        numeric_cols,
        default=['Open', 'High', 'Low', 'Volume']
    )
    
    st.session_state.features = selected_features
    st.session_state.data = df
    
    st.dataframe(df.head())

def perform_train_test_split():
    st.header("✂️ Train-Test Split")
    
    if st.session_state.data is None:
        st.warning("Please load data first!")
        return
        
    if not st.session_state.features:
        st.warning("Please select features in Feature Engineering step!")
        return

    df = st.session_state.data
    
    # Data validation
    if len(df) < 10:
        st.error(f"Not enough data! Only {len(df)} rows available. Need at least 10.")
        return
    
    try:
        X = df[st.session_state.features].apply(pd.to_numeric)
        y = pd.to_numeric(df[st.session_state.target])
    except Exception as e:
        st.error(f"Non-numeric data detected: {str(e)}")
        return
    
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=42
        )
    except ValueError as e:
        st.error(f"Split failed: {str(e)}")
        return
    
    st.session_state.update({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    })
    
    st.success(f"Split successful! Train: {len(X_train)}, Test: {len(X_test)}")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(df, y=st.session_state.target, title="Price Trend")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.pie(names=['Train', 'Test'], 
                     values=[len(X_train), len(X_test)],
                     title="Split Ratio")
        st.plotly_chart(fig2, use_container_width=True)

def train_model():
    st.header("🤖 Model Training")
    
    if not all(key in st.session_state for key in ['X_train', 'y_train']):
        st.warning("Please complete train-test split first!")
        return
    
    if st.button("Train Linear Regression Model"):
        try:
            model = LinearRegression()
            model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.model = model
            
            # Calculate R-squared score
            train_score = model.score(st.session_state.X_train, st.session_state.y_train)
            test_score = model.score(st.session_state.X_test, st.session_state.y_test)
            
            st.success(f"""
            Model trained successfully!
            - Training R²: {train_score:.2f}
            - Testing R²: {test_score:.2f}
            """)
            
            # Visualize predictions vs actual
            fig = px.scatter(
                x=st.session_state.y_test,
                y=model.predict(st.session_state.X_test),
                labels={'x': 'Actual', 'y': 'Predicted'},
                title="Actual vs Predicted Values"
            )
            fig.add_shape(type='line', x0=min(st.session_state.y_test), y0=min(st.session_state.y_test),
                         x1=max(st.session_state.y_test), y1=max(st.session_state.y_test))
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

# Main app flow
show_welcome()
load_data()

if st.session_state.data is not None:
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
