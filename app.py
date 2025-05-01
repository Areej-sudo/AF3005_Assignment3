import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Configure page
st.set_page_config(
    page_title="Financial ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with SAMPLE DATA
if 'data' not in st.session_state:
    # Create sample stock data if no data loaded
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq='D')
    prices = np.cumsum(np.random.randn(len(dates)) * 0.01 + 100)
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': prices + np.random.rand(len(dates)),
        'Low': prices - np.random.rand(len(dates)),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, len(dates))
    }).set_index('Date')
    st.session_state.data = sample_data
    st.session_state.use_sample_data = True

if 'features' not in st.session_state:
    st.session_state.features = ['Open', 'High', 'Low', 'Volume']
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
        1. Data is pre-loaded with sample stock data
        2. Follow the ML pipeline steps below
        3. Analyze results
        """)
    
    if st.session_state.get('use_sample_data', False):
        st.info("⚠️ Using sample data. Try loading your own data via the sidebar.")

def load_data():
    st.sidebar.header("📥 Data Input Options")
    
    # Option 1: File Upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV/Excel File",
        type=["csv", "xlsx"],
        help="Supported formats: CSV, Excel (XLSX)"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            st.session_state.data = df
            st.session_state.use_sample_data = False
            st.sidebar.success("✅ Dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")

def show_data_preview():
    st.header("📊 Data Preview")
    st.write(st.session_state.data.head())
    st.line_chart(st.session_state.data['Close'])

def preprocess_data():
    st.header("🧹 Data Preprocessing")
    df = st.session_state.data.copy()
    
    st.subheader("Missing Values")
    if df.isnull().sum().sum() > 0:
        df.fillna(method='ffill', inplace=True)
        st.session_state.data = df
        st.success("Filled missing values!")
    else:
        st.info("No missing values found")
    
    st.subheader("Outliers")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR))]
        st.write(f"{col}: {len(outliers)} outliers")

def feature_engineering():
    st.header("⚙️ Feature Engineering")
    df = st.session_state.data.copy()
    
    st.subheader("Technical Indicators")
    if st.checkbox("Add Moving Averages"):
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        st.success("Added moving averages!")
    
    st.subheader("Select Features")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = st.multiselect(
        "Choose features for modeling",
        numeric_cols,
        default=['Open', 'High', 'Low', 'Volume']
    )
    
    st.session_state.features = selected_features
    st.session_state.data = df
    st.dataframe(df.head())

def train_test():
    st.header("✂️ Train-Test Split")
    df = st.session_state.data
    
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    X = df[st.session_state.features]
    y = df[st.session_state.target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    
    st.success(f"Split complete! Train: {len(X_train)}, Test: {len(X_test)}")
    fig = px.pie(names=['Train', 'Test'], values=[len(X_train), len(X_test)])
    st.plotly_chart(fig)

def train_model():
    st.header("🤖 Model Training")
    
    if st.button("Train Model"):
        model = LinearRegression()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        
        train_score = model.score(st.session_state.X_train, st.session_state.y_train)
        test_score = model.score(st.session_state.X_test, st.session_state.y_test)
        
        st.success(f"""
        Model trained successfully!
        - Training R²: {train_score:.2f}
        - Testing R²: {test_score:.2f}
        """)
        
        preds = model.predict(st.session_state.X_test)
        fig = px.scatter(x=st.session_state.y_test, y=preds, 
                         labels={'x': 'Actual', 'y': 'Predicted'})
        fig.add_shape(type='line', x0=min(st.session_state.y_test), y0=min(preds),
                     x1=max(st.session_state.y_test), y1=max(preds))
        st.plotly_chart(fig)

# Main app flow
show_welcome()
load_data()
show_data_preview()

tab1, tab2, tab3, tab4 = st.tabs([
    "Preprocessing", "Feature Engineering", 
    "Train-Test Split", "Model Training"
])

with tab1:
    preprocess_data()
with tab2:
    feature_engineering()
with tab3:
    train_test()
with tab4:
    train_model()
