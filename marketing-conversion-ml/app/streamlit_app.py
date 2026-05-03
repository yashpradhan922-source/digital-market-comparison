# streamlit_app.py - Main Streamlit application for the Marketing Conversion ML Dashboard
# Provides an interactive UI for training models, making predictions, and viewing EDA

import streamlit as st  # Import Streamlit for building the web UI
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
import plotly.express as px  # Import plotly express for interactive charts
import plotly.graph_objects as go  # Import plotly graph objects for custom charts
from plotly.subplots import make_subplots  # Import for creating subplot layouts
import os  # Import os for file path operations
import sys  # Import sys for system operations
import json  # Import json for reading model reports

# Add project root to Python path so imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # Add root to path for module resolution

from src.pipelines.training_pipeline import TrainingPipeline  # Import training pipeline
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData  # Import prediction components
from src.utils.common import get_project_root  # Import project root helper

# ============ PAGE CONFIGURATION ============
# Configure the Streamlit page with title, icon, and layout
st.set_page_config(
    page_title="Marketing Conversion Predictor",  # Browser tab title
    page_icon="📊",  # Browser tab icon
    layout="wide",  # Use full-width layout
    initial_sidebar_state="expanded",  # Start with sidebar open
)

# ============ CUSTOM CSS STYLING ============
# Inject custom CSS for a premium dark-themed UI
st.markdown("""
<style>
    /* Import Google Fonts for modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font and background styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header gradient text effect */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        letter-spacing: -0.5px;
    }

    /* Subtitle styling */
    .sub-header {
        font-size: 1.1rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Metric card container with glassmorphism */
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102,126,234,0.2);
    }

    /* Metric value styling */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Metric label styling */
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* Prediction result cards */
    .prediction-positive {
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.15) 100%);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .prediction-negative {
        background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(220,38,38,0.15) 100%);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    /* Section divider styling */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e5e7eb;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102,126,234,0.3);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    /* Info box styling */
    .info-box {
        background: rgba(102,126,234,0.1);
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)  # Allow raw HTML/CSS injection


def load_raw_data():
    """Load the raw dataset for EDA visualizations."""
    # Construct the path to the raw CSV file
    data_path = os.path.join(project_root, "data", "raw", "digital_marketing_campaign_dataset.csv")

    # Check if the processed data exists, otherwise use the original file
    if not os.path.exists(data_path):
        # Try the original location one level up
        data_path = os.path.join(os.path.dirname(project_root), "digital_marketing_campaign_dataset.csv")

    # Read and return the CSV file as a DataFrame
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


def render_eda_page():
    """Render the Exploratory Data Analysis page with interactive charts."""
    # Display section header
    st.markdown('<p class="section-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    # Load the dataset
    df = load_raw_data()

    # Check if data loaded successfully
    if df is None:
        st.error("⚠️ Dataset not found. Please ensure the CSV file is in the data/raw/ directory.")
        return

    # Display dataset overview metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        conv_rate = df["Conversion"].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{conv_rate:.1f}%</div>
            <div class="metric-label">Conversion Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.isnull().sum().sum()}</div>
            <div class="metric-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Conversion Distribution pie chart
        st.markdown("##### 🎯 Conversion Distribution")
        conv_counts = df["Conversion"].value_counts()
        fig_pie = px.pie(
            values=conv_counts.values, names=["Converted", "Not Converted"],
            color_discrete_sequence=["#667eea", "#ef4444"],
            hole=0.4,
        )
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=350,
            font=dict(family="Inter"), margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        # Campaign Channel distribution bar chart
        st.markdown("##### 📡 Campaign Channel Distribution")
        channel_conv = df.groupby("CampaignChannel")["Conversion"].mean().reset_index()
        channel_conv.columns = ["Channel", "Conversion Rate"]
        fig_bar = px.bar(
            channel_conv, x="Channel", y="Conversion Rate",
            color="Conversion Rate",
            color_continuous_scale=["#ef4444", "#667eea", "#10b981"],
        )
        fig_bar.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=350,
            font=dict(family="Inter"), margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Second row of charts
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # Age distribution histogram
        st.markdown("##### 👤 Age Distribution by Conversion")
        fig_hist = px.histogram(
            df, x="Age", color="Conversion", barmode="overlay",
            color_discrete_map={0: "#ef4444", 1: "#667eea"},
            opacity=0.7, nbins=30,
        )
        fig_hist.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=350,
            font=dict(family="Inter"), margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart_col4:
        # Income vs AdSpend scatter plot
        st.markdown("##### 💰 Income vs Ad Spend")
        sample_df = df.sample(min(1000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df, x="Income", y="AdSpend", color="Conversion",
            color_discrete_map={0: "#ef4444", 1: "#667eea"},
            opacity=0.6, size_max=8,
        )
        fig_scatter.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=350,
            font=dict(family="Inter"), margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Third row - Campaign Type and Correlation
    chart_col5, chart_col6 = st.columns(2)

    with chart_col5:
        st.markdown("##### 🏷️ Campaign Type Conversion Rate")
        type_conv = df.groupby("CampaignType")["Conversion"].mean().reset_index()
        type_conv.columns = ["Type", "Rate"]
        fig_type = px.bar(
            type_conv, x="Type", y="Rate", color="Rate",
            color_continuous_scale=["#764ba2", "#667eea", "#10b981"],
        )
        fig_type.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=350,
            font=dict(family="Inter"), margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with chart_col6:
        st.markdown("##### 🔥 Feature Correlation Heatmap")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_cols = [c for c in num_cols if c not in ["CustomerID"]]
        corr_matrix = df[corr_cols].corr()
        fig_heat = px.imshow(
            corr_matrix, color_continuous_scale="RdBu_r",
            aspect="auto", text_auto=".2f",
        )
        fig_heat.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=350,
            font=dict(family="Inter", size=8), margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Show raw data preview
    with st.expander("📋 View Raw Data Sample"):
        st.dataframe(df.head(100), use_container_width=True)


def render_training_page():
    """Render the Model Training page."""
    st.markdown('<p class="section-header">🏋️ Model Training</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Training Pipeline</strong> will execute three steps:<br>
        1️⃣ <b>Data Ingestion</b> — Read CSV & split into train/test<br>
        2️⃣ <b>Data Transformation</b> — Scale numerical & encode categorical features<br>
        3️⃣ <b>Model Training</b> — Train Random Forest, Gradient Boosting, Logistic Regression & XGBoost
    </div>
    """, unsafe_allow_html=True)

    # Training button
    if st.button("🚀 Start Training Pipeline", key="train_btn"):
        with st.spinner("Training in progress... This may take a minute."):
            try:
                pipeline = TrainingPipeline()
                results = pipeline.run_pipeline()
                st.session_state["training_results"] = results
                st.success(f"✅ Training complete! Best model: **{results['best_model']}**")
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

    # Display results if training has been done
    if "training_results" in st.session_state:
        results = st.session_state["training_results"]

        st.markdown("### 🏆 Model Comparison")

        # Show metrics for each model
        model_names = list(results["model_report"].keys())
        cols = st.columns(len(model_names))

        for i, model_name in enumerate(model_names):
            metrics = results["model_report"][model_name]
            is_best = model_name == results["best_model"]
            border_color = "#10b981" if is_best else "rgba(102,126,234,0.2)"

            with cols[i]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.08));
                    border: 2px solid {border_color}; border-radius: 16px; padding: 1.2rem; text-align: center;">
                    <div style="font-weight: 700; font-size: 1rem; margin-bottom: 0.8rem;
                        {'color: #10b981;' if is_best else 'color: #e5e7eb;'}">
                        {"🏆 " if is_best else ""}{model_name}
                    </div>
                    <div style="font-size: 0.85rem; color: #9ca3af;">
                        Accuracy: <b style="color: #667eea;">{metrics['accuracy']}</b><br>
                        Precision: <b style="color: #667eea;">{metrics['precision']}</b><br>
                        Recall: <b style="color: #667eea;">{metrics['recall']}</b><br>
                        F1: <b style="color: #667eea;">{metrics['f1_score']}</b><br>
                        AUC: <b style="color: #667eea;">{metrics.get('roc_auc', 'N/A')}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Bar chart comparing model accuracies
        st.markdown("### 📊 Accuracy Comparison")
        acc_data = pd.DataFrame({
            "Model": model_names,
            "Accuracy": [results["model_report"][m]["accuracy"] for m in model_names],
        })
        fig = px.bar(
            acc_data, x="Model", y="Accuracy", color="Accuracy",
            color_continuous_scale=["#764ba2", "#667eea", "#10b981"],
            text="Accuracy",
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=400,
            font=dict(family="Inter"), yaxis_range=[0, 1.1],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Check for existing model report
    report_path = os.path.join(project_root, "artifacts", "model_report.json")
    if os.path.exists(report_path) and "training_results" not in st.session_state:
        with open(report_path, "r") as f:
            report = json.load(f)
        st.info("📂 Found existing model report from previous training.")
        st.json(report)


def render_prediction_page():
    """Render the Prediction page with input form and results."""
    st.markdown('<p class="section-header">🔮 Conversion Prediction</p>', unsafe_allow_html=True)

    # Check if model exists
    model_path = os.path.join(project_root, "artifacts", "model.pkl")
    if not os.path.exists(model_path):
        st.warning("⚠️ No trained model found. Please train a model first in the Training tab.")
        return

    st.markdown("""
    <div class="info-box">
        Enter customer and campaign details below to predict whether the customer will convert.
    </div>
    """, unsafe_allow_html=True)

    # Create input form with two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 👤 Customer Information")
        age = st.slider("Age", 18, 70, 35, help="Customer age")
        gender = st.selectbox("Gender", ["Male", "Female"])
        income = st.number_input("Annual Income ($)", 20000, 150000, 75000, step=5000)
        previous_purchases = st.slider("Previous Purchases", 0, 10, 3)
        loyalty_points = st.number_input("Loyalty Points", 0, 5000, 2000, step=100)

    with col2:
        st.markdown("##### 📢 Campaign Details")
        campaign_channel = st.selectbox("Campaign Channel", ["Email", "PPC", "SEO", "Social Media", "Referral"])
        campaign_type = st.selectbox("Campaign Type", ["Awareness", "Consideration", "Conversion", "Retention"])
        ad_spend = st.number_input("Ad Spend ($)", 0.0, 10000.0, 3000.0, step=100.0)
        ctr = st.slider("Click Through Rate", 0.0, 0.3, 0.15, step=0.01)
        conversion_rate = st.slider("Conversion Rate", 0.0, 0.2, 0.1, step=0.01)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("##### 🌐 Website Metrics")
        website_visits = st.slider("Website Visits", 0, 50, 20)
        pages_per_visit = st.slider("Pages Per Visit", 1.0, 10.0, 5.0, step=0.5)
        time_on_site = st.slider("Time on Site (min)", 0.0, 15.0, 7.0, step=0.5)

    with col4:
        st.markdown("##### 📱 Engagement Metrics")
        social_shares = st.slider("Social Shares", 0, 100, 30)
        email_opens = st.slider("Email Opens", 0, 20, 8)
        email_clicks = st.slider("Email Clicks", 0, 10, 4)

    st.markdown("---")

    # Predict button
    if st.button("🔮 Predict Conversion", key="predict_btn"):
        with st.spinner("Analyzing..."):
            try:
                # Create CustomData object from form inputs
                data = CustomData(
                    Age=age, Gender=gender, Income=income,
                    CampaignChannel=campaign_channel, CampaignType=campaign_type,
                    AdSpend=ad_spend, ClickThroughRate=ctr, ConversionRate=conversion_rate,
                    WebsiteVisits=website_visits, PagesPerVisit=pages_per_visit,
                    TimeOnSite=time_on_site, SocialShares=social_shares,
                    EmailOpens=email_opens, EmailClicks=email_clicks,
                    PreviousPurchases=previous_purchases, LoyaltyPoints=loyalty_points,
                )

                # Convert to DataFrame
                input_df = data.get_data_as_dataframe()

                # Initialize prediction pipeline and make prediction
                pipeline = PredictionPipeline()
                prediction = pipeline.predict(input_df)
                probabilities = pipeline.predict_proba(input_df)

                # Display prediction result
                result_col1, result_col2 = st.columns([1, 1])

                with result_col1:
                    if prediction[0] == 1:
                        st.markdown("""
                        <div class="prediction-positive">
                            <div style="font-size: 3rem;">✅</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #10b981; margin: 0.5rem 0;">
                                WILL CONVERT
                            </div>
                            <div style="color: #9ca3af;">This customer is likely to convert!</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="prediction-negative">
                            <div style="font-size: 3rem;">❌</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444; margin: 0.5rem 0;">
                                WILL NOT CONVERT
                            </div>
                            <div style="color: #9ca3af;">This customer is unlikely to convert.</div>
                        </div>
                        """, unsafe_allow_html=True)

                with result_col2:
                    # Probability gauge chart
                    conv_prob = probabilities[0][1] * 100
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=conv_prob,
                        title={'text': "Conversion Probability", 'font': {'size': 16, 'color': '#e5e7eb'}},
                        number={'suffix': '%', 'font': {'color': '#667eea', 'size': 40}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#9ca3af'},
                            'bar': {'color': '#667eea'},
                            'bgcolor': 'rgba(0,0,0,0)',
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(239,68,68,0.2)'},
                                {'range': [30, 70], 'color': 'rgba(234,179,8,0.2)'},
                                {'range': [70, 100], 'color': 'rgba(16,185,129,0.2)'},
                            ],
                            'threshold': {
                                'line': {'color': '#10b981', 'width': 3},
                                'thickness': 0.8, 'value': 50,
                            },
                        },
                    ))
                    fig_gauge.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        height=300, font=dict(family="Inter"),
                        margin=dict(t=60, b=20, l=30, r=30),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Show input data summary
                with st.expander("📋 View Input Data"):
                    st.dataframe(input_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")


# ============ MAIN APP LAYOUT ============
def main():
    """Main function to run the Streamlit application."""
    # Render the main header
    st.markdown('<h1 class="main-header">📊 Marketing Conversion Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Digital Marketing Campaign Conversion Analysis & Prediction</p>',
                unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        st.markdown("---")
        page = st.radio(
            "Select a page:",
            ["📊 EDA Dashboard", "🏋️ Train Model", "🔮 Predict"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6b7280; font-size: 0.8rem; padding: 1rem;">
            <b>Marketing Conversion ML</b><br>
            v1.0.0 • Built with Streamlit
        </div>
        """, unsafe_allow_html=True)

    # Render the selected page
    if page == "📊 EDA Dashboard":
        render_eda_page()
    elif page == "🏋️ Train Model":
        render_training_page()
    elif page == "🔮 Predict":
        render_prediction_page()


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
