# 📊 Marketing Conversion ML

AI-powered machine learning pipeline for predicting digital marketing campaign conversions with an interactive Streamlit dashboard.

> **🌐 Live App URL:** [http://localhost:8501](http://localhost:8501)

---

## 🏗️ Project Architecture

```
marketing-conversion-ml/
│
├── artifacts/                  # Saved outputs (trained model, preprocessor, report)
│   ├── model.pkl               # Best trained model (Gradient Boosting)
│   ├── preprocessor.pkl        # Fitted preprocessing pipeline (scaler + encoder)
│   └── model_report.json       # Evaluation metrics for all models
│
├── data/                       # Data storage
│   ├── raw/                    # Original CSV dataset
│   ├── processed/              # Train/test split CSVs
│   └── external/               # External datasets (optional)
│
├── notebooks/                  # Jupyter notebooks (EDA, experiments)
│
├── src/                        # Core source code
│   ├── __init__.py
│   ├── components/             # ML pipeline components
│   │   ├── data_ingestion.py       # Reads CSV, splits into train/test
│   │   ├── data_transformation.py  # Scales numerical, encodes categorical features
│   │   └── model_trainer.py        # Trains 4 models, selects the best one
│   ├── pipelines/              # Training & prediction pipelines
│   │   ├── training_pipeline.py    # Orchestrates end-to-end training
│   │   └── prediction_pipeline.py  # Loads model & makes predictions
│   ├── utils/                  # Helper functions
│   │   ├── common.py               # Save/load objects, read YAML
│   │   ├── logger.py               # Timestamped logging to files
│   │   └── exception.py            # Custom exception with file/line details
│   └── config/                 # Configuration files
│       ├── config.yaml             # Paths, feature columns, settings
│       └── params.yaml             # Model hyperparameters
│
├── app/                        # Streamlit application
│   └── streamlit_app.py            # Main UI (EDA, Training, Prediction)
│
├── tests/                      # Unit tests
│   └── test_pipeline.py
│
├── .github/workflows/ci.yml   # GitHub Actions CI/CD
├── Dockerfile                  # Container setup
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── .env                        # Environment variables
```

---

## 🚀 Step-by-Step Setup Guide

### Step 1: Clone or Navigate to the Project

```bash
cd marketing-conversion-ml
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`, `plotly`, `pyyaml`, `python-dotenv`

### Step 4: Place the Dataset

Ensure the CSV file is in the `data/raw/` directory:

```bash
# If not already there, copy it
cp digital_marketing_campaign_dataset.csv data/raw/
```

The dataset contains **8,001 records** with 20 columns including demographics, campaign details, engagement metrics, and the binary target `Conversion`.

### Step 5: Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will start and open in your browser at:

> **🌐 http://localhost:8501**

---

## 📖 How to Use the App (Step-by-Step)

### 📊 Page 1 — EDA Dashboard

1. Open the app at [http://localhost:8501](http://localhost:8501)
2. The **EDA Dashboard** loads by default
3. View key metrics: Total Records, Features, Conversion Rate, Missing Values
4. Explore interactive charts:
   - **Conversion Distribution** — Pie chart showing converted vs not converted
   - **Campaign Channel Distribution** — Bar chart of conversion rates by channel
   - **Age Distribution** — Histogram colored by conversion status
   - **Income vs Ad Spend** — Scatter plot of the relationship
   - **Campaign Type Rates** — Bar chart of conversion by campaign type
   - **Correlation Heatmap** — Feature correlation matrix
5. Expand "View Raw Data Sample" to inspect the first 100 rows

### 🏋️ Page 2 — Train Model

1. Click **"🏋️ Train Model"** in the sidebar
2. Read the pipeline description (3-step process)
3. Click **"🚀 Start Training Pipeline"**
4. Wait ~30-60 seconds while the pipeline runs:
   - **Step 1:** Data Ingestion — reads CSV, creates 80/20 train/test split
   - **Step 2:** Data Transformation — scales 13 numerical features with StandardScaler, encodes 3 categorical features with OneHotEncoder
   - **Step 3:** Model Training — trains 4 classifiers and evaluates each
5. On success, you'll see:
   - ✅ Green success banner with the best model name
   - Model comparison cards showing Accuracy, Precision, Recall, F1, AUC
   - Bar chart comparing all model accuracies
6. Artifacts are saved to `artifacts/` (model.pkl, preprocessor.pkl, model_report.json)

### 🔮 Page 3 — Predict Conversion

1. Click **"🔮 Predict"** in the sidebar
2. Fill in the input form across 4 sections:
   - **Customer Info:** Age, Gender, Income, Previous Purchases, Loyalty Points
   - **Campaign Details:** Channel, Type, Ad Spend, Click-Through Rate, Conversion Rate
   - **Website Metrics:** Visits, Pages/Visit, Time on Site
   - **Engagement Metrics:** Social Shares, Email Opens, Email Clicks
3. Click **"🔮 Predict Conversion"**
4. View the result:
   - ✅ **WILL CONVERT** (green) or ❌ **WILL NOT CONVERT** (red)
   - Probability gauge showing the conversion confidence percentage
5. Expand "View Input Data" to see the submitted feature values

---

## 🤖 Models Trained (with SMOTE)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Gradient Boosting** 🏆 | **0.9137** | **0.9284** | **0.9772** | **0.9522** | **0.8241** |
| XGBoost | 0.9125 | 0.9260 | 0.9787 | 0.9516 | 0.8322 |
| Random Forest | 0.8975 | 0.9162 | 0.9723 | 0.9434 | 0.8127 |
| Logistic Regression | 0.7256 | 0.9506 | 0.7255 | 0.8229 | 0.7895 |

The **Gradient Boosting Classifier** remains the best model with **91.37% accuracy** and a high F1-score, indicating robust performance across both classes.

---

## ⚖️ Handling Class Imbalance (SMOTE)

The original dataset was imbalanced, with significantly more samples of successful conversions than non-conversions. To ensure the model doesn't just bias towards the majority class, we implemented **SMOTE (Synthetic Minority Over-sampling Technique)**.

**Key Changes:**
- **Technique**: Used `imbalanced-learn` library to apply SMOTE.
- **Workflow**: SMOTE is applied only to the **training data** after numerical scaling and categorical encoding. This ensures the test set remains a true representation of real-world data.
### 📉 Before vs After SMOTE Comparison

| Model | Accuracy (Before) | Accuracy (After) | F1 Score (Before) | F1 Score (After) | Impact |
|-------|-------------------|------------------|-------------------|------------------|--------|
| **Gradient Boosting** | 0.9219 | 0.9137 | 0.9569 | 0.9522 | More balanced across classes |
| **XGBoost** | 0.9119 | 0.9125 | 0.9517 | 0.9516 | Slight accuracy gain |
| **Random Forest** | 0.8869 | 0.8975 | 0.9393 | 0.9434 | ⬆️ Improved overall |
| **Logistic Regression** | 0.8881 | 0.7256 | 0.9393 | 0.8229 | **Significant shift**: Now detects the minority class much better. |

**Key Observations:**
- **Minority Class Detection**: Before SMOTE, models (especially Logistic Regression) tended to ignore the minority class (non-converters) to achieve high overall accuracy. 
- **Logistic Regression Transformation**: After SMOTE, its ability to identify the minority class improved drastically (True Negatives went from ~35 to ~141), even though overall accuracy dropped.
- **Robustness**: Gradient Boosting and XGBoost remained very stable, maintaining high performance while becoming fairer.

---

## 📦 Dataset Overview

- **Source:** `digital_marketing_campaign_dataset.csv`
- **Records:** 8,001
- **Features:** 20 columns

| Feature | Type | Description |
|---------|------|-------------|
| CustomerID | ID | Unique customer identifier (dropped) |
| Age | Numerical | Customer age (18–69) |
| Gender | Categorical | Male / Female |
| Income | Numerical | Annual income ($20K–$150K) |
| CampaignChannel | Categorical | Email, PPC, SEO, Social Media, Referral |
| CampaignType | Categorical | Awareness, Consideration, Conversion, Retention |
| AdSpend | Numerical | Advertising spend ($0–$10K) |
| ClickThroughRate | Numerical | CTR (0–0.3) |
| ConversionRate | Numerical | Historical conversion rate (0–0.2) |
| WebsiteVisits | Numerical | Number of website visits |
| PagesPerVisit | Numerical | Average pages viewed per visit |
| TimeOnSite | Numerical | Time spent on site (minutes) |
| SocialShares | Numerical | Number of social media shares |
| EmailOpens | Numerical | Number of email opens |
| EmailClicks | Numerical | Number of email clicks |
| PreviousPurchases | Numerical | Number of past purchases |
| LoyaltyPoints | Numerical | Customer loyalty points |
| AdvertisingPlatform | Categorical | Always "IsConfid" (dropped — no variance) |
| AdvertisingTool | Categorical | Always "ToolConfid" (dropped — no variance) |
| **Conversion** | **Target** | **0 = No, 1 = Yes** |

---

## 🐳 Docker Deployment

```bash
# Build the Docker image
docker build -t marketing-conversion-ml .

# Run the container
docker run -p 8501:8501 marketing-conversion-ml
```

Access the app at [http://localhost:8501](http://localhost:8501)

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 📁 Key Configuration Files

- **`src/config/config.yaml`** — Data paths, feature columns, artifact locations
- **`src/config/params.yaml`** — Model hyperparameters (n_estimators, learning_rate, etc.)

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** — Interactive web dashboard
- **scikit-learn** — ML preprocessing & models
- **XGBoost** — Gradient boosting classifier
- **Plotly** — Interactive data visualizations
- **Pandas / NumPy** — Data manipulation
- **PyYAML** — Configuration management
