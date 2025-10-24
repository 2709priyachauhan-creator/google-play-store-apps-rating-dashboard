import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import accuracy_score
import joblib

# Load trained models and transformers (must exist in same folder)
reg_model = joblib.load('reg_model.joblib')
clf = joblib.load('clf.joblib')
scaler = joblib.load('scaler.joblib')
imputer = joblib.load('imputer.joblib')
encoder = joblib.load('encoder.joblib')
cat_only_encoder = joblib.load('cat_only_encoder.joblib')
tfidf = joblib.load('tfidf.joblib')

X_test_reg = joblib.load('X_test_reg.joblib')
y_test_reg = joblib.load('y_test_reg.joblib')
features_reg = joblib.load('features_reg.joblib')

st.title("Google Play Store App Analysis Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload apps/reviews CSV", type="csv")

def parse_installs(val):
    """Safely parse Installs: Handles '3.0M', '50,000+', 'Free', blank, etc."""
    try:
        v = str(val).replace('+', '').replace(',', '').strip()
        if v.lower() in ['free', '', 'nan']:
            return np.nan
        if 'M' in v:
            return float(v.replace('M','')) * 1_000_000
        if 'k' in v or 'K' in v:
            return float(v.replace('k','').replace('K','')) * 1_000
        return float(v)
    except:
        return np.nan

if uploaded_file:
    user_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", user_data.head())

    reg_features = ['Reviews', 'Installs', 'Price', 'Sentiment_Polarity', 'Days Since Last Update']
    categorical_cols = ['Category', 'Type', 'Content Rating']

    # Clean/conversion for numeric columns
    if 'Installs' in user_data.columns:
        user_data['Installs'] = user_data['Installs'].apply(parse_installs)
    if 'Reviews' in user_data.columns:
        user_data['Reviews'] = pd.to_numeric(user_data['Reviews'], errors='coerce')
    if 'Price' in user_data.columns:
        user_data['Price'] = user_data['Price'].astype(str).str.replace('$','',regex=False)
        user_data['Price'] = pd.to_numeric(user_data['Price'], errors='coerce')
    if 'Sentiment_Polarity' in user_data.columns:
        user_data['Sentiment_Polarity'] = pd.to_numeric(user_data['Sentiment_Polarity'], errors='coerce')
    if 'Days Since Last Update' in user_data.columns:
        user_data['Days Since Last Update'] = pd.to_numeric(user_data['Days Since Last Update'], errors='coerce')

    # Robust categorical column handling: ensure all exist, fill NaN, force string!
    for col in categorical_cols:
        if col not in user_data.columns:
            user_data[col] = 'Unknown'
        user_data[col] = user_data[col].fillna('Unknown').astype(str)

    # Fill missing numerical columns (regression) with NaN
    for col in reg_features:
        if col not in user_data.columns:
            user_data[col] = np.nan

    # Regression prediction only if all regression columns present
    can_regress = all(col in user_data.columns for col in (reg_features + categorical_cols))
    if can_regress:
        try:
            encoded_cats = encoder.transform(user_data[categorical_cols])
            encoded_cat_df = pd.DataFrame(
                encoded_cats, 
                columns=encoder.get_feature_names_out(categorical_cols),
                index=user_data.index
            )
            user_features = pd.concat([user_data[reg_features], encoded_cat_df], axis=1)
            user_features_imputed = imputer.transform(user_features)
            X_input_reg = scaler.transform(user_features_imputed)
            rating_pred = reg_model.predict(X_input_reg)
            st.subheader("Predicted Ratings (Regression)")
            st.write(rating_pred)
        except Exception as exc:
            st.warning(f"Regression prediction failed: {exc}")
    else:
        st.info("Regression prediction not available for this file (missing required columns).")

    # Sentiment prediction—must have 'Translated_Review' & 'Category'
    if 'Translated_Review' in user_data.columns:
        user_data['Category'] = user_data.get('Category', 'Unknown')
        user_data['Category'] = user_data['Category'].fillna('Unknown').astype(str)
        review_texts = user_data['Translated_Review'].fillna('')
        X_text = tfidf.transform(review_texts)
        user_cats = cat_only_encoder.transform(user_data['Category'].values.reshape(-1, 1))
        X_input_cls = np.concatenate([X_text.toarray(), user_cats], axis=1)
        sentiment_pred = clf.predict(X_input_cls)
        st.subheader("Predicted Sentiment Classes")
        st.write(sentiment_pred)
    else:
        st.info("Sentiment prediction not available for this file (missing 'Translated_Review').")

st.header("Dashboard Tabs")
tab1, tab2 = st.tabs(["EDA", "Model Metrics"])

with tab1:
    st.write("Upload data and explore summary/statistics here.")
    if uploaded_file:
        st.dataframe(user_data.describe())

with tab2:
    st.write("Model Performance Metrics")
    st.write(f"Regression R2 Score (sample set): {reg_model.score(X_test_reg, y_test_reg):.3f}")


