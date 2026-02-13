# ML Classification Model Demo App
# Built for BITS Pilani ML Assignment 2
# Wine Quality Classification

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="ML Classification App",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTab {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions

@st.cache_resource
def load_model(model_name):
    """Load the trained model - cached for performance"""
    model_file = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    try:
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file {model_file} not found! Make sure you've trained the models.")
        return None

@st.cache_resource
def load_scaler():
    """Load the saved scaler"""
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("Scaler not found. Using data without scaling.")
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all required metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC'] = 0.0
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create an interactive confusion matrix using Plotly"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create text annotations
    text = [[f'{count}<br>({percent:.1f}%)' 
             for count, percent in zip(row_counts, row_percents)]
            for row_counts, row_percents in zip(cm, cm_percent)]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Bad Wine (0)', 'Predicted: Good Wine (1)'],
        y=['Actual: Bad Wine (0)', 'Actual: Good Wine (1)'],
        text=text,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=500,
        font=dict(size=12)
    )
    
    return fig

def plot_metrics_radar(metrics, model_name):
    """Create a radar chart for metrics visualization"""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Close the radar chart
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name=model_name,
        line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 1],
                tickformat='.2f'
            )
        ),
        showlegend=True,
        title=f"Performance Metrics - {model_name}",
        height=500,
        font=dict(size=12)
    )
    
    return fig

def plot_metrics_bar(metrics, model_name):
    """Create a bar chart for metrics"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        text=[f'{v:.4f}' for v in metrics.values()],
        textposition='auto',
        marker=dict(
            color=list(metrics.values()),
            colorscale='Blues',
            showscale=False
        )
    ))
    
    fig.update_layout(
        title=f'Performance Metrics - {model_name}',
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.1]),
        height=400,
        showlegend=False
    )
    
    return fig

# Main App Function

def main():
    # Header
    st.markdown('<h1 class="main-header">ML Classification Model Demo</h1>', 
                unsafe_allow_html=True)
    st.markdown("##### Wine Quality Prediction using Multiple Machine Learning Models")
    st.markdown("---")
    
    # Sidebar for model selection and file upload
    st.sidebar.title("Configuration Panel")
    st.sidebar.markdown("### Model Selection")
    
    # Model selection dropdown
    model_options = [
        'Logistic Regression',
        'Decision Tree',
        'kNN',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ]
    
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        model_options,
        help="Select which ML model to use for predictions"
    )
    
    # Display model info
    model_descriptions = {
        'Logistic Regression': 'Linear model using sigmoid function for binary classification',
        'Decision Tree': 'Tree-based classifier that learns decision rules from features',
        'kNN': 'Instance-based learning that classifies based on nearest neighbors',
        'Naive Bayes': 'Probabilistic classifier based on Bayes theorem',
        'Random Forest': 'Ensemble of decision trees for robust predictions',
        'XGBoost': 'Gradient boosting ensemble method for high performance'
    }
    
    st.sidebar.info(f"**About this model:** {model_descriptions[selected_model]}")
    
    # File upload section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your test dataset (CSV format). Should contain wine features and optionally an 'actual_label' column."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Note:** Upload test data CSV to see predictions and evaluation metrics!")
    
    # Main content area
    if uploaded_file is not None:
        # Load the data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Dataset shape: {df.shape}")
            
            # Show data preview
            with st.expander("View Uploaded Data", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", df.shape[0])
                with col2:
                    st.metric("Total Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
            
            # Check if actual labels exist
            has_labels = 'actual_label' in df.columns or 'label' in df.columns or 'quality_binary' in df.columns
            
            if has_labels:
                if 'actual_label' in df.columns:
                    label_col = 'actual_label'
                elif 'label' in df.columns:
                    label_col = 'label'
                else:
                    label_col = 'quality_binary'
                    
                y_true = df[label_col]
                X_test = df.drop(label_col, axis=1)
            else:
                st.warning("No label column found. Will show predictions only (no evaluation metrics).")
                X_test = df
                y_true = None
            
            # Load model
            model = load_model(selected_model)
            scaler = load_scaler()
            
            if model is not None:
                # Make predictions
                try:
                    # Apply scaling for models that need it
                    if selected_model in ['Logistic Regression', 'kNN', 'Naive Bayes'] and scaler is not None:
                        X_test_processed = scaler.transform(X_test)
                    else:
                        X_test_processed = X_test
                    
                    y_pred = model.predict(X_test_processed)
                    
                    # Try to get prediction probabilities
                    try:
                        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
                    except:
                        y_pred_proba = None
                    
                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Performance Metrics", "Predictions", "Confusion Matrix", "Classification Report"
                    ])
                    
                    with tab1:
                        if y_true is not None:
                            st.subheader(f"Performance Metrics - {selected_model}")
                            
                            # Calculate metrics
                            metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
                            
                            # Display metrics in a nice grid
                            st.markdown("#### Overall Scores")
                            cols = st.columns(3)
                            metric_items = list(metrics.items())
                            
                            for idx, (metric_name, value) in enumerate(metric_items):
                                with cols[idx % 3]:
                                    # Color code based on performance
                                    if value >= 0.8:
                                        delta_color = "normal"
                                    elif value >= 0.6:
                                        delta_color = "off"
                                    else:
                                        delta_color = "inverse"
                                    
                                    st.metric(
                                        label=metric_name,
                                        value=f"{value:.4f}",
                                        delta=None
                                    )
                            
                            st.markdown("---")
                            
                            # Visualizations
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.plotly_chart(
                                    plot_metrics_radar(metrics, selected_model),
                                    use_container_width=True
                                )
                            
                            with col2:
                                st.plotly_chart(
                                    plot_metrics_bar(metrics, selected_model),
                                    use_container_width=True
                                )
                        else:
                            st.info("Upload data with an 'actual_label' column to see performance metrics.")
                    
                    with tab2:
                        st.subheader("Model Predictions")
                        
                        # Create predictions dataframe
                        pred_df = X_test.copy()
                        pred_df['Predicted_Label'] = y_pred
                        pred_df['Predicted_Class'] = pred_df['Predicted_Label'].map({0: 'Bad Wine', 1: 'Good Wine'})
                        
                        if y_pred_proba is not None:
                            pred_df['Confidence_Score'] = y_pred_proba
                            pred_df['Confidence_Percentage'] = (y_pred_proba * 100).round(2)
                        
                        if y_true is not None:
                            pred_df['Actual_Label'] = y_true.values
                            pred_df['Actual_Class'] = pred_df['Actual_Label'].map({0: 'Bad Wine', 1: 'Good Wine'})
                            pred_df['Correct_Prediction'] = (pred_df['Predicted_Label'] == pred_df['Actual_Label'])
                        
                        st.dataframe(pred_df, use_container_width=True, height=400)
                        
                        # Download button
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name=f'{selected_model.replace(" ", "_")}_predictions.csv',
                            mime='text/csv'
                        )
                        
                        # Prediction statistics
                        if y_true is not None:
                            st.markdown("---")
                            st.markdown("#### Prediction Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                accuracy = (pred_df['Correct_Prediction'].sum() / len(pred_df)) * 100
                                st.metric("Accuracy", f"{accuracy:.2f}%")
                            with col2:
                                st.metric("Correct", pred_df['Correct_Prediction'].sum())
                            with col3:
                                st.metric("Incorrect", (~pred_df['Correct_Prediction']).sum())
                            with col4:
                                st.metric("Total Predictions", len(pred_df))
                        
                        # Prediction distribution
                        st.markdown("---")
                        st.markdown("#### Prediction Distribution")
                        pred_counts = pd.DataFrame({
                            'Class': ['Bad Wine (0)', 'Good Wine (1)'],
                            'Count': [
                                (y_pred == 0).sum(),
                                (y_pred == 1).sum()
                            ]
                        })
                        
                        fig = px.bar(pred_counts, x='Class', y='Count', 
                                    title='Distribution of Predicted Classes',
                                    color='Count', color_continuous_scale='Blues')
                        fig.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        if y_true is not None:
                            st.subheader("Confusion Matrix Analysis")
                            
                            # Plot confusion matrix
                            st.plotly_chart(
                                plot_confusion_matrix(y_true, y_pred, selected_model),
                                use_container_width=True
                            )
                            
                            # Show confusion matrix values in table
                            st.markdown("#### Confusion Matrix Values")
                            cm = confusion_matrix(y_true, y_pred)
                            cm_df = pd.DataFrame(
                                cm,
                                columns=['Predicted: Bad Wine (0)', 'Predicted: Good Wine (1)'],
                                index=['Actual: Bad Wine (0)', 'Actual: Good Wine (1)']
                            )
                            st.table(cm_df)
                            
                            # Calculate additional metrics from confusion matrix
                            tn, fp, fn, tp = cm.ravel()
                            
                            st.markdown("#### Detailed Breakdown")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**True Positives (TP):** " + str(tp))
                                st.markdown("Correctly predicted as Good Wine")
                                st.markdown("")
                                st.markdown("**False Positives (FP):** " + str(fp))
                                st.markdown("Incorrectly predicted as Good Wine")
                            
                            with col2:
                                st.markdown("**True Negatives (TN):** " + str(tn))
                                st.markdown("Correctly predicted as Bad Wine")
                                st.markdown("")
                                st.markdown("**False Negatives (FN):** " + str(fn))
                                st.markdown("Incorrectly predicted as Bad Wine")
                        else:
                            st.info("Upload data with an 'actual_label' column to see confusion matrix.")
                    
                    with tab4:
                        if y_true is not None:
                            st.subheader("Classification Report")
                            
                            # Generate classification report
                            report = classification_report(
                                y_true, y_pred, 
                                target_names=['Bad Wine (0)', 'Good Wine (1)'],
                                output_dict=True
                            )
                            report_df = pd.DataFrame(report).transpose()
                            
                            # Style the dataframe
                            st.dataframe(
                                report_df.style.format("{:.4f}").background_gradient(cmap='Blues'),
                                use_container_width=True
                            )
                            
                            st.markdown("---")
                            st.markdown("#### Metric Definitions")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                **Precision:** Of all predicted positive cases, how many were actually positive?
                                - Formula: TP / (TP + FP)
                                - High precision means low false positive rate
                                """)
                                
                                st.markdown("""
                                **Recall (Sensitivity):** Of all actual positive cases, how many did we correctly predict?
                                - Formula: TP / (TP + FN)
                                - High recall means low false negative rate
                                """)
                            
                            with col2:
                                st.markdown("""
                                **F1-Score:** Harmonic mean of precision and recall
                                - Formula: 2 * (Precision * Recall) / (Precision + Recall)
                                - Balances precision and recall
                                """)
                                
                                st.markdown("""
                                **Support:** Number of actual occurrences of each class
                                - Helps understand class distribution
                                - Important for imbalanced datasets
                                """)
                        else:
                            st.info("Upload data with an 'actual_label' column to see classification report.")
                
                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")
                    st.error("Please ensure your data has the correct features matching the training data.")
                    st.error("Expected features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        # Welcome screen when no file is uploaded
        st.info("Please upload a CSV file from the sidebar to get started!")
        
        st.markdown("""
        ### How to Use This App
        
        1. **Select a Model:** Choose from 6 different ML models in the sidebar
        2. **Upload Data:** Upload your test data CSV file (with or without labels)
        3. **View Results:** Explore predictions, metrics, and visualizations across different tabs
        
        ### Data Format Requirements
        
        Your CSV file should contain the following wine quality features:
        - fixed acidity
        - volatile acidity
        - citric acid
        - residual sugar
        - chlorides
        - free sulfur dioxide
        - total sulfur dioxide
        - density
        - pH
        - sulphates
        - alcohol
        - **Optional:** `actual_label` column (0 for bad wine, 1 for good wine) for evaluation
        
        ### Available Models
        
        This app includes 6 trained classification models:
        
        1. **Logistic Regression** - Linear model for binary classification
        2. **Decision Tree** - Tree-based rule learning algorithm
        3. **kNN** - K-Nearest Neighbors (k=5)
        4. **Naive Bayes** - Gaussian probabilistic classifier
        5. **Random Forest** - Ensemble of 100 decision trees
        6. **XGBoost** - Gradient boosting ensemble method
        
        ### Example Data
        
        Here's an example of what your data should look like:
        """)
        
        # Show example data format
        example_df = pd.DataFrame({
            'fixed acidity': [7.4, 7.8, 7.8],
            'volatile acidity': [0.70, 0.88, 0.76],
            'citric acid': [0.00, 0.00, 0.04],
            'residual sugar': [1.9, 2.6, 2.3],
            'chlorides': [0.076, 0.098, 0.092],
            'free sulfur dioxide': [11, 25, 15],
            'total sulfur dioxide': [34, 67, 54],
            'density': [0.9978, 0.9968, 0.9970],
            'pH': [3.51, 3.20, 3.26],
            'sulphates': [0.56, 0.68, 0.65],
            'alcohol': [9.4, 9.8, 9.8],
            'actual_label': [0, 0, 1]
        })
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("**Note:** The 'actual_label' column is optional but required for viewing performance metrics.")

if __name__ == "__main__":
    main()
