import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    .title-container {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .title-text {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .subtitle-text {
        color: #f0f0f0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title with gradient background
st.markdown("""
    <div class="title-container">
        <h1 class="title-text">📊 Customer Churn Prediction</h1>
        <p class="subtitle-text">Predict customer retention with AI-powered insights</p>
    </div>
""", unsafe_allow_html=True)

# Load models and encoders
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model('model.h5')
        
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        return model, label_encoder_gender, onehot_encoder_geo, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, label_encoder_gender, onehot_encoder_geo, scaler = load_models()

if model is None:
    st.stop()

# Create two columns for layout
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 👤 Customer Information")
    
    # Organize inputs in expandable sections
    with st.expander("📍 Demographics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            geography = st.selectbox(
                'Geography',
                onehot_encoder_geo.categories_[0],
                help="Customer's location"
            )
            age = st.slider(
                'Age',
                18, 92, 35,
                help="Customer's age in years"
            )
        with col2:
            gender = st.selectbox(
                'Gender',
                label_encoder_gender.classes_,
                help="Customer's gender"
            )
            tenure = st.slider(
                'Tenure (years)',
                0, 10, 5,
                help="Years with the bank"
            )
    
    with st.expander("💰 Financial Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input(
                'Credit Score',
                300, 850, 650,
                help="Credit score (300-850)"
            )
            balance = st.number_input(
                'Balance ($)',
                0.0, 250000.0, 50000.0, 1000.0,
                help="Current account balance"
            )
        with col2:
            estimated_salary = st.number_input(
                'Estimated Salary ($)',
                0.0, 200000.0, 50000.0, 1000.0,
                help="Annual estimated salary"
            )
            num_of_products = st.slider(
                'Number of Products',
                1, 4, 2,
                help="Number of bank products"
            )
    
    with st.expander("🔧 Account Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            has_cr_card = st.selectbox(
                'Has Credit Card',
                [0, 1],
                format_func=lambda x: 'Yes' if x == 1 else 'No',
                help="Does customer have a credit card?"
            )
        with col2:
            is_active_member = st.selectbox(
                'Is Active Member',
                [0, 1],
                format_func=lambda x: 'Yes' if x == 1 else 'No',
                help="Is customer actively using services?"
            )
    
    # Predict button
    predict_button = st.button('🔮 Predict Churn', type="primary", use_container_width=True)

with col_right:
    st.markdown("### 📈 Prediction Results")
    
    if predict_button:
        with st.spinner('Analyzing customer data...'):
            # Prepare the input data
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary]
            })
            
            # One-hot encode 'Geography'
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_encoded_df = pd.DataFrame(
                geo_encoded,
                columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
            )
            
            # Combine one-hot encoded columns with input data
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
            
            # Scale the input data
            input_data_scaled = scaler.transform(input_data)
            
            # Predict churn
            prediction = model.predict(input_data_scaled, verbose=0)
            prediction_proba = prediction[0][0]
            
            # Display results
            st.markdown("---")
            
            # Gauge chart for probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)", 'font': {'size': 20}},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Churn Risk",
                    f"{prediction_proba:.1%}",
                    delta=f"{(prediction_proba - 0.5):.1%}" if prediction_proba != 0.5 else "0%",
                    delta_color="inverse"
                )
            with col2:
                risk_level = "High" if prediction_proba > 0.7 else "Medium" if prediction_proba > 0.3 else "Low"
                st.metric("Risk Level", risk_level)
            
            # Detailed analysis
            st.markdown("---")
            if prediction_proba > 0.7:
                st.error("⚠️ **High Risk**: This customer is highly likely to churn. Immediate retention efforts recommended.")
                st.markdown("""
                **Recommended Actions:**
                - 🎯 Offer personalized retention package
                - 📞 Schedule priority call with account manager
                - 💎 Provide loyalty rewards or special benefits
                """)
            elif prediction_proba > 0.3:
                st.warning("⚡ **Medium Risk**: This customer shows some churn indicators. Consider preventive measures.")
                st.markdown("""
                **Recommended Actions:**
                - 📧 Send engagement campaigns
                - 🎁 Offer product upgrades or cross-sell opportunities
                - 📊 Monitor account activity closely
                """)
            else:
                st.success("✅ **Low Risk**: This customer is likely to stay. Maintain good service.")
                st.markdown("""
                **Recommended Actions:**
                - 😊 Continue excellent customer service
                - 📈 Look for upsell opportunities
                - 💬 Request feedback and testimonials
                """)
    else:
        st.info("👆 Fill in customer details and click 'Predict Churn' to see results")
        
        # Show example insights
        st.markdown("---")
        st.markdown("### 💡 Key Factors in Churn Prediction")
        st.markdown("""
        - **Age & Tenure**: Longer relationships reduce churn risk
        - **Balance & Salary**: Financial stability indicators
        - **Product Usage**: More products = higher engagement
        - **Activity Level**: Active members less likely to churn
        - **Geography**: Regional patterns affect retention
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🤖 Powered by TensorFlow & Streamlit | Built for Customer Success Teams</p>
    </div>
""", unsafe_allow_html=True)