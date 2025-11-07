# app.py - Medical Insurance Cost Prediction System
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Try to import optional dependencies
try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e86ab;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .cost-low {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    }
    .cost-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    }
    .cost-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2e86ab;
        margin: 0.5rem 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
    .insurance-plans {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class InsurancePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'age', 'sex', 'bmi', 'children', 'smoker', 'region'
        ]

        self.feature_info = {
            'age': {
                'desc': 'Age of the primary beneficiary',
                'type': 'number',
                'min': 18, 'max': 80, 'step': 1,
                'normal_range': (18, 65)
            },
            'sex': {
                'desc': 'Gender of the beneficiary',
                'type': 'select',
                'options': {'0': 'Female', '1': 'Male'}
            },
            'bmi': {
                'desc': 'Body Mass Index',
                'type': 'number',
                'min': 15, 'max': 50, 'step': 0.1,
                'normal_range': (18.5, 24.9)
            },
            'children': {
                'desc': 'Number of children covered by health insurance',
                'type': 'number',
                'min': 0, 'max': 10, 'step': 1,
                'normal_range': (0, 3)
            },
            'smoker': {
                'desc': 'Smoking status',
                'type': 'select',
                'options': {'0': 'No', '1': 'Yes'}
            },
            'region': {
                'desc': 'Residential area in the US',
                'type': 'select',
                'options': {
                    '0': 'Northeast',
                    '1': 'Northwest',
                    '2': 'Southeast',
                    '3': 'Southwest'
                }
            }
        }

        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            if JOBLIB_AVAILABLE and os.path.exists('models/insurance_model.pkl'):
                self.model = joblib.load('models/insurance_model.pkl')
                if os.path.exists('models/insurance_feature_names.pkl'):
                    self.feature_names = joblib.load('models/insurance_feature_names.pkl')
                st.sidebar.success("✅ AI Model Loaded Successfully!")
            else:
                st.sidebar.info("🤖 Using Rule-Based System (AI model not found)")
        except Exception as e:
            st.sidebar.info("🤖 Using Rule-Based System")

    def get_cost_level(self, cost):
        """Determine cost level based on predicted insurance cost"""
        if cost < 5000:
            return 'Low', '🟢', 'cost-low', "Affordable insurance range. Good health profile."
        elif cost < 10000:
            return 'Medium', '🟡', 'cost-medium', "Moderate insurance cost. Consider lifestyle improvements."
        elif cost < 15000:
            return 'High', '🟠', 'cost-high', "High insurance cost. Recommended to review health factors."
        else:
            return 'Very High', '🔴', 'cost-high', "Very high insurance cost. Immediate health review recommended."

    def rule_based_prediction(self, input_data):
        """Rule-based prediction when ML model is not available"""
        base_cost = 2000

        # Age factor
        age_factor = input_data['age'] * 100

        # BMI factor
        bmi = input_data['bmi']
        if bmi < 18.5:
            bmi_factor = 500  # Underweight
        elif bmi <= 24.9:
            bmi_factor = 0  # Normal
        elif bmi <= 29.9:
            bmi_factor = 1000  # Overweight
        else:
            bmi_factor = 2000  # Obese

        # Smoking factor
        smoker_factor = 8000 if input_data['smoker'] == 1 else 0

        # Children factor
        children_factor = input_data['children'] * 500

        # Region factor (variation)
        region_factors = [0, 300, 500, 200]  # NE, NW, SE, SW
        region_factor = region_factors[input_data['region']]

        # Gender factor (small difference)
        gender_factor = 300 if input_data['sex'] == 1 else 0

        total_cost = base_cost + age_factor + bmi_factor + smoker_factor + children_factor + region_factor + gender_factor

        return total_cost

    def predict(self, input_data):
        """Make prediction for insurance cost"""
        try:
            if self.model and JOBLIB_AVAILABLE:
                # ML Model prediction
                input_df = pd.DataFrame([input_data])
                input_df = input_df[self.feature_names]
                predicted_cost = self.model.predict(input_df)[0]
                method = "AI Machine Learning Model"
            else:
                # Rule-based prediction
                predicted_cost = self.rule_based_prediction(input_data)
                method = "Insurance Rule-Based System"

            cost_level, cost_emoji, cost_class, recommendation = self.get_cost_level(predicted_cost)

            return {
                'predicted_cost': predicted_cost,
                'cost_level': cost_level,
                'cost_emoji': cost_emoji,
                'cost_class': cost_class,
                'recommendation': recommendation,
                'method': method,
                'success': True
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def create_sidebar():
    """Create the sidebar with information and controls"""
    st.sidebar.title("💰 Insurance Cost Predictor")

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This AI-powered system predicts medical insurance costs based on demographic and health factors.

    **Note:** Predictions are estimates for educational purposes. Actual insurance costs may vary.
    """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")

    # Dependency status
    if JOBLIB_AVAILABLE:
        st.sidebar.success("✅ ML Engine: Available")
    else:
        st.sidebar.warning("⚠️ ML Engine: Limited")

    if PLOTLY_AVAILABLE:
        st.sidebar.success("✅ Visualizations: Available")
    else:
        st.sidebar.warning("⚠️ Visualizations: Basic")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Actions")

    if st.sidebar.button("🔄 Reset Form"):
        st.rerun()

    if st.sidebar.button("📊 View Cost Analysis"):
        show_cost_analysis()

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Insurance Analytics Platform<br>
        Version 1.0 • Predictive AI
    </div>
    """, unsafe_allow_html=True)


def show_cost_analysis():
    """Show insurance cost analysis"""
    st.sidebar.subheader("💡 Cost Factors")
    st.sidebar.write("""
    **Key Cost Drivers:**
    - 🚬 Smoking: +$8,000
    - 📊 High BMI: +$1,000-2,000
    - 👴 Age: +$100/year
    - 👶 Children: +$500/child
    - 🗺️ Region: Varies by location
    """)


def create_feature_input(feature, info):
    """Create input for a single feature"""
    with st.container():
        if info['type'] == 'number':
            value = st.number_input(
                label=f"**{feature.upper()}**",
                min_value=info['min'],
                max_value=info['max'],
                value=info.get('default', (info['min'] + info['max']) // 2),
                step=info['step'],
                help=info['desc']
            )
        elif info['type'] == 'select':
            option_key = st.selectbox(
                label=f"**{feature.upper()}**",
                options=list(info['options'].keys()),
                format_func=lambda x: f"{info['options'][x]}",
                help=info['desc']
            )
            value = int(option_key)

        # Show normal range if available
        if 'normal_range' in info:
            min_val, max_val = info['normal_range']
            if min_val <= value <= max_val:
                st.markdown(
                    f"<span style='color: green; font-size: 0.8rem;'>✓ Within normal range ({min_val}-{max_val})</span>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<span style='color: orange; font-size: 0.8rem;'>⚠️ Outside normal range ({min_val}-{max_val})</span>",
                    unsafe_allow_html=True)

        return value


def create_input_form(predictor):
    """Create the main input form"""
    st.markdown('<div class="main-header">Medical Insurance Cost Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align: center; color: #666; margin-bottom: 2rem;">AI-Powered Insurance Premium Estimation</div>',
        unsafe_allow_html=True)

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Detailed Input", "⚡ Quick Estimate"])

    input_data = {}

    with tab1:
        st.subheader("Personal & Health Information")

        # Create columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            input_data['age'] = create_feature_input('age', predictor.feature_info['age'])
            input_data['bmi'] = create_feature_input('bmi', predictor.feature_info['bmi'])
            input_data['children'] = create_feature_input('children', predictor.feature_info['children'])

        with col2:
            input_data['sex'] = create_feature_input('sex', predictor.feature_info['sex'])
            input_data['smoker'] = create_feature_input('smoker', predictor.feature_info['smoker'])
            input_data['region'] = create_feature_input('region', predictor.feature_info['region'])

    with tab2:
        st.subheader("Quick Insurance Estimate")
        st.info("Provide basic information for quick cost estimation")

        col1, col2 = st.columns(2)

        with col1:
            input_data['age'] = st.slider("**AGE**", 18, 80, 35)
            input_data['bmi'] = st.slider("**BMI**", 15.0, 50.0, 25.0, 0.1)
            input_data['smoker'] = st.selectbox("**SMOKER**", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with col2:
            input_data['sex'] = st.selectbox("**GENDER**", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            input_data['children'] = st.slider("**CHILDREN**", 0, 10, 1)
            input_data['region'] = st.selectbox("**REGION**", [0, 1, 2, 3],
                                                format_func=lambda x:
                                                ["Northeast", "Northwest", "Southeast", "Southwest"][x])

    return input_data


def display_metric_analysis(feature, value, info):
    """Display analysis for a single feature"""
    if 'normal_range' not in info:
        return

    min_val, max_val = info['normal_range']
    status = "Optimal" if min_val <= value <= max_val else "Review Recommended"
    color = "green" if status == "Optimal" else "orange"

    st.markdown(f"""
    <div class="feature-card">
        <h4>{feature.upper()}</h4>
        <h3 style="color: {color};">{value}</h3>
        <p>Status: <strong style="color: {color};">{status}</strong></p>
        <p style="font-size: 0.8rem;">Optimal Range: {min_val}-{max_val}</p>
    </div>
    """, unsafe_allow_html=True)


def display_insurance_plans(predicted_cost):
    """Display recommended insurance plans based on predicted cost"""
    st.subheader("💼 Recommended Insurance Plans")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="insurance-plans">
            <h4>🛡️ Basic Plan</h4>
            <h3>${:,.0f}</h3>
            <p>• Basic coverage<br>• Emergency care<br>• Annual checkup</p>
        </div>
        """.format(predicted_cost * 0.7), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insurance-plans">
            <h4>⭐ Standard Plan</h4>
            <h3>${:,.0f}</h3>
            <p>• Comprehensive<br>• Specialist visits<br>• Prescription drugs</p>
        </div>
        """.format(predicted_cost), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="insurance-plans">
            <h4>🚀 Premium Plan</h4>
            <h3>${:,.0f}</h3>
            <p>• Full coverage<br>• Dental & Vision<br>• Low deductibles</p>
        </div>
        """.format(predicted_cost * 1.3), unsafe_allow_html=True)


def display_results(results, input_data, predictor):
    """Display comprehensive prediction results"""
    st.markdown("---")

    if not results['success']:
        st.error(f"❌ Prediction failed: {results['error']}")
        return

    # Main prediction card
    cost_class = results['cost_class']
    cost_emoji = results['cost_emoji']

    st.markdown(f"""
    <div class="prediction-card {cost_class}">
        <h2 style="color: white; margin: 0;">{cost_emoji} {results['cost_level']} COST</h2>
        <h1 style="color: white; margin: 0.5rem 0;">${results['predicted_cost']:,.2f}</h1>
        <h4 style="color: white; margin: 0;">Annual Insurance Premium</h4>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Method: {results['method']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Results in columns
    col1, col2 = st.columns(2)

    with col1:
        if PLOTLY_AVAILABLE:
            # Cost breakdown chart
            factors = {
                'Base Cost': 2000,
                'Age Factor': input_data['age'] * 100,
                'BMI Factor': 0 if 18.5 <= input_data['bmi'] <= 24.9 else 1000 if input_data['bmi'] <= 29.9 else 2000,
                'Smoking Factor': 8000 if input_data['smoker'] == 1 else 0,
                'Children Factor': input_data['children'] * 500,
                'Region Factor': [0, 300, 500, 200][input_data['region']]
            }

            fig = px.pie(
                values=list(factors.values()),
                names=list(factors.keys()),
                title="Insurance Cost Breakdown",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple cost display
            st.metric("Monthly Premium", f"${results['predicted_cost'] / 12:,.2f}")
            st.metric("Cost Level", f"{results['cost_level']} {results['cost_emoji']}")

    with col2:
        st.subheader("🎯 Cost Optimization Tips")
        st.info(results['recommendation'])

        # Specific recommendations based on input
        tips = []
        if input_data['smoker'] == 1:
            tips.append("• **Quit smoking** - Could save up to $8,000 annually")
        if input_data['bmi'] > 24.9:
            tips.append("• **Improve BMI** - Target 18.5-24.9 for better rates")
        if input_data['age'] > 50:
            tips.append("• **Consider senior plans** - Specialized coverage options")
        if input_data['children'] > 3:
            tips.append("• **Family plans** - May offer better value")

        if tips:
            st.warning("**Potential Savings:**")
            for tip in tips:
                st.write(tip)
        else:
            st.success("**Good Profile** - You have optimal factors for insurance costs")

    # Display recommended insurance plans
    display_insurance_plans(results['predicted_cost'])

    # Detailed analysis
    st.markdown("---")
    st.subheader("📊 Health & Demographic Analysis")

    # Key parameters analysis
    col1, col2, col3 = st.columns(3)

    key_metrics = ['age', 'bmi', 'children']

    for i, metric in enumerate(key_metrics):
        if metric in input_data:
            with [col1, col2, col3][i]:
                display_metric_analysis(metric, input_data[metric], predictor.feature_info[metric])


def save_prediction(input_data, results):
    """Save prediction to history"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_data = {
            'timestamp': timestamp,
            **input_data,
            'predicted_cost': results['predicted_cost'],
            'cost_level': results['cost_level'],
            'method': results['method']
        }

        df = pd.DataFrame([save_data])

        # Ensure directory exists
        os.makedirs('data', exist_ok=True)

        file_path = 'data/insurance_predictions.csv'
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

        st.success("✅ Prediction saved to records!")
    except Exception as e:
        st.error(f"❌ Error saving prediction: {e}")


def main():
    """Main application function"""

    # Initialize predictor
    predictor = InsurancePredictor()

    # Create sidebar
    create_sidebar()

    # Main content area
    if not JOBLIB_AVAILABLE:
        st.warning("""
        ⚠️ **ML Engine Notice:** 
        Advanced machine learning features are limited. The system is using insurance rule-based prediction.
        For full AI capabilities, install: `pip install joblib scikit-learn`
        """)

    # Create input form and get data
    input_data = create_input_form(predictor)

    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button(
            "💰 PREDICT INSURANCE COST",
            type="primary",
            use_container_width=True,
            help="Click to estimate annual insurance premium"
        )

    if predict_clicked:
        with st.spinner("🔄 Calculating insurance premium..."):
            # Add small delay for better UX
            import time
            time.sleep(1)

            results = predictor.predict(input_data)
            display_results(results, input_data, predictor)

            # Save prediction
            if st.button("💾 Save Quote", use_container_width=True):
                save_prediction(input_data, results)


# Run the application
if __name__ == "__main__":
    main()