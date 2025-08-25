import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🚚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
    }
    .prediction-result {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}

def generate_sample_data(n_samples=1000):
    """Generate realistic delivery data for training"""
    np.random.seed(42)
    
    # Define location zones with different characteristics
    zones = ['Downtown', 'Suburbs', 'Industrial', 'Residential', 'Commercial']
    traffic_conditions = ['Light', 'Moderate', 'Heavy']
    
    data = []
    for _ in range(n_samples):
        # Random location and destination
        origin = np.random.choice(zones)
        destination = np.random.choice(zones)
        
        # Distance (km) - varies by zone combination
        base_distance = np.random.uniform(2, 50)
        if origin == destination:
            distance = np.random.uniform(1, 15)  # Same zone
        elif origin in ['Downtown', 'Commercial'] and destination in ['Downtown', 'Commercial']:
            distance = base_distance * 0.7  # Urban areas closer
        else:
            distance = base_distance
            
        # Time of day (hour)
        hour = np.random.randint(6, 22)  # Delivery hours 6 AM to 10 PM
        
        # Traffic condition based on time and location
        if hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
            traffic = np.random.choice(['Moderate', 'Heavy'], p=[0.3, 0.7])
        elif hour in [10, 11, 12, 13, 14, 15, 16]:  # Business hours
            traffic = np.random.choice(['Light', 'Moderate'], p=[0.4, 0.6])
        else:  # Off-peak hours
            traffic = np.random.choice(['Light', 'Moderate'], p=[0.8, 0.2])
            
        # Calculate delivery time based on factors
        base_time = distance * 2.5  # Base: 2.5 minutes per km
        
        # Traffic multiplier
        traffic_multiplier = {'Light': 1.0, 'Moderate': 1.3, 'Heavy': 1.8}[traffic]
        
        # Zone complexity multiplier
        zone_multiplier = {
            'Downtown': 1.4,
            'Commercial': 1.2,
            'Industrial': 1.1,
            'Residential': 1.0,
            'Suburbs': 0.9
        }
        
        # Time of day multiplier
        if hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
            time_multiplier = 1.5
        elif hour in [12, 13]:  # Lunch hours
            time_multiplier = 1.2
        else:
            time_multiplier = 1.0
            
        # Calculate final delivery time with some randomness
        delivery_time = (base_time * traffic_multiplier * 
                        zone_multiplier[origin] * zone_multiplier[destination] * 
                        time_multiplier * np.random.uniform(0.8, 1.2))
        
        # Add day of week effect
        day_of_week = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                      'Friday', 'Saturday', 'Sunday'])
        if day_of_week in ['Saturday', 'Sunday']:
            delivery_time *= 0.85  # Faster on weekends
            
        data.append({
            'origin_zone': origin,
            'destination_zone': destination,
            'distance_km': round(distance, 2),
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'traffic_condition': traffic,
            'delivery_time_minutes': round(max(delivery_time, 5), 1)  # Minimum 5 minutes
        })
    
    return pd.DataFrame(data)

def prepare_features(df):
    """Prepare features for training"""
    df_processed = df.copy()
    
    # Create time-based features
    df_processed['is_rush_hour'] = df_processed['hour_of_day'].apply(
        lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0
    )
    df_processed['is_weekend'] = df_processed['day_of_week'].apply(
        lambda x: 1 if x in ['Saturday', 'Sunday'] else 0
    )
    df_processed['is_lunch_hour'] = df_processed['hour_of_day'].apply(
        lambda x: 1 if x in [12, 13] else 0
    )
    
    return df_processed

def train_model():
    """Train the delivery time prediction model"""
    # Generate training data
    with st.spinner("Generating training data..."):
        df = generate_sample_data(2000)
        df_processed = prepare_features(df)
    
    # Prepare features
    categorical_features = ['origin_zone', 'destination_zone', 'traffic_condition', 'day_of_week']
    numerical_features = ['distance_km', 'hour_of_day', 'is_rush_hour', 'is_weekend', 'is_lunch_hour']
    
    # Encode categorical variables
    encoders = {}
    df_encoded = df_processed.copy()
    
    for feature in categorical_features:
        encoder = LabelEncoder()
        df_encoded[feature] = encoder.fit_transform(df_processed[feature])
        encoders[feature] = encoder
    
    # Prepare training data
    X = df_encoded[categorical_features + numerical_features]
    y = df_encoded['delivery_time_minutes']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    with st.spinner("Training Random Forest model..."):
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store in session state
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.feature_columns = categorical_features + numerical_features
    st.session_state.model_trained = True
    st.session_state.training_data = df
    
    return model, encoders, (mae, rmse, r2), (X_test, y_test, y_pred)

def predict_delivery_time(origin, destination, distance, hour, day, traffic):
    """Make delivery time prediction"""
    if not st.session_state.model_trained:
        return None
    
    # Prepare input data
    input_data = pd.DataFrame({
        'origin_zone': [origin],
        'destination_zone': [destination],
        'distance_km': [distance],
        'hour_of_day': [hour],
        'day_of_week': [day],
        'traffic_condition': [traffic],
        'is_rush_hour': [1 if hour in [7, 8, 9, 17, 18, 19] else 0],
        'is_weekend': [1 if day in ['Saturday', 'Sunday'] else 0],
        'is_lunch_hour': [1 if hour in [12, 13] else 0]
    })
    
    # Encode categorical variables
    for feature in ['origin_zone', 'destination_zone', 'traffic_condition', 'day_of_week']:
        if feature in st.session_state.encoders:
            try:
                input_data[feature] = st.session_state.encoders[feature].transform(input_data[feature])
            except ValueError:
                # Handle unseen categories
                input_data[feature] = 0
    
    # Make prediction
    prediction = st.session_state.model.predict(input_data[st.session_state.feature_columns])
    return max(prediction[0], 5)  # Minimum 5 minutes

# Main app
def main():
    st.markdown('<div class="main-header">🚚 Delivery Time Predictor</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Model Training", "Make Prediction", "Analytics"])
    
    if page == "Model Training":
        st.header("📊 Model Training")
        
        st.write("""
        This system uses machine learning to predict delivery times based on:
        - **Origin and destination zones**
        - **Distance to travel**
        - **Time of day and day of week**
        - **Traffic conditions**
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Train Model", type="primary"):
                model, encoders, metrics, test_data = train_model()
                mae, rmse, r2 = metrics
                
                st.success("Model trained successfully!")
                
                # Display metrics
                st.subheader("Model Performance")
                col_mae, col_rmse, col_r2 = st.columns(3)
                
                with col_mae:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Mean Absolute Error</h4>
                        <h2>{mae:.2f} minutes</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_rmse:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Root Mean Square Error</h4>
                        <h2>{rmse:.2f} minutes</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_r2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>R² Score</h4>
                        <h2>{r2:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prediction vs Actual plot
                X_test, y_test, y_pred = test_data
                fig = px.scatter(x=y_test, y=y_pred, 
                               labels={'x': 'Actual Delivery Time (minutes)', 
                                      'y': 'Predicted Delivery Time (minutes)'},
                               title='Predicted vs Actual Delivery Times')
                fig.add_line(x=[y_test.min(), y_test.max()], 
                           y=[y_test.min(), y_test.max()], 
                           line_color='red', name='Perfect Prediction')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.session_state.model_trained:
                st.success("✅ Model Ready")
                st.info("Navigate to 'Make Prediction' to start using the model!")
            else:
                st.warning("⚠️ Model not trained yet")
    
    elif page == "Make Prediction":
        st.header("🎯 Delivery Time Prediction")
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first!")
            st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Route Information")
            origin = st.selectbox("Origin Zone", 
                                ['Downtown', 'Suburbs', 'Industrial', 'Residential', 'Commercial'])
            destination = st.selectbox("Destination Zone", 
                                     ['Downtown', 'Suburbs', 'Industrial', 'Residential', 'Commercial'])
            distance = st.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
        
        with col2:
            st.subheader("Timing Information")
            hour = st.slider("Hour of Day", min_value=6, max_value=22, value=14)
            day = st.selectbox("Day of Week", 
                             ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            traffic = st.selectbox("Traffic Condition", ['Light', 'Moderate', 'Heavy'])
        
        if st.button("🔮 Predict Delivery Time", type="primary"):
            prediction = predict_delivery_time(origin, destination, distance, hour, day, traffic)
            
            if prediction:
                # Convert to hours and minutes
                hours = int(prediction // 60)
                minutes = int(prediction % 60)
                
                if hours > 0:
                    time_str = f"{hours}h {minutes}m"
                else:
                    time_str = f"{minutes}m"
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>🕐 Estimated Delivery Time</h2>
                    <h1>{time_str}</h1>
                    <p>({prediction:.1f} minutes)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                st.subheader("📈 Delivery Insights")
                
                insights = []
                if traffic == 'Heavy':
                    insights.append("🚦 Heavy traffic may increase delivery time")
                if hour in [7, 8, 9, 17, 18, 19]:
                    insights.append("⏰ Rush hour - expect delays")
                if day in ['Saturday', 'Sunday']:
                    insights.append("📅 Weekend delivery - typically faster")
                if distance > 30:
                    insights.append("📏 Long distance delivery")
                if origin == 'Downtown' or destination == 'Downtown':
                    insights.append("🏙️ Downtown area - complex navigation")
                
                for insight in insights:
                    st.info(insight)
    
    elif page == "Analytics":
        st.header("📊 Delivery Analytics")
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first to see analytics!")
            st.stop()
        
        df = st.session_state.training_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average delivery time by zone
            zone_avg = df.groupby('origin_zone')['delivery_time_minutes'].mean().sort_values(ascending=False)
            fig1 = px.bar(x=zone_avg.index, y=zone_avg.values,
                         title='Average Delivery Time by Origin Zone',
                         labels={'x': 'Zone', 'y': 'Average Time (minutes)'})
            st.plotly_chart(fig1, use_container_width=True)
            
            # Traffic condition impact
            traffic_avg = df.groupby('traffic_condition')['delivery_time_minutes'].mean()
            fig3 = px.bar(x=traffic_avg.index, y=traffic_avg.values,
                         title='Average Delivery Time by Traffic Condition',
                         labels={'x': 'Traffic Condition', 'y': 'Average Time (minutes)'})
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Delivery time distribution
            fig2 = px.histogram(df, x='delivery_time_minutes', nbins=30,
                              title='Distribution of Delivery Times')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Delivery time by hour
            hourly_avg = df.groupby('hour_of_day')['delivery_time_minutes'].mean()
            fig4 = px.line(x=hourly_avg.index, y=hourly_avg.values,
                          title='Average Delivery Time by Hour of Day',
                          labels={'x': 'Hour', 'y': 'Average Time (minutes)'})
            st.plotly_chart(fig4, use_container_width=True)
        
        # Summary statistics
        st.subheader("📋 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Deliveries", len(df))
        with col2:
            st.metric("Average Time", f"{df['delivery_time_minutes'].mean():.1f} min")
        with col3:
            st.metric("Fastest Delivery", f"{df['delivery_time_minutes'].min():.1f} min")
        with col4:
            st.metric("Longest Delivery", f"{df['delivery_time_minutes'].max():.1f} min")

if __name__ == "__main__":
    main()