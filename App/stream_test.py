import streamlit as st
import numpy as np
import pandas as pd
import pickle
import random
import os  # For file checking and operations

from vehicle_allocation import run_ga_with_redistribution, get_top_5_revenue_locations, plot_top_5_revenue_locations, calculate_demand,classify_demand, location_map_v  # Import the demand/revenue logic from vehicle_allocation.py

# Load your trained XGBoost model
model = pickle.load(open('C:\\Users\\adnan\\OneDrive\\Desktop\\DP Web Project\\xgb_model.pkl', 'rb'))

# Load the taxi zone lookup data (include borough and zone names)
taxi_zone_lookup = pd.read_csv('C:\\Users\\adnan\\OneDrive\\Desktop\\DP Web Project\\taxi_zone_lookup.csv')

historical_data = pd.read_parquet('C:\\Users\\adnan\\OneDrive\\Desktop\\DP Web Project\\data_pp.parquet')

# Create a dictionary to map Zone names to Location IDs and boroughs
location_map = dict(zip(taxi_zone_lookup['Zone'], taxi_zone_lookup['LocationID']))
borough_map = dict(zip(taxi_zone_lookup['LocationID'], taxi_zone_lookup['Borough']))

#============================================================================== 
# CSS 
#==============================================================================
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #000;
        color: #fff;
        padding: 10px 24px;
        border: 1;
        border-color: #000;
        border-radius: 5px;
        cursor: pointer;
        display: block; 
        margin: 0 auto;
        margin-top: 15px;
    }
    .stButton>button:hover {
        background-color: #fff;
        color: #000;
        border-color: #000;
    }
    </style>
    """, unsafe_allow_html=True
)

#==============================================================================

# Define your Streamlit app
st.markdown(
    """
    <h1 style='text-align: center; font-size: 30px;'>E-hailing Taxi Fare Prediction App with Demand and Revenue</h1>
    """, 
    unsafe_allow_html=True
)

# Dropdown selections for Pickup and Dropoff Zones
pickup_zone = st.selectbox('Select Pickup Zone', taxi_zone_lookup['Zone'].unique())
dropoff_zone = st.selectbox('Select Dropoff Zone', taxi_zone_lookup['Zone'].unique())

# Map the selected zones back to numeric Location IDs and boroughs for model input
PULocationID = location_map.get(pickup_zone)
DOLocationID = location_map.get(dropoff_zone)
pickup_borough = borough_map.get(PULocationID)
dropoff_borough = borough_map.get(DOLocationID)

# Input fields for other user data
pickup_time = st.selectbox('Pickup Time', ['Morning', 'Afternoon', 'Evening', 'Night'])

# Create three columns to position the checkboxes side by side
col1, col2, col3 = st.columns(3)
# Toggle switches (using checkboxes for binary values) placed side by side
with col1:
    is_weekend = st.toggle('Weekend')
with col2:
    shared_match_flag = st.toggle('Shared Ride')
with col3:
    wav_match_flag = st.toggle('Wheelchair Accessible')

# Map user inputs to numerical values
pickup_time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
weekday = 0 if is_weekend else 1  # If weekend is checked, set weekday to 0

# Convert checkboxes (toggles) to binary flags
shared_match_flag = 1 if shared_match_flag else 0
wav_match_flag = 1 if wav_match_flag else 0

# Initialize demand/revenue data by running the genetic algorithm (GA)
predicted_fares = np.load('C:\\Users\\adnan\\OneDrive\\Desktop\\DP Web Project\\predicted_fares.npy')  # Load the predicted fare matrix
num_locations = predicted_fares.shape[0]  # Number of locations

# Run the Genetic Algorithm to calculate demand and revenue data
redistributed_allocation, redistributed_revenues = run_ga_with_redistribution(predicted_fares, num_locations, num_vehicles=300)
demand_per_location = calculate_demand(redistributed_allocation, redistributed_revenues)
demand_categories = classify_demand(redistributed_revenues)

# Function to reset trip time only, but keep trip distance
def reset_trip_time():
    st.session_state['avg_trip_time'] = None

# Initialize session state for trip time, trip distance, and previously selected locations
if 'avg_trip_time' not in st.session_state:
    reset_trip_time()
if 'avg_trip_km' not in st.session_state:
    st.session_state['avg_trip_km'] = None

if 'prev_PULocationID' not in st.session_state:
    st.session_state['prev_PULocationID'] = None
if 'prev_DOLocationID' not in st.session_state:
    st.session_state['prev_DOLocationID'] = None

if 'pickup_time' not in st.session_state:
    st.session_state['pickup_time'] = 'Morning'

# Add realistic trip time calculation function
def calculate_realistic_trip_time(pickup_borough, dropoff_borough, distance_km, pickup_time, is_weekend):
    """
    Calculate realistic trip time based on distance and congestion factors, keeping distance constant.
    """
    # Base average speeds for different areas
    if pickup_borough == dropoff_borough:
        if distance_km < 5:
            avg_speed_kmh = 30  # Urban short distances
        elif distance_km <= 10:
            avg_speed_kmh = 40  # Urban longer distances
        else:
            avg_speed_kmh = 50  # Suburban longer distances
    else:
        # Assume suburban or highway for different boroughs
        avg_speed_kmh = 70 if distance_km <= 20 else 80  # Highway travel

    # Calculate time in hours based on the constant distance
    trip_time_hours = distance_km / avg_speed_kmh

    # Convert time to seconds
    trip_time_seconds = trip_time_hours * 3600

    # Apply congestion factor based on the time of day and weekend
    congestion_factor = 1.0  # Default congestion factor
    if is_weekend == 0:
        if pickup_time == 'Morning':  
            congestion_factor = random.uniform(0.9, 1.1)  # Lighter traffic in the morning
        elif pickup_time == 'Afternoon':  
            congestion_factor = random.uniform(1.1, 1.3)  # Heavier afternoon traffic
        elif pickup_time == 'Evening':  
            congestion_factor = random.uniform(1.2, 1.4)  # Evening traffic
        elif pickup_time == 'Night':  
            congestion_factor = random.uniform(0.8, 1.0)  # Lighter traffic at night
    else:
        if pickup_time == 'Morning':  
            congestion_factor = random.uniform(1.2, 1.4)  # Rush hour
        elif pickup_time == 'Afternoon':  
            congestion_factor = random.uniform(1.2, 1.4)  # Rush hour
        elif pickup_time == 'Evening':  
            congestion_factor = random.uniform(1.2, 1.4)  # Rush hour
        elif pickup_time == 'Night':  
            congestion_factor = random.uniform(0.7, 0.9)  # Lighter traffic

    # Adjust the trip time based on congestion factor
    adjusted_trip_time_seconds = trip_time_seconds * congestion_factor

    return adjusted_trip_time_seconds

# Function to get trip time and distance based on historical data from CSV or realistic values
def get_trip_time_and_km(PULocationID, DOLocationID, pickup_borough, dropoff_borough, pickup_time, is_weekend):
    # Check if the pickup or dropoff location has changed since the last prediction
    if (st.session_state['prev_PULocationID'] != PULocationID or 
        st.session_state['prev_DOLocationID'] != DOLocationID):
        # If locations have changed, get historical distance
        trip_data = historical_data[
            (historical_data['PULocationID'] == PULocationID) & 
            (historical_data['DOLocationID'] == DOLocationID)
        ]
        if not trip_data.empty:
            # Use the average distance for similar trips
            st.session_state['avg_trip_km'] = trip_data['avg_trip_km'].mean()
        else:
            # If no historical data is found, use a default random distance
            st.session_state['avg_trip_km'] = random.uniform(5.0, 15.0)

        reset_trip_time()

        # Save the current locations as the previous ones in session state
        st.session_state['prev_PULocationID'] = PULocationID
        st.session_state['prev_DOLocationID'] = DOLocationID

    # Update only the trip time based on the pickup_time and weekend factor
    st.session_state['avg_trip_time'] = calculate_realistic_trip_time(
        pickup_borough, dropoff_borough, st.session_state['avg_trip_km'], pickup_time, is_weekend
    )
    
    return st.session_state['avg_trip_time'], st.session_state['avg_trip_km']

# Predict the total fare, trip time, and trip distance
if st.button('Predict Fare'):
    # Check if pickup and dropoff locations are the same
    if PULocationID == DOLocationID:
        st.warning("Pickup and Dropoff locations are the same. Please change the Dropoff location.")
    else:
        # Compute or get trip_time and trip_km based on historical data or realistic values
        avg_trip_time, avg_trip_km = get_trip_time_and_km(
            PULocationID, DOLocationID, pickup_borough, dropoff_borough, pickup_time, is_weekend
        )
       
    input_data = np.array([[PULocationID, DOLocationID, pickup_time_mapping[pickup_time], 
                            weekday, shared_match_flag, wav_match_flag, avg_trip_time, avg_trip_km]])

    # Ensure input_data is of the correct dtype (float32)
    input_data = input_data.astype(np.float32)

    # Predict the fare
    fare_prediction = model.predict(input_data)
    
    st.write(f"Initial Fare Prediction: {fare_prediction[0]:.2f}")
    st.write(f"Shared Ride Flag: {shared_match_flag}")
    
    # If wheelchair accessible is selected (wav_match_flag == 1)
    if wav_match_flag == 1:
    # Apply wheelchair surcharge
        wheelchair_surcharge_percentage = random.uniform(5, 10) / 100  # Random surcharge between 5% and 10%
        fare_with_surcharge = fare_prediction[0] + (fare_prediction[0] * wheelchair_surcharge_percentage)
        fare_prediction[0] = fare_with_surcharge  # Apply the surcharge
    
    # Add additional time for wheelchair access
    extra_time_seconds = random.uniform(300, 600)  # Add 5 to 10 minutes (300 to 600 seconds)
    avg_trip_time += extra_time_seconds  # Add the extra time to the trip duration

    # If shared ride is selected (shared_match_flag == 1)
    if shared_match_flag == 1:
        fare_reduction_percentage = random.uniform(15, 20) / 100  # Random percentage between 3% and 7%
        reduced_fare = fare_prediction[0] - (fare_prediction[0] * fare_reduction_percentage)
        fare_prediction[0] = max(0, reduced_fare)  # Ensure fare does not go below 0
    
    # Store the results in session state
    st.session_state['prediction_data'] = {
            'PULocationID': PULocationID,
            'DOLocationID': DOLocationID,
            'Pickup Time': pickup_time,
            'Weekend': 'Yes' if is_weekend else 'No',
            'Shared Ride': 'Yes' if shared_match_flag else 'No',
            'Wheelchair Accessible': 'Yes' if wav_match_flag else 'No',
            'Predicted Total Fare': fare_prediction[0],
            'Predicted Trip Time': avg_trip_time,
            'Predicted Trip Distance': avg_trip_km
    }

    # Display the results as cards using HTML and CSS
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: row; justify-content: space-around;">
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(52, 73, 
94,0.3);">
                    <h4>Total Fare</h4>
                    <p>${fare_prediction[0]:.2f}</p>
                </div>
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(52, 73, 94,0.3); text-align: center">
                    <h4>Trip Time</h4>
                    <p>{avg_trip_time:.2f} seconds</p>
                </div>
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(52, 73, 94,0.3); text-align: center">
                    <h4>Trip Distance</h4>
                    <p>{avg_trip_km:.2f} km</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Display demand and revenue for the selected pickup location (PULocationID)
    demand = demand_per_location[PULocationID - 1]  # Adjust for zero-indexing
    revenue = redistributed_revenues[PULocationID - 1]
    demand_category = demand_categories[PULocationID - 1]

    # Determine background color based on demand category
    if demand_category == "High":
        demand_color = "#4caf50"  # Green for high demand
    elif demand_category == "Medium":
        demand_color = "#ffeb3b"  # Yellow for medium demand
    else:
        demand_color = "#f44336"  # Red for low demand

    # Render the card with the updated styles
    st.markdown(
    f"""
    <div style="background-color: #e0f7fa; padding: 15px; border-radius: 10px; margin-top: 20px;">
        <h4 style="margin-bottom: 10px;">Demand and Revenue for Pickup Location {pickup_zone}</h4>
        <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center;">
            <div style="background-color: #ffffff; padding: 5px; border-radius: 10px; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); margin-right: 15px; text-align: center; width: 45%;">
                <h4 style="margin-bottom: 5px;">Revenue</h4>
                <p style="font-size: 24px; font-weight: bold;">${revenue:.2f}</p>
            </div>
            <div style="background-color: {demand_color}; padding: 5px; border-radius: 10px; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); text-align: center; width: 45%;">
                <h4 style="margin-bottom: 5px;">Demand Category</h4>
                <p style="font-size: 24px; font-weight: bold; color: #ffffff;">{demand_category}</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
    )
# Plot the top 5 high-revenue locations
if st.button("Show Top 5 Revenue Locations"):
    # After getting the top 5 indices and revenues
    top_5_indices, top_5_revenues = get_top_5_revenue_locations(redistributed_revenues)
    # Plot the top 5 revenue-generating locations with zone names   
    plot_top_5_revenue_locations(top_5_indices, top_5_revenues, location_map_v, redistributed_allocation)

# Save to CSV when Save button is clicked
if st.button('Save'):
    if 'prediction_data' in st.session_state:
        prediction_df = pd.DataFrame([st.session_state['prediction_data']])
        file_path = 'C:\\Users\\adnan\\OneDrive\\Desktop\\DP Web Project\\predicted_fare_data.csv'

        try:
            if not os.path.exists(file_path):
                prediction_df.to_csv(file_path, index=False)
                st.write("Data saved to a new CSV file.")
            else:
                # Append to existing CSV
                prediction_df.to_csv(file_path, mode='a', header=False, index=False)
                st.write("Data appended to the existing CSV file.")
        except Exception as e:
            st.error(f"Error saving data: {e}")
    else:
        st.warning("Please predict the fare before saving the data.")

