import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# Load the taxi zone lookup data (assuming you have this CSV file)
taxi_zone_lookup = pd.read_csv('C:\\Users\\adnan\\OneDrive\\Desktop\\DP Web Project\\taxi_zone_lookup.csv')

# Create a dictionary to map LocationID to Zone
location_map_v = dict(zip(taxi_zone_lookup['LocationID'], taxi_zone_lookup['Zone']))

# Fitness function to calculate total revenue
def fitness(individual, pred_prices):
    total_revenue = 0
    for i in range(len(individual)):
        total_revenue += sum(pred_prices[i, :]) * individual[i]
    return total_revenue

# Function to calculate revenue per location
def revenue_per_location(individual, pred_prices):
    revenues = []
    for i in range(len(individual)):
        revenue_for_loc = sum(pred_prices[i, :]) * individual[i]
        revenues.append(revenue_for_loc)
    return np.array(revenues)

# Function to adjust vehicle allocation
def adjust_vehicle_allocation(vehicle_allocation, total_vehicles):
    rounded_allocation = np.floor(vehicle_allocation).astype(int)  # Use floor to prevent over-allocation at first
    current_total = np.sum(rounded_allocation)
    deficit = total_vehicles - current_total
    
    if deficit > 0:
        # Add vehicles to locations with the highest remaining fraction (before rounding)
        fractional_part = vehicle_allocation - np.floor(vehicle_allocation)
        for _ in range(deficit):
            idx = np.argmax(fractional_part)  # Find the location with the highest fractional part
            rounded_allocation[idx] += 1
            fractional_part[idx] = 0  # Set to 0 so it's not selected again
    elif deficit < 0:
        # Remove vehicles from locations with the smallest fractional part (before rounding)
        fractional_part = vehicle_allocation - np.floor(vehicle_allocation)
        for _ in range(abs(deficit)):
            idx = np.argmin(fractional_part)  # Find the location with the smallest fractional part
            if rounded_allocation[idx] > 0:
                rounded_allocation[idx] -= 1
            fractional_part[idx] = 1  # Set to 1 to avoid removing again

    return rounded_allocation


# Function to calculate demand
def calculate_demand(vehicle_allocation, revenues):
    # Avoid division by zero by replacing zeros in revenue with a small number (or another logic as needed)
    revenues_safe = np.where(revenues == 0, 1e-10, revenues)
    return vehicle_allocation / revenues_safe

# Function to get demand category thresholds
def classify_demand(redistributed_revenues):
    high_demand_threshold = np.percentile(redistributed_revenues, 50)
    medium_demand_threshold = np.percentile(redistributed_revenues, 25)
    low_demand_threshold = np.percentile(redistributed_revenues, 10)

    demand_categories = []
    for revenue in redistributed_revenues:
        if revenue >= high_demand_threshold:
            demand_categories.append("High")
        elif revenue >= medium_demand_threshold:
            demand_categories.append("Medium")
        else:
            demand_categories.append("Low")

    return demand_categories

# Function to get top 5 revenue locations
def get_top_5_revenue_locations(redistributed_revenues):
    top_5_indices = np.argsort(redistributed_revenues)[-5:][::-1]
    return top_5_indices, redistributed_revenues[top_5_indices]

def redistribute_vehicles_for_coverage(vehicle_allocation, demand, threshold=3):
    # Identify locations with high vehicle concentration and low vehicle allocation
    high_concentration_locs = np.where(vehicle_allocation > threshold)[0]
    zero_vehicle_locs = np.where(vehicle_allocation == 0)[0]
    
    for zero_loc in zero_vehicle_locs:
        if len(high_concentration_locs) == 0:
            break
        
        high_loc = high_concentration_locs[0]
        vehicle_allocation[high_loc] -= 1
        vehicle_allocation[zero_loc] += 1
        
        # Update high concentration locations after reducing vehicles
        if vehicle_allocation[high_loc] <= threshold:
            high_concentration_locs = np.delete(high_concentration_locs, 0)

    return vehicle_allocation


# Main function to run GA and return required data
# Main function to run GA and return required data, including vehicle redistribution
def run_ga_with_redistribution(pred_prices, num_locations, num_vehicles=500):
    population_size = 100
    num_generations = 100
    crossover_rate = 0.8
    mutation_rate = 0.05

    # Step 1: Run GA to get the best vehicle allocation
    population = np.random.randint(0, num_vehicles, (population_size, num_locations))
    population = np.array([ind / ind.sum() * num_vehicles for ind in population])

    best_individual = population[np.argmax([fitness(ind, pred_prices) for ind in population])]
    
    # Step 2: Adjust initial vehicle allocation
    adjusted_best_allocation = adjust_vehicle_allocation(best_individual, num_vehicles)
    
    # Step 3: Redistribute vehicles to improve coverage
    redistributed_allocation = redistribute_vehicles_for_coverage(adjusted_best_allocation, pred_prices, threshold=5)
    
    # Step 4: Calculate revenues after redistribution
    redistributed_revenues = revenue_per_location(redistributed_allocation, pred_prices)

    return redistributed_allocation, redistributed_revenues

# Call this new function in your app wherever you are using `run_ga`

location_ids = list(location_map_v.keys())  # List of all LocationIDs

# Function to plot top 5 high revenue locations
def plot_top_5_revenue_locations(top_5_indices, top_5_revenues, location_map_v, redistributed_allocation):

    # Plots the top 5 revenue-generating locations using a donut chart and maps the location IDs to their zone names.

    # Map the indices from the top 5 revenue locations to their actual LocationIDs
    top_5_location_ids = [location_ids[i] for i in top_5_indices]

    # Map LocationIDs to zone names using the location_map dictionary
    zone_names = [location_map_v.get(loc_id, f"Unknown ID {loc_id}") for loc_id in top_5_location_ids]

    # Labels for the donut chart - showing zone names and revenue
    labels = [f"{zone}\nVehicles: {redistributed_allocation[i]}\nRevenue: ${rev:.2f}" 
              for zone, rev, i in zip(zone_names, top_5_revenues, top_5_indices)]
    
    # Create a pie chart (which will become a donut chart)
    plt.figure(figsize=(6, 6))
    plt.pie(top_5_revenues, labels=labels, autopct='%1.1f%%', startangle=140, 
            colors=plt.cm.Paired.colors, wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
    
    # Add a circle at the center to create the donut shape
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Top 5 Revenue Generating Locations (Donut Chart)')
    plt.gca().set_aspect('equal')  # Equal aspect ratio to make the pie chart a circle
    
    # Render the donut chart using Streamlit
    st.pyplot(plt)

