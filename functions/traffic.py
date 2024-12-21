import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def passengers_flow(data):

    '''
    Computes the total passenger flow for each directed route in the dataset.
    ( flights from A to B and B to A are not considered as the same ). 

    Parameters: data (pd.DataFrame): Input DataFrame 

    Returns: pd.DataFrame ( updated with additional columns for route and passenger flow)
    '''

    # creates `route` column to represent directed connections using airport pairs as (Origin_airport, Destination_airport)
    data['route'] = data.apply(
    lambda row: (row['Origin_airport'], row['Destination_airport']), axis=1
    )

    # sums the `Passengers` column for each directed route
    flow_by_route = data.groupby('route')['Passengers'].sum()
    
    # maps the total passenger flow for each route back to the original dataset as `flow_by_route`.
    data['flow_by_route'] = data['route'].map(flow_by_route)

    return data


def buisiest_routes(data: pd.DataFrame , k: int):
    '''
    Computes the total passenger flow for each directed route 
    using the `passengers_flow` function. It then extracts the top-k routes with the 
    highest total passenger flow_by_route and creates a bar chart to visualize the results.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing flight information.
        k (int): The number of busiest routes to extract and visualize.

    Returns:
        pd.DataFrame: A DataFrame containing the top-k busiest routes 
    '''

    # computes passenger flow and total flow by route using the `passengers_flow` function
    data = passengers_flow(data)
    # selects the top-k routes with the highest passenger flow and removes duplicates
    top_k = data[['route','flow_by_route']].sort_values(by='flow_by_route', ascending=False).drop_duplicates().head(k)
    
    # generates a bar chart to display the passenger flow of the top-k busiest routes
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x = top_k['route'].astype(str), y = top_k['flow_by_route'])
    ax.set_xlabel('Route')
    ax.set_ylabel('Passenger Flow')
    ax.set_title(f'Top {k} Busiest Routes by Passenger Flow')
    ax.tick_params(axis='x', rotation=45, labelsize=10)  
    ax.tick_params(axis='y', labelsize=10)  
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return top_k


def routes_efficiency(data: pd.DataFrame , k: int):
    '''
    Calculate the passenger efficiency as the average passengers per flight for each route and highlight under/over-utilized connections.
    The connection utilization of each route is computed by comparing the ratio of passengers to available seats. 
    It identifies and visualizes the top `k` over-utilized and under-utilized routes based on the passenger-to-seat ratio.
    
    '''    

    # counting total flights number and total passengers for every directed route
    flights_number = data.groupby('route')['Flights'].sum()
    all_pass = data.groupby('route')['Passengers'].sum()
    # aritmetic average
    average_passengers_per_route = np.round(all_pass/flights_number, 2)
    data['Average_passengers'] = data['route'].map(average_passengers_per_route)
    # selecting highest 
    top_k = data[['route','Average_passengers'] ].sort_values(by = 'Average_passengers', ascending=False).head()

    # prints the top `k` routes by passengers efficiency
    print(f"Top {k} routes by passenger efficiency :\n"
          f"{top_k}")

    # barchart of top_k most efficient routes 
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x = top_k['route'].astype(str), y = top_k['Average_passengers'])
    ax.set_xlabel('Route')
    ax.set_ylabel('Passenger Efficiency (average passengers per flight)')
    ax.set_title(f'Top {k} Busiest Routes by Passenger Efficiency')
    ax.tick_params(axis='x', rotation=45, labelsize=10)  
    ax.tick_params(axis='y', labelsize=10)  
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    
    # defining 'connectin utilization' as the rate of passengers (demand) over seats (availability)

    # top k over-utilized: sorting by highest utilization ratio 
    over_utilized = data[['route','Passengers', 'Seats']][data['Seats'] > 0].assign(utilization=(data['Passengers'] +0.0001) / data['Seats']) \
    .sort_values(by='utilization', ascending = False) \
    .drop_duplicates() \
    .head(k)
    
    # top k under-utilzed: sorting by lowest utilization ratio
    under_utilized = data[['route','Passengers', 'Seats']][data['Seats'] > 0].assign(utilization=(data['Passengers'] +0.0001) / data['Seats']) \
    .sort_values(by='utilization') \
    .drop_duplicates() \
    .head(k)

    # Print the top `k` over-utilized and under-utilized tables
    print(f"Top {k} over-utilized routes:\n"
          f"{over_utilized}")
    print(f"Top {k} under-utilized routes:\n"
          f"{under_utilized}")