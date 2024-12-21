import folium
import numpy as np

def visualize(flight_net):

    '''
    Visualizes a flight network on a Folium map.

    Parameters:
        flight_net (networkx.Graph): A graph representing the flight network, 
        where nodes represent airports with 'latitude' and 'longitude' attributes,
        and edges represent flights with a 'passengers' attribute.

    Returns:
        folium.Map: An interactive map showing the flight network.
    '''

    # create a copy of the network and remove the nodes with invalid coordinates
    G = flight_net
    nodes_to_remove = {v for v, attributes in G.nodes(data=True) if attributes['latitude'] == 'none' or attributes['longitude'] == 'none'}
    G.remove_nodes_from(nodes_to_remove)

    # initialize a folium map centered on an average location (Center of the US)
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4) 

    passenger_values = [data['passengers'] for a, b, data in G.edges(data=True)]

    # max passenger count - for scaling edge thickness.
    M = max(passenger_values)

    # thresholds - to cathegorize edges colors
    p50 = np.percentile(passenger_values, 50)  # 50th percentile: median traffic
    p99 = np.percentile(passenger_values, 99) # 99th percentile: top 1% traffic

    # adding nodes from our graph as markers
    for v, data in G.nodes(data = True):
        if isinstance(data['latitude'], (int, float)) and isinstance(data['longitude'], (int, float)):
            folium.Marker(location=[data['latitude'], data['longitude']], popup = v).add_to(m)

    # adding edges as multylines :
    # define a different color based on passengers count 
    # and a different weight, proportionally to passangers count, scaled by the maximum
    for origin, destination, data in G.edges(data = True):

        origin_coords = G.nodes[origin]['latitude'], G.nodes[origin]['longitude']
        destination_coords = G.nodes[destination]['latitude'], G.nodes[destination]['longitude'] 

        if data['passengers'] < p50: 
            color = 'gray' 
        elif data['passengers'] > p99: 
            color = 'red' 
        else:
            color = 'green'

        folium.PolyLine(
            locations=[origin_coords, destination_coords],
            color = color,
            weight = data['passengers']/M    
        ).add_to(m)

    return m 
