import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_graph_features(flight_network):
    '''

    '''
    # Count the number of nodes
    N = len(flight_network.nodes())
    # Count the number of edges in the graph
    E = len(flight_network.edges())
    # Compute the density for directed graph with loops
    density = E/(N**2)
    # Determine if the graph is sparse or dense based on its density.
    tresh1 = 0.1
    tresh2 = 0.5
    
    if density<tresh1:
       density_type = 'sparse'

    elif density>tresh2:
        density_type = 'dense'

    else:
        density_type = 'moderate'

    # Compute in-degree, out-degree and degree for each node
    in_degrees = dict(flight_network.in_degree())
    out_degrees = dict(flight_network.out_degree())
    degrees = dict(flight_network.degree())

    # Identify airports with degrees higher than the 90th percentile and list them as "hubs."
    perc_90 = np.percentile(list(degrees.values()), 90) 
    hubs = { v : degree for v, degree in degrees.items() if degree > perc_90 }
    hubs_df = pd.DataFrame( hubs.items(), columns=['Airport', 'Degree'] )

    # Return all computed features in a dictionary
    features = {
        'N': N,
        'E': E,
        'density': density,
        'density_type': density_type,
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
        'hubs_df': hubs_df
    }

    return features

    
def summarize_graph_features(flight_network):
    ''' generates a detailed report of the graph's features'''

    features = analyze_graph_features(flight_network)

    print(f"Number of nodes N = {features['N']}")
    print(f"Number of edges E = {features['E']}")
    print(f"Density D = {features['density']}")
    print(f"The graph is {features['density_type']} (Density = {features['density']})")

    # Plot histograms for in-degree and out-degree
    plt.hist(list(features['in_degrees'].values()), bins=15, alpha=0.5, label='In-degree')
    plt.show()
    plt.hist(list(features['out_degrees'].values()), bins=15, alpha=0.5, label='Out-degree')
    plt.show()

    # Display hubs (airports with degree higher than 90th percentile)
    print("\nHubs (airports with degree > 90th percentile):")
    print(features['hubs_df'])


    
 