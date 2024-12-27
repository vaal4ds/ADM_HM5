import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import heapq
from IPython.display import display


def create_graph(data): 
    
    ''' 
    This function creates a directed graph (DiGraph) from flight data. 
    Each airport is represented as a node, and each flight route as a directed edge. 
    Node attributes include latitude and longitude of the airports. 
    Edges are created with attributes for distance, number of flights, and number of passengers.
    '''
    
    # initializing an empty graph object
    G = nx.DiGraph()

    # creating the set of nodes: V
    V = set(data['Origin_airport']).union(data['Destination_airport'])

    # preparing locations (latitude and longitude) to assign as node attributes
    airports= pd.concat([
        data[['Origin_airport', 'Org_airport_lat', 'Org_airport_long']].rename(columns={
        'Origin_airport': 'Airport',
        'Org_airport_lat': 'Latitude',
        'Org_airport_long': 'Longitude'
        }),
        data[['Destination_airport', 'Dest_airport_lat', 'Dest_airport_long']].drop_duplicates().rename(columns={
        'Destination_airport': 'Airport',
        'Dest_airport_lat': 'Latitude',
        'Dest_airport_long': 'Longitude'
        })
    ]).drop_duplicates().set_index('Airport')[['Latitude', 'Longitude']].to_dict('index')

    # adding nodes with location as attribute
    for v in V:
        if v in airports:
            lat, long = airports[v]['Latitude'], airports[v]['Longitude']
            G.add_node(v, latitude = lat, longitude = long)

            #if Nan, assign "none" as a location
            if pd.isna(lat) or pd.isna(long):
                G.add_node(v, latitude = 'none', longitude = 'none')
            else:
                G.add_node(v, latitude = lat, longitude = long)

        else:
            G.add_node(v) #if not in airports

    # adding edges with distance as weight and flights as attributes:
    for indx, row in data.iterrows():
        G.add_edge( row['Origin_airport'], row['Destination_airport'],
                            weight = row['Distance'], 
                            flights = row['Flights'],
                            passengers = row['Passengers']
                            )
    return G


def analyze_graph_features(flight_network):
    '''
    This function computes the main feature of the given flight_network
    '''
    # Count the number of nodes
    N = len(flight_network.nodes())
    # Count the number of edges in the graph
    E = len(flight_network.edges())
    # Compute the density for directed graph with loops
    density = 2*E/(N*(N-1))
    # Determine if the graph is sparse or dense based on its density.
    tresh1 = 0.15
    tresh2 = 0.5
    
    if density<tresh1:
       density_type = 'sparse'

    elif density>tresh2:
        density_type = 'dense'

    else:
        density_type = 'moderatly dense'

    # Compute in-degree, out-degree and degree for each node iterating on edges
    in_degrees = {v: 0 for v in flight_network.nodes() }
    out_degrees = {v: 0 for v in flight_network.nodes() }

    for u, v in flight_network.edges():
        out_degrees[u] +=1 # out from u
        in_degrees[v] += 1 # in in v

    degrees = {v : in_degrees[v]+ out_degrees[v] for v in flight_network.nodes()}


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
        'degrees':degrees,
        'hubs_df': hubs_df
    }

    return features
    

def summarize_graph_features(flight_network):
    ''' generates a detailed report of the graph's features'''

    features = analyze_graph_features(flight_network)
    print("Summary Report:\n"
          f"Number of nodes N = {features['N']}\n"
          f"Number of edges E = {features['E']}\n"
          f"Density D = {features['density']}\n"
          f"The graph is {features['density_type']}"
          )

    # histogram of in-degree distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].hist(list(features['in_degrees'].values()), bins=25, alpha=0.5, label='In-degree', color='magenta')
    axes[0].set_title('In-degree Distribution')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Frequency')

    # histogram of out-degree distribution
    axes[1].hist(list(features['out_degrees'].values()), bins=25, alpha=0.5, label='Out-degree', color='cyan')
    axes[1].set_title('Out-degree Distribution')
    axes[1].set_xlabel('Degree')
    axes[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Display hubs (airports with degree higher than 90th percentile)
    print(f"Hubs (airports with degree > 90th percentile):\n")
    display(features['hubs_df'])


def dijkstra(graph, source):
    # Initialize distances and predecessors
    distances = {node: float('inf') for node in graph.nodes}
    distances[source] = 0
    predecessors = {node: [] for node in graph.nodes}  # Predecessor tracking for path reconstruction
    priority_queue = [(0, source)]  # Priority queue initialized with the source node
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight['weight']
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = [current_node]  # Record the predecessor
                heapq.heappush(priority_queue, (distance, neighbor))
            elif distance == distances[neighbor]:
                predecessors[neighbor].append(current_node)
    
    return distances, predecessors


def reconstruct_path(predecessors, u, v):
    """
    Iteratively reconstruct paths between two nodes.
    """
    stack = [(v, [v])]
    paths = []

    while stack:
        current_node, path = stack.pop()

        if current_node == u:
            paths.append(path[::-1])  # Reverse the path to start from u
            continue

        for predecessor in predecessors.get(current_node, []):
            stack.append((predecessor, path + [predecessor]))

    return paths


def calculate_betweenness(graph):
    betweenness = {node: 0 for node in graph.nodes}
    # For each node, run Dijkstra's algorithm and compute betweenness
    for source in graph.nodes:
        distances, predecessors = dijkstra(graph, source)
        for target in graph.nodes:
            if source != target:
                # Find all shortest paths between source and target
                all_paths = reconstruct_path(predecessors, source, target)
                # Update betweenness centrality for nodes on the shortest path
                for path in all_paths:
                    for node in path[1:-1]:  # Exclude source and target from the middle nodes
                        betweenness[node] += 1

    # Normalize the betweenness centrality
    N = len(graph.nodes)
    for node in betweenness:
        betweenness[node] /= (N - 1) * (N - 2)  # Normalized betweenness

    return betweenness


def pagerank(graph, airport, weight='distance'):
    '''
    Support function to compute the PageRank of the given airport with the distance (in miles) attribute as a weight
    '''
    # Default values
    alpha = 0.85
    max_iter = 100
    tol = 1.0e-6

    V = list(graph.nodes())  # Convert to a list to access nodes
    N = len(V)

    pagerank_values = {node: 1 / N for node in V}  # Initialize uniform pagerank values

    for _ in range(max_iter):
        new_pagerank_values = pagerank_values.copy()
        change = 0

        for node in V:
            if node == airport:
                continue  # Skip the airport node itself

            rank_sum = 0
            for neighbor in graph.predecessors(node):  # Accessing predecessors of a node
                edge_data = graph.get_edge_data(neighbor, node)
                weight_val = edge_data.get(weight, 1)  # Get weight from edge attribute (default to 1 if not found)
                out_degree = len(list(graph.successors(neighbor)))  # Number of outgoing edges for the neighbor
                rank_sum += pagerank_values[neighbor] * weight_val / out_degree

            new_pagerank_values[node] = (1 - alpha) / N + alpha * rank_sum
            change += abs(new_pagerank_values[node] - pagerank_values[node])

        pagerank_values = new_pagerank_values
        
        if change < tol:
            break

    return pagerank_values[airport]


def analyze_centrality(flight_network, airport):

    V = list(flight_network.nodes())
    N = len(V)
    features = analyze_graph_features(flight_network)
    degrees = features['degrees']

    # Compute all shortest paths from the airport to all other nodes
    distances, predecessors = dijkstra(flight_network, airport)

    # Betweenness Centrality
    betweenness = calculate_betweenness(flight_network)

    # Closeness Centrality
    tot_d_airport = sum(distances[v] for v in V if v != airport and distances[v] != float('inf'))
    if tot_d_airport > 0:
        closeness_centrality = (N - 1) / tot_d_airport
    else:
        closeness_centrality = 0  

    # Degree Centrality 
    degree_centrality = degrees[airport] / (N - 1)  

    # PageRank
    airport_pagerank = pagerank(flight_network, airport, weight='distance')

    # Return centralities
    centralities = {
        'Betweenness': betweenness.get(airport, 0),
        'Closeness': closeness_centrality,
        'Degree': degree_centrality,
        'PageRank': airport_pagerank
    }

    return centralities

def compare_centralities(flight_network):
    '''
    Computes and compares centrality values for all nodes in the graph.
    Plots centrality distributions (histograms for each centrality measure).
    Returns the top 5 airports for each centrality measure.
    '''
    centralities = {v: analyze_centrality(flight_network, v) for v in flight_network.nodes()}

    # Extract individual centrality values
    BC = {v: centralities[v]['Betweenness'] for v in flight_network.nodes()}
    CC = {v: centralities[v]['Closeness'] for v in flight_network.nodes()}
    DC = {v: centralities[v]['Degree'] for v in flight_network.nodes()}
    PR = {v: centralities[v]['PageRank'] for v in flight_network.nodes()}

    # Plotting histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    centrality_data = [
        (BC, 'Betweenness', 'magenta'),
        (CC, 'Closeness', 'blue'),
        (DC, 'Degree', 'cyan'),
        (PR, 'PageRank', 'magenta')
    ]
    
    for ax, (data, title, color) in zip(axes.flatten(), centrality_data):

        ax.hist(list(data.values()), bins=25, alpha=0.5, label=title, color=color)
        ax.set_title(f'{title} Centrality')
        ax.set_xlabel('Centrality')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    top_5_BC = sorted(BC.items(), key=lambda item: item[1], reverse=True)[:5]
    top_5_CC = sorted(CC.items(), key=lambda item: item[1], reverse=True)[:5]
    top_5_DC = sorted(DC.items(), key=lambda item: item[1], reverse=True)[:5]
    top_5_PR = sorted(PR.items(), key=lambda item: item[1], reverse=True)[:5]

    # Create a DataFrame for the top 5 results
    results_table = pd.DataFrame({
        'Betweenness': [f"{airport} ({score:.5f})" for airport, score in top_5_BC],
        'Closeness': [f"{airport} ({score:.5f})" for airport, score in top_5_CC],
        'Degree': [f"{airport} ({score:.5f})" for airport, score in top_5_DC],
        'PageRank': [f"{airport} ({score:.5f})" for airport, score in top_5_PR]
    })

    # Display the table
    print("\nTop 5 Airports for Each Centrality Measure:")

    return results_table



def adj_matrix(graph):

    '''
    Builds  adjacency matrix from a directed graph
    '''
    nodes = list(graph.nodes)
    node_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    #initialize matrix with zeros
    adj_matrix = np.zeros((n, n))
    #add ones only if there is an edge 
    for node in graph:
        for neighbor in graph[node]:
            adj_matrix[node_idx[node], node_idx[neighbor]] = 1
    
    return adj_matrix, node_idx


def eigenvector_centrality(adj_matrix, max_iter=100, tol=1e-6):
    '''
    Computes eigenvector centrality for a graph represented by an adjacency matrix.

    Parameters:
        adj_matrix (np.ndarray): The adjacency matrix of the graph.
        max_iter (int): Maximum number of iterations for convergence.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Eigenvector centrality scores for each node.
    '''    
    # Number of nodes
    n = adj_matrix.shape[0]
    
    # Initialize centrality vector
    x = np.ones(n)
    
    # Iteratively compute centrality
    for _ in range(max_iter):
        # Multiply adjacency matrix with centrality vector x_new= Ax
        x_new = np.dot(adj_matrix, x)
        
        # Normalize 
        x_new = x_new / np.linalg.norm(x_new, ord=2)
        
        # repeat until convergence
        if np.linalg.norm(x_new - x, ord=1) < tol:
            break
        
        x = x_new
        
    #  visualizing results
    plt.figure(figsize=(8, 6))
    plt.hist(x, bins=25, alpha=0.5, color='magenta')
    plt.title('Eigenvector Centrality')
    plt.xlabel('Centrality')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return x

