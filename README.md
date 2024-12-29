# ADM - Homework 5: USA Airport Flight Analysis, Group #5

![image](https://github.com/user-attachments/assets/5c3545dc-19c4-4a48-8f68-b211378f0e72)

This GitHub repository contains the implementation of the fifth homework of the **Algorithmic Methods of Data Mining** course for the master's degree in Data Science at Sapienza (2024-2025). This homework was completed by Group #5. The details of the assignement are specified here:  https://github.com/Sapienza-University-Rome/ADM/tree/master/2024/Homework_5


**Team Members:**
* Valeria Avino, 1905974, avino.1905974@studenti.uniroma1.it
* Viktoriia Vlasenko, 2088928, vlasenko.2088928@studenti.uniroma1.it
* Evan Theodar, 2186832, evanthedor545@gmail.com

The ```main.ipynb``` with the main script can be visualized in the [jupyter notebook viewer](https://nbviewer.org/github/vaal4ds/ADM_HM5/blob/main/main.ipynb)
## Repository Structure

```
├── functions/                                      : directory containing core project modules
│   ├── flight_net_analysis.py/                     : py file containing graph-related functions for Q1 and Q2
|   |   ├── create_graph                            : creates a directed graph from the airports dataset      
│   |   ├── analyze_graph_features                  : computes graph features -nodes, edges, density- and identifies hubs
│   |   ├── summarize_graph_features                : provides a summary report of the former results
|   |   ├── dijkstra                                : computes shortest paths evaluating distance in miles
|   |   ├── reconstruct_path                        : reconstructs paths in node hops
|   |   ├── calculate_betweenness                   : calculates Betweenness Centrality for each node
|   |   ├── pagerank                                : calculates PageRank for each node
|   |   ├── analyze_centrality                      : calculates Degree Centrality and Closeness Centrality
|   |   ├── compare_centralities                    : computes centralities for each node, visualizes distribution and higlights the highest ones 
|   |   ├── adj_matrix                              : computes the adjency matrix for a given graph 
|   |   └── eigenvector_centrality                  : computes and visualizes the eigenvector centrality distribution
│   ├── netmap.py\                                  : py file containing graph visualization function
│   |   └── visualize                               : creates an interactive geographic visualization of a given network
│   └── traffic.py\                                 : py file containing traffic related computations for Q1 and Q2
|       ├── passengers_flow                         : computes total passengers flow
|       ├── buisiest_routes                         : identifies buisiest routes by passengers flow
|       └── routes_efficiency                       : computes efficiency, highlights most efficient routes, under and over-utilized ones
├── main.ipynb                                      : Main notebook containg all the runned cells
├── .gitignore                                      : File to specify files and directories ignored by Git
├── README.md                                       : Project documentation
└── LICENSE                                         : License file for the project
```

Here are links to all the files:
* [main.ipynb](https://github.com/vaal4ds/ADM_HM5/blob/main/main.ipynb): The main notebook containing implementation and results
* [functions](https://github.com/vaal4ds/ADM_HM5/tree/main/functions): Contains modularized scripts for the homework
* [README.md](https://github.com/vaal4ds/ADM_HM5/blob/main/README.md): Project documentation
* [LICENSE](https://github.com/vaal4ds/ADM_HM5/blob/main/LICENSE): License file for the project

---

## Technologies Used

### Libraries and Frameworks:
* pandas - *Data storage and manipulation*
* matplotlib - *Visualizing distributions*
* numpy - *Efficient numerical calculations*
* seaborn - *Enhancing visualization aesthetics*
* networkx - *Graph creation and handeling only*
* folium - *Geographical visualization of the network on interactive maps*
* PySpark - *Distributed data processing and big data analytics*
* Community (community_louvain) - *Community detection in networks*

---

## Project Overview

### **0. EDA and data preprocessing**
Exploring dataset to ensure consistency in the creation of the flight network and installing all required libraries.

### **1. Flight Network Analysis** 
  The analysis begins by generating a directed graph of the flight network, where airports are nodes and flights are directed edges. Key graph features such as the number of nodes, edges, and the network's density are computed. Degree distributions for in-degree, out-degree, and total degree are evaluated and visualized to identify hubs—airports with a higher than 90th percentile degree.
  
  The busiest flight routes are identified based on total passenger flow, while passenger efficiency (passengers to seat ratio) helps highlight the most efficient routes. Over-utilized routesand under-utilized routes are also pinpointed.
  
  Finally, an interactive map is created to visualize the flight routes, providing a geographical perspective on the network.

---

### **2. Nodes' Contribution**

Computing four centrality measures for each airport in the flight network: Betweenness centrality, Closeness centrality, Degree centrality, and PageRank. These measures are then compared across all airports to identify their relative importance within the network. Histograms are plotted for each centrality to visualize the distribution of centrality values.

The top 5 airports for each centrality measure are identified. Additionally, alternative centrality measures are suggested, such as Eigenvector centrality, to offer further insights into the network. To evaluate the trustworthiness of these suggestions, validation methods like reviewing theoretical foundations and cross-referencing results with known benchmarks are applied.

Finally, Eigenvector Centrality Measure is implemented, and the results are compared with the original centrality values. The analysis assesses whether the new measure provides additional insights into the airport network’s structure or further refines the existing understanding.


---

### **3. Finding Best Routes**

 
---

### **4. Airline Network Partitioning**
- **Steps:**  

---

### **5. Finding and Extracting Communities**

Analyzing the flight network to detect communities using the **Louvain method**, and visualizing the largest one. Building a graph from flight data, running the community detection algorithm, and checking if two cities belong to the same community. The results include the total number of communities, a community overview, and a graph visualization. Lastly an alternative suggested by an LLM tool is implemented: the **Girvan-Newman method** and compared to the former.  

---

### **Bonus Question - Connected Components on MapReduce**
This PySpark script identifies connected components in a flight network using the MapReduce paradigm. It filters flights within a specified date range, creates an adjacency list of airport connections, and initializes labels for each airport. The labels are then propagated iteratively using MapReduce, where the minimum label is chosen for each connected pair of airports. After 10 iterations, connected components are grouped, and the number of components, their sizes, and the largest component are computed. Execution time is measured. To compare, GraphFrames’ `connectedComponents()` function can be used to find the same components, offering potentially better performance due to optimized graph algorithms.

---


### **Algorithmic Question (AQ)**
Implementation of an algorith to find the cheapest route from a starting city to a destination city with at most `k` stops, leveraging a priority queue to explore paths in increasing cost order. The graph is represented as an adjacency list where each city is connected by a flight with a specific cost. A priority queue stores routes with the current cost, city, and the number of stops made so far. The algorithm iterates through the queue, expanding cities while checking whether the destination is reached or if the number of stops exceeds the allowed limit. If a route is valid, the algorithm updates the visited states and pushes new paths into the queue. The algorithm terminates when the destination is found or no valid route exists.

The time complexity is **O((k + 1) * E * log E)**, where `E` is the number of edges (flights), and `k` is the number of allowed stops. Space complexity is **O(E + n * (k + 1))**, where `n` is the number of cities. For large graphs, this approach can become inefficient, especially if `k` is large. Optimizations like pruning and dynamic programming can help reduce redundant calculations and memory usage.


---


## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
