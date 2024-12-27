# ADM - Homework 5: USA Airport Flight Analysis, Group #5

![image](https://github.com/user-attachments/assets/5c3545dc-19c4-4a48-8f68-b211378f0e72)

This GitHub repository contains the implementation of the fifth homework of the **Algorithmic Methods of Data Mining** course for the master's degree in Data Science at Sapienza (2024-2025). This homework was completed by Group #5. The details of the assignement are specified here:  https://github.com/Sapienza-University-Rome/ADM/tree/master/2024/Homework_5


**Team Members:**
* Valeria Avino, 1905974, avino.1905974@studenti.uniroma1.it
* Viktoriia Vlasenko, 2088928, vlasenko.2088928@studenti.uniroma1.it
* Evan Theodar, 2186832, evanthedor545@gmail.com

The ```main.ipynb``` with the main script can be visualized in the jupyter notebook viewer: 
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

- **Steps:**  
 
---

### **4. Airline Network Partitioning**
- **Steps:**  

---

### **5. Finding and Extracting Communities**
- **Steps:**  

---

### **Bonus Question - Connected Components on MapReduce**
- **Steps:**  

---


### **Algorithmic Question (AQ)**
- **Steps:**  

---


## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
