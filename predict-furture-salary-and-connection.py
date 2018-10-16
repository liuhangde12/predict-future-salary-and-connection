
# coding: utf-8

# # Company Emails
# 
# For this project you will be workking with a company's email network where each node corresponds to a person at the company, and each edge indicates that at least one email has been sent between two people.
# 
# The network also contains the node attributes `Department` and `ManagementSalary`.
# 
# `Department` indicates the department in the company which the person belongs to, and `ManagementSalary` indicates whether that person is receiving a management position salary.

# In[ ]:


import networkx as nx
import pandas as pd
import numpy as np
import pickle

G = nx.read_gpickle('email_prediction.txt')

print(nx.info(G))


# ### Part A - Salary Prediction
# 
# Using network `G`, identify the people in the network with missing values for the node attribute `ManagementSalary` and predict whether or not these individuals are receiving a management position salary.
# 
# To accomplish this, you will need to create a matrix of node features using networkx, train a sklearn classifier on nodes that have `ManagementSalary` data, and predict a probability of the node receiving a management salary for nodes where `ManagementSalary` is missing.
# 
# 
# 
# Your predictions will need to be given as the probability that the corresponding employee is receiving a management position salary.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC).
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUC of 0.88 or higher will receive full points, and with an AUC of 0.82 or higher will pass (get 80% of the full points).
# 
# Using your trained classifier, return a series of length 252 with the data being the probability of receiving management salary, and the index being the node id.
# 
#     Example:
#     
#         1       1.0
#         2       0.0
#         5       0.8
#         8       1.0
#             ...
#         996     0.7
#         1000    0.5
#         1001    0.0
#         Length: 252, dtype: float64

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

def salary_predictions():
    
    def is_management(node) :
        ManagementSalary = node[1]['ManagementSalary']
        if ManagementSalary == 0:
            return 0
        elif ManagementSalary == 1:
            return 1
        else:
            return None
        
    df = pd.DataFrame(index=G.nodes())
    df['clustering'] = pd.Series(nx.clustering(G))
    df['degree'] = pd.Series(nx.degree(G))
    df['degree_centrality'] = pd.Series(nx.degree_centrality(G))
    df['closeness'] = pd.Series(nx.closeness_centrality(G, normalized= True))
    df['betweenness'] = pd.Series(nx.betweenness_centrality(G, normalized=True))
    df['pr'] = pd.Series(nx.pagerank(G))
    df['is_management'] = pd.Series([is_management(node) for node in G.nodes(data=True)])
    
    df_train = df[~pd.isnull(df['is_management'])]
    df_test = df[pd.isnull(df['is_management'])]
    features = ['clustering', 'degree', 'degree_centrality', 'closeness', 'betweenness', 'pr']
    
    X_train = df_train[features]
    y_train = df_train['is_management']
    X_test = df_test[features]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = MLPClassifier(hidden_layer_sizes=[10,5], alpha=5, random_state=0, solver='lbfgs')
    clf.fit(X_train_scaled, y_train)
    test_proba = clf.predict_proba(X_test_scaled)[:,1]
    
    return pd.Series(test_proba, X_test.index)

salary_predictions()


# ### Part B - New Connections Prediction
# 
# For the last part of this project, you will predict future connections between employees of the network. The future connections information has been loaded into the variable `future_connections`. The index is a tuple indicating a pair of nodes that currently do not have a connection, and the `Future Connection` column indicates if an edge between those two nodes will exist in the future, where a value of 1.0 indicates a future connection.

# In[ ]:


future_connections = pd.read_csv('Future_Connections.csv', index_col=0, converters={0: eval})
future_connections.head(10)


# Using network `G` and `future_connections`, identify the edges in `future_connections` with missing values and predict whether or not these edges will have a future connection.
# 
# To accomplish this, you will need to create a matrix of features for the edges found in `future_connections` using networkx, train a sklearn classifier on those edges in `future_connections` that have `Future Connection` data, and predict a probability of the edge being a future connection for those edges in `future_connections` where `Future Connection` is missing.
# 
# 
# 
# Your predictions will need to be given as the probability of the corresponding edge being a future connection.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC).
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUC of 0.88 or higher will receive full points, and with an AUC of 0.82 or higher will pass (get 80% of the full points).
# 
# Using your trained classifier, return a series of length 122112 with the data being the probability of the edge being a future connection, and the index being the edge as represented by a tuple of nodes.
# 
#     Example:
#     
#         (107, 348)    0.35
#         (542, 751)    0.40
#         (20, 426)     0.55
#         (50, 989)     0.35
#                   ...
#         (939, 940)    0.15
#         (555, 905)    0.35
#         (75, 101)     0.65
#         Length: 122112, dtype: float64

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

def new_connections_predictions():
    
    for node in G.nodes():
        G.node[node]['community'] = G.node[node]['Department']
    
    preferential_attachment = list(nx.preferential_attachment(G))
    df = pd.DataFrame(index = [(x[0], x[1]) for x in preferential_attachment])
    df['preferential_attachment'] = [x[2] for x in preferential_attachment]
    cn_sh = list(nx.cn_soundarajan_hopcroft(G))
    df_cn_sh = pd.DataFrame(index = [(x[0], x[1]) for x in cn_sh])
    df_cn_sh['cn_soundarajan_hopcroft'] = [x[2] for x in cn_sh]
    df = df.join(df_cn_sh, how='outer')
    df['cn_soundarajan_hopcroft'] = df['cn_soundarajan_hopcroft'].fillna(value=0)
    df['resource_allocation_index'] = [x[2] for x in list(nx.resource_allocation_index(G))]
    df['jaccard_coefficient'] = [x[2] for x in list(nx.jaccard_coefficient(G))]
    df = future_connections.join(df, how='outer')
    
    df_train = df[~pd.isnull(df['Future Connection'])]
    df_test = df[pd.isnull(df['Future Connection'])]
    features = ['cn_soundarajan_hopcroft', 'preferential_attachment', 'resource_allocation_index', 'jaccard_coefficient']
    
    X_train = df_train[features]
    y_train = df_train['Future Connection']
    X_test = df_test[features]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = MLPClassifier(hidden_layer_sizes=[10,5], alpha=5, random_state=0, solver='lbfgs')
    clf.fit(X_train_scaled, y_train)
    test_proba = clf.predict_proba(X_test_scaled)[:,1]
    
    predictions = pd.Series(test_proba, X_test.index)
    target = future_connections[pd.isnull(future_connections['Future Connection'])]
    target['prob'] = [predictions[x] for x in target.index]
    
    return  target['prob']

new_connections_predictions()

