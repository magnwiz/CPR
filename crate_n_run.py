# from mesa.batchrunner import BatchRunner

import matplotlib.pyplot as plt
import numpy as np
import model_networks as mn
import random
import networkx as nx
import pickle


def create_n_run(num_ticks, 
                #  n_actions, 
                #  max_action,
                #  weight_triadic,
                #  weight_attribute,
                #  weight_geo, 
                 n_fragments, 
                #  rewiring_rate, 
                #  num_voting_agents,
                 emergence_time,
                 ba_m,
                 ws_k,
                 ws_p,
                #  k_0,
                #  r,
                #  energy_consumption,
                 threshold_institutional_change,
                #  innovation_rate,
                 max_social_influence,
                 seed=None
                 ):

    # defining weights for rewiring probs
    # other_weights = (1 - weight_geo) / 2
    weights = {"triadic": 0.33, "attribute": 0.33, "geo": 0.33}

   
    seed = random.randint(1, 100000)
    # seed = 31874
    drawing = {'figures':'no', 'allign': 'not'} 
    # print('list_actions:', list_actions, 'step_action = ', max_action, '/', n_actions)
    model = mn.CPRModel(N=100, 
                        num_ticks=num_ticks, 
                        # max_action=max_action,
                        # n_actions=n_actions, 
                        weights=weights,
                        n_fragments=n_fragments,
                        # rewiring_rate=rewiring_rate, 
                        drawing=drawing, 
                        # num_voting_agents=num_voting_agents,
                        emergence_time=emergence_time,
                        ba_m=ba_m,
                        ws_k=ws_k,
                        ws_p=ws_p,
                        # k_0=k_0,
                        # r=r,
                        # energy_consumption=energy_consumption,
                        threshold_institutional_change=threshold_institutional_change,
                        # innovation_rate=innovation_rate,
                        max_social_influence=max_social_influence,
                        seed=seed
                        )
    
    model.step()

    degree_centrality = {'kins_net': nx.degree_centrality(model.kins_net), 
                         'geo_net': nx.degree_centrality(model.geo_net), 
                         'friends_net': nx.degree_centrality(model.friends_net),
                         'friends_net (out)': nx.out_degree_centrality(model.friends_net), 
                         'friends_net (in)': nx.in_degree_centrality(model.friends_net), 
                         'combined_net': nx.degree_centrality(model.combined_net)
                        }
    friends_evoluton = model.friends_evolution
    comb_evoultion = model.combined_evolution
    friends_density = nx.density(model.friends_net)

    in_degrees = list(degree_centrality['friends_net (in)'].values()) # it was (out), so the current results are doubtful
    mean_in_degree = np.mean(in_degrees)
    variance_in_degree = np.var(in_degrees)
    out_degrees = list(degree_centrality['friends_net (out)'].values()) # it was (out), so the current results are doubtful
    mean_out_degree = np.mean(out_degrees)
    variance_out_degree = np.var(out_degrees)

    comb_degrees = list(degree_centrality['combined_net'].values())
    mean_comb_degree = np.mean(comb_degrees)
    variance_comb_degree = np.var(comb_degrees)

    if len(model.list_institutions) > 1:
        inst_age_list = []
        for (t1, t2) in model.dict_institutions.keys():
            if model.dict_institutions[(t1, t2)] == -1:
                emergence_time = t2 - t1
            else:
                inst_age_list.append(t2 - t1)

        mean_inst_age = np.mean(inst_age_list)
        variance_inst_age = np.var(inst_age_list)
    else:
        mean_inst_age = 0
        variance_inst_age = 0
        emergence_time = 0

    num_institutions = len(model.list_institutions)
    # list_institutions = model.list_institutions
    # list_ticks_inst = model.list_tick_emerge_institution
    # dict_inst = model.dict_institutions
    # dict_fine_monitoring = model.dict_fine_monitoring
    # distribution__probabilities = model.probabilities
    # clustering_comb = model.clustering_comb
    # clustering_friends = model.clustering_friends
    # agents_voting = model.ts_voting_agents
    agents_consumption = 0
    for agent in model.schedule.agents:
        agents_consumption += agent.consumed_resource
    agents_avg_consumption = agents_consumption / 100
    # p_total_mean = np.mean(model.probabilities['total'])
    # p_total_variance = np.var(model.probabilities['total'])
    # p_geo_mean = np.mean(model.probabilities['geographical'])
    # p_geo_variance = np.var(model.probabilities['geographical'])
    resource = model.resource
    time_step = model.stepcounter

    # creating a dictionary for all the generated input parameters

    
    max_action = model.max_action
    n_actions = model.n_actions
    w_triadic = model.weights['triadic']
    w_attribute = model.weights['attribute']
    w_geo = model.weights['geo']
    n_fragments = n_fragments
    max_fragment = max(model.fragments)
    rewiring_rate = model.rewiring_rate
    num_voting_agents = model.num_voting_agents
    emergence_rate = model.institutional_emergence_time
    energy_consumption = model.energy_consumption
    initial_resource = model.resource_first
    r = model.r
    threshold_institutional_change = model.threshold_institutional_change
    innovation_rate = model.innovation_rate
    max_social_influence = model.max_social_influence
    max_fine = model.max_fine
    max_monitoring = model.max_monitoring
    monitoring_cost_weight = model.monitoring_cost_weight
    seed = model._seed
    

    return num_institutions, emergence_time, mean_inst_age, variance_inst_age, \
        agents_avg_consumption, mean_in_degree, variance_in_degree, \
        mean_out_degree, variance_out_degree,  \
        mean_comb_degree, variance_comb_degree, friends_density, \
        resource, time_step, \
        max_action, n_actions, w_triadic, w_attribute, w_geo, \
        n_fragments, max_fragment, rewiring_rate, num_voting_agents,\
        emergence_rate, energy_consumption, initial_resource, r,\
        threshold_institutional_change, innovation_rate, max_social_influence,\
        max_fine, max_monitoring, monitoring_cost_weight, seed 
        # p_total_mean, p_total_variance,\
        # p_geo_mean, p_geo_variance, seed
        #    list_institutions, list_ticks_inst,  \
        #    dict_fine_monitoring,  \
        #    comb_evolution, friends_evolution,




# create_n_run(num_ticks=2000, 
#             n_actions=8, 
#             max_action=20,
#             # weight_triadic=0.5,
#             # weight_attribute=0.5,
#             # weight_geo=0.33, 
#             n_fragments=1, 
#             rewiring_rate=10, 
#             num_voting_agents=50,
#             emergence_time=200,
#             ba_m=3,
#             ws_k=2,
#             ws_p=0.4,
#             # k_0=20000,
#             # r=0.2,
#             energy_consumption=5,
#             threshold_institutional_change=0.3,
#             innovation_rate=0.5,
#             max_social_influence=0.5,
#             seed=None
    
# )




# Defining the parameters of the model 


# problem = {
#     'names': ['num_ticks', 
#               'n_actions',
#               'max_action',
#               'weight_triadic',
#               'weight_attribute',
#               'weight_geo', 
#               'n_fragments', 
#               'rewiring_rate', 
#               'num_voting_agents',
#               'emergence_time',
#               'ba_m', 
#               'ws_k', 
#               'ws_p', 
#               'k_0',
#               'r', 
#               'energy_consumption', 
#               'threshold_institutional_change',
#               ],
#     'num_vars': 17,
#     'bounds': [[1000, 5000], # num_ticks
#                [5, 12], # n_actions
#                [1, 50], # max_action
#                [0, 1], # weight triadic
#                [0, 1], # weight attribute
#                [0, 1], # weight geo
#                [1, 4], # n_fragments
#                [1, 500], # rewiring rate
#                [1, 100], # num voting agents
#                [50, 1000], # emergence time
#                [1, 5], # BA: m
#                [1, 5], # WS: k
#                [0, 1], # WS: p
#                [10000, 50000], # K0: 
#                [0, 1], # r
#                [1, 10], # energy consupmption
#                [0.2, 0.8] # threshold institutional change
#                ],
#     'dists': ['unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif',
#               'unif'
#               ]
# }

# n_saltelli=1024
# X = saltelli.sample(problem, n_saltelli)
# Y = create_n_run(X[:,0],X[:,1],X[:,2],X[:,3],
#                  X[:,4],X[:,5],X[:,6],X[:,7],
#                  X[:,8],X[:,9],X[:,10],X[:,11],
#                  X[:,12],X[:,13],X[:,14],X[:,15],
#                  X[:,16])
# print(np.mean(Y))
# Si = sobol.analyze(problem, Y, print_to_console=True)








# list_actions = [0, 1, 2, 4, 8, 12, 16, 20]

# fragments_list = [20, 30, 50]



# rewiring_rate = 10

# num_ticks = 1999

# drawing parameters: 
# save or show the figures with the graphs
# allign nodes' positions according to one of the graphs

 
# num_voting_agents = 10




# Plot each histogram in a separate subplot
# subplot_names = list(model.probabilities.keys())  # Convert dict_keys to a list
# num_subplots = len(subplot_names)

# # Create subplots with shared y-axis
# fig, axes = plt.subplots(1, num_subplots, figsize=(15, 5), sharey=False)

# for i, name in enumerate(subplot_names):
#     ax = axes[i]
#     ax.hist(model.probabilities[name], bins=50, edgecolor='black')
#     ax.set_xlabel("Values")
#     ax.set_ylabel("Frequency")
#     ax.set_title(f"Distribution of {name.capitalize()} Probabilities")

# # Adjust layout and display
# plt.tight_layout()
# plt.show()

