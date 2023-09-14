from mesa import Agent, Model
from mesa.model import Model
from mesa.time import RandomActivation
import os
import matplotlib.pyplot as plt
import networkx as nx
import itertools




class MyAgent(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)

        self.energy = self.random.randint(1000, 2000)
        self.initial_energy = self.energy

        # calculating the fragment an agent belongs to:
        previous = 0
        self.fragment = {}
        for fragment_index, fragment in enumerate(self.model.fragments):
            if previous <= self.unique_id < (fragment + previous):
                self.fragment = {fragment_index: fragment} 
                break
            previous += fragment

        self.consumed_resource = 0
        self.innovation_rate_agent = self.random.uniform(0, 1)  # how much one agent is innovative
        # the impact of best strategy on changing strategy, if more then we will have same strategy as institution each time
        self.confidence = self.random.choice([True, False])
        self.cheating_propension = self.random.uniform(0, 0.2) # from Ghorbani paper
        self.social_influence = self.random.uniform(0, self.model.max_social_influence)
        self.cheating_profitable = False  # whether agent obeys or not (in mind)
        self.cheated = False  # real cheat in act
        """
        self.cheated_last = False   #it won't change based on monitoring
        """
        self.action = self.random.choice(self.model.list_actions)
        self.best_action = self.action
        self.best_energy = 0
        self.memory = {}
        self.own_idea_fine = self.random.randint(0, self.model.max_fine)
        self.own_idea_monitoring = self.random.randint(0, self.model.max_monitoring)
        self.voting = False

        self.friends_in = set()
        self.friends_out = set()
        self.kins = set(nx.neighbors(self.model.kins_net, self.unique_id))
        self.geo_neighbors = set(nx.neighbors(self.model.geo_net, self.unique_id))
        self.in_degree = len(self.get_all_neighbors('in'))
        self.out_degree = len(self.get_all_neighbors('out'))
        # self.all_neighbors_out = set(self.kins | self.geo_neighbors)
        # self.all_neighbors_in = set(self.kins | self.geo_neighbors)

        # print(
        #     "Setup:",
        #     "agentID:",
        #     self.unique_id,
        #     "fragment:",
        #     self.fragment,
        #     "action:",
        #     self.action,
        #     "fine:",
        #     self.own_idea_fine,
        #     "monitoring:",
        #     self.own_idea_monitoring,
        #     "energy:",
        #     self.energy
        # )


    def step(self):
        # update friends and neighbors
        self.friends_in = set([source for source, target in self.model.friends_net.in_edges(self.unique_id)])
        self.friends_out = set([target for source, target in self.model.friends_net.out_edges(self.unique_id)])
        # self.all_neighbors_in = self.kins | self.friends_in | self.geo_neighbors
        # self.all_neighbors_out = self.kins | self.friends_out | self.geo_neighbors
        # memorizing strategy if it has led to the best outcome
        if self.energy > self.best_energy:
            self.best_action = self.action
        # consuming energy
        self.energy = self.energy - self.model.energy_consumption

        # Check for conditions to change a strategy
        if self.energy < 1000:
            if self.innovation_rate_agent >= self.model.innovation_rate:
                if self.confidence: # change strategy based on the best strategy
                    self.action = self.best_action
                    self.own_idea_fine = self.random.randint(0, self.model.max_fine)
                    self.own_idea_monitoring = self.random.randint(0, self.model.max_monitoring)                    
                else:
                    # Choose new random strategy
                    options = [option for option in self.model.list_actions if option != self.action]
                    self.action = self.random.choice(options)
                    self.own_idea_fine = self.random.randint(0, self.model.max_fine)
                    self.own_idea_monitoring = self.random.randint(0, self.model.max_monitoring)
            else: 
                # chooses a strategy of the most successful neighbor
                highest_energy = self.energy
                successful_neighbor = self
                relation = "self"
                # when agent is copying a strategy from his neighbors
                # and there is more than one agent with the highest energy 
                # it will first give preference to his kins, then to his friends and 
                # then to his geographical neighbors
                for kin in self.kins:
                    if self.model.schedule.agents[kin].energy > highest_energy:
                        highest_energy = self.model.schedule.agents[kin].energy
                        successful_neighbor = self.model.schedule.agents[kin]
                        relation = "kin"
                for friend in self.friends_out: 
                    if self.model.schedule.agents[friend].energy > highest_energy:
                        highest_energy = self.model.schedule.agents[friend].energy
                        successful_neighbor = self.model.schedule.agents[friend]
                        relation = "friend"
                for neighbor in self.geo_neighbors:
                    if self.model.schedule.agents[neighbor].energy > highest_energy:
                        highest_energy = self.model.schedule.agents[neighbor].energy
                        successful_neighbor = self.model.schedule.agents[neighbor]
                        relation = "geographical"
                self.action = successful_neighbor.action
                self.own_idea_fine = successful_neighbor.own_idea_fine
                self.own_idea_monitoring = successful_neighbor.own_idea_monitoring
                # print("Strategy copied from ", successful_neighbor.unique_id, "to ", \
                #       self.unique_id, "relation type: ", relation, " Time: ", self.model.stepcounter)
                    

        # Check if there is an institution
        if (len(self.model.list_institutions) >= 2):
        # if own action brings more, agent might cheat
            if self.action > self.model.list_institutions[-1]:
                self.cheating_profitable = True
                # social parameter represents probability that surroundings influence the decision to cheat
                social_parameter = (self.cheating_propension) * ((1 - self.social_influence) + \
                     (self.count_cheated_neighbors() / len(self.get_all_neighbors('out'))) * self.social_influence)
                if self.cheating_profitable and \
                    self.random.random() < social_parameter:
                    # if all conditions are satisfied, agent cheats by following own strategy
                        self.energy += self.action
                        self.consumed_resource += self.action
                        self.model.resource -= self.action
                        self.cheated = True
            else: # otherwise agent follows institution
                self.energy += self.model.list_institutions[-1]
                self.consumed_resource += self.model.list_institutions[-1]
                self.model.resource -= self.model.list_institutions[-1]  
        else: # agent does own strategy if there is no institution
            self.energy += self.action
            self.consumed_resource += self.action
            self.model.resource -= self.action


    # Returns adjacent nodes from all networks 
    def get_all_neighbors(self, direction):
        if direction == "out":
            all_neighbors = self.kins | self.geo_neighbors | self.friends_out
        elif direction == "in":
            all_neighbors = self.kins | self.geo_neighbors | self.friends_in

        return all_neighbors
        
    def count_cheated_neighbors(self):
        cheated_neihbors = self.kins | self.geo_neighbors | self.friends_out
        num_cheated_neighbors = len(cheated_neihbors)

        return num_cheated_neighbors

    def init_friendship_probability(self, another_agent):
        # triadic closure
        self_neighbors = self.kins | self.geo_neighbors
        another_neighbors = another_agent.kins | another_agent.geo_neighbors
        common_neighbors = self_neighbors & another_neighbors
        if common_neighbors:
            total_neighbors = self_neighbors | another_neighbors
            p1 = len(common_neighbors) / len(total_neighbors)
        else:
            p1 = 0

        # attribute-driven
        if self.action == another_agent.action:
            p2 = 1
        else:
            p2 = 0

        # geographical proximity
        if another_agent in self.geo_neighbors:
            p3 = 1
        else:
            p3 = 0

        # norm = sum(self.model.weights.values())
        p = (self.model.weights["triadic"] * p1 + self.model.weights["attribute"] * p2 + self.model.weights["geo"] * p3)

        return p
    
    def rewiring_probability(self, another_agent, backup_net):
        # triadic closure
        friends_out = set(backup_net.neighbors(self.unique_id))
        friends_in = set(backup_net.successors(self.unique_id))
        self_neighbors = friends_in | friends_out | self.kins
        another_friends_out = set(backup_net.neighbors(another_agent.unique_id))
        another_friends_in = set(backup_net.successors(another_agent.unique_id))
        another_neighbors = another_friends_in | another_friends_out | another_agent.kins
        common_neighbors = self_neighbors & another_neighbors
        if common_neighbors:
            total_neighbors = self_neighbors | another_neighbors
            p1 = len(common_neighbors) / len(total_neighbors)
        else:
            p1 = 0
        self.model.probabilities['triadic'].append(p1)

        # attribute-driven or gear homophily
        if self.action == another_agent.action:
            p2 = 1
        else:
            p2 = 0
        self.model.probabilities['gear'].append(p2)

        # geographical proximity
        if another_agent.unique_id in self.geo_neighbors:
            p3 = 1
        else:
            p3 = 0
        self.model.probabilities['geographical'].append(p3)

        # normalizing weights:
        # norm_coeff = sum(self.model.weights.values())

        p = (self.model.weights["triadic"] * p1 + self.model.weights["attribute"] * p2 + self.model.weights["geo"] * p3) #/ norm_coeff

        self.model.probabilities['total'].append(p)
        return p


class CPRModel(Model):
    def __init__(self, 
                 N,
                 num_ticks,
                #  max_action,
                #  n_actions, 
                 weights, 
                 n_fragments, 
                #  rewiring_rate,
                 drawing, 
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
                 seed):
        
        self.num_agents = N

         # creating a list of actions
        self.max_action = self.random.randint(10, 25) #PRIM was done on (10, 20) but I want more unstable institutions
        self.n_actions = self.random.randint(2, 10)
        step_action = self.max_action / self.n_actions
        list_actions = []
        action = 0
        for _ in range(self.n_actions - 1):
            list_actions.append(action)
            action = int(action + step_action)
        list_actions.append(self.max_action)
        self.list_actions = list_actions

        self.num_voting_agents = self.random.randint(1, 100)
        self.innovation_rate = self.random.uniform(0, 1)
        self.resource = self.random.randint(25000, 35000) #self.random.randint(k_0, k_0 + 5000)
        self.resource_first = self.resource
        self.r = self.random.uniform(0.25, 0.4)
        self.institutional_emergence_time = emergence_time #self.random.randint(100, 400) #200  
        self.energy_consumption = self.random.randint(5, 20)   # based on sensitivity analysis
        self.threshold_institutional_change = threshold_institutional_change # 0.4
        self.stepcounter = 0  # calculate the tick
        self.schedule = RandomActivation(self)
        self.num_ticks = num_ticks
        self.list_institutions = [-1]
        self.list_tick_emerge_institution = [-1]
        self.max_social_influence = max_social_influence

        # attributes associated with fines and monitoring
        self.dict_fine_monitoring = {"fine": [-1], "monitoring": [-1]}
        self.max_fine = self.random.randint(0, 5)
        self.max_monitoring = self.random.randint(0, 10)
        self.monitoring_cost_weight = self.random.randint(1, 5)
        self.list_tick_monitoring = [-1]

        # attributes associated with rewiring
        self.rewiring_rate = self.random.randint(1, 100)
        self.weights = weights
        # self.weights = {}
        # self.weights["triadic"] = self.random.randrange(0, 1)
        # self.weights["geo"] = self.random.randrange(0, 1)
        # self.weights["attribute"] = self.random.randrange(0, 1)

        # self.fragments = fragments_list
        self.fragments = []
        upper_limit = self.num_agents
        if n_fragments > 1:
            for _ in range(n_fragments - 1):
                # delta that will not allow size of the fragment to be smaller than ba_m
                delta = (ba_m + 1) * (n_fragments - _)
                fragment = self.random.randint(ba_m + 1, upper_limit - delta)
                self.fragments.append(fragment)
                upper_limit -= fragment
            fragment = upper_limit
            self.fragments.append(fragment)
        else:
            self.fragments = [self.num_agents]

        self.kins_net = self.fragmented_network(self.fragments, ba_m)
        self.geo_net = nx.watts_strogatz_graph(N, ws_k, ws_p, seed=self._seed)

        # creating agents
        for i in range(self.num_agents):
            a = MyAgent(i, self)
            self.schedule.add(a)

        # creating friends' network
        self.friends_net = nx.DiGraph()
        self.friends_net.add_nodes_from(range(self.num_agents))
        for node in self.friends_net:
            agent1 = self.schedule.agents[node]
            for another_node in self.friends_net:
                if node != another_node:
                    agent2 = self.schedule.agents[another_node]
                    prob = agent1.init_friendship_probability(agent2)
                    if self.random.random() < prob:
                        self.friends_net.add_edge(node, another_node, weight=prob)
                        agent1.friends_out.add(agent2.unique_id)
                        agent2.friends_in.add(agent1.unique_id)
        
        # Create a combined network
        self.combined_net = nx.DiGraph()
        # Set to keep track of added edges
        added_edges = set()
        # Add edges from the undirected networks
        for edge in self.kins_net.edges():
            added_edges.add((edge[0], edge[1]))
            added_edges.add((edge[1], edge[0]))  # Add the reverse edge

        for edge in self.geo_net.edges():
            added_edges.add((edge[0], edge[1]))
            added_edges.add((edge[1], edge[0]))  # Add the reverse edge

        for edge in self.friends_net.edges():
            added_edges.add((edge[0], edge[1]))

        # Add edges to the combined network
        for edge in added_edges:
            self.combined_net.add_edge(edge[0], edge[1])


        # data collection
        
        # data collection of probabilities associated with rewiring:
        self.probabilities = {'total': [], 'triadic': [], 'gear': [], 'geographical': []}
        # data colletcion related to networks:
        self.links_emerged = {0: set(self.friends_net.edges())}
        self.links_broke = {0: {}}
        self.friends_evolution = {0: self.friends_net}
        self.combined_evolution = {0: self.combined_net}
        self.clustering_comb = nx.clustering(self.combined_net)
        self.clustering_friends = nx.clustering(self.friends_net)
        self.dict_institutions = {}
        self.ts_voting_agents = {}

        # parameters for drawing
        self.allign = drawing['allign']
        self.figures = drawing['figures']
        self.cmap = plt.cm.get_cmap('tab10', len(self.fragments))
        self.nodes_color = [list(agent.fragment.keys())[0] for agent in self.schedule.agents]
        if self.allign == 'geo':
            self.nodes_positions = nx.spring_layout(self.geo_net, seed=self._seed)
        elif self.allign == 'friends':
            self.nodes_positions = nx.spring_layout(self.friends_net, seed=self._seed)
        elif self.allign == 'kins':
            self.nodes_positions = nx.spring_layout(self.kins_net, seed=self._seed)
        elif self.allign == 'complete':
            self.nodes_positions = nx.spring_layout(nx.complete_graph(100), seed=self._seed)
        elif self.allign == 'not':
            self.nodes_positions = {}
        else:
            self.nodes_positions = nx.spring_layout(self.geo_net, seed=self._seed)  
            # print('wrong argument, alligned as geographical neighbors')

        # print('List of actions: ', self.list_actions)
        if self.figures != 'no':
            self.draw_graphs("Initial_graphs")
        # self.print_input_parameters(ba_m, ws_k, ws_p)
        
    def step(self):
        
        while (self.resource > 0 and self.stepcounter < self.num_ticks):  
            self.schedule.step()
            # print("Resource = ", self.resource, " Time: ", self.stepcounter, "/", self.num_ticks)
            # defining institution
            if (((self.stepcounter + 1) % self.institutional_emergence_time) == 0) \
                    and (self.list_tick_emerge_institution[-1] != self.stepcounter):  
                self.define_institution()
                
            if len(self.dict_fine_monitoring['fine']) > 1:
                self.monitor_agents()
            
            if (self.stepcounter + 1) % self.rewiring_rate == 0:
                self.update_friends_net()
                self.update_combined_net()
            self.resource = self.resource_grow()
            
            self.stepcounter += 1
            if self.stepcounter >= self.num_ticks or self.resource < 0:
                self.dict_institutions[(self.list_tick_emerge_institution[-1], self.stepcounter)] = self.list_institutions[-1]

  
    def print_input_parameters(self, ba_m, ws_k, ws_p):
        print("Model was created with these Input Parameters:")
        print(f"N: {self.num_agents}")
        print(f"num_ticks: {self.num_ticks}")
        print(f"list_actions: {self.list_actions}")
        print(f"weights: {self.weights}")
        print(f"fragments: {self.fragments}")
        print(f"rewiring_rate: {self.rewiring_rate}")
        print(f"num_voting_agents: {self.num_voting_agents}")
        print(f"emergence_time: {self.institutional_emergence_time}")
        print(f"ba_m: {ba_m}")
        print(f"ws_k: {ws_k}")
        print(f"ws_p: {ws_p}")
        print(f"energy_consumption: {self.energy_consumption}")
        print(f"threshold_institutional_change: {self.threshold_institutional_change}")
        print(f"innovation_rate: {self.innovation_rate}")
        print(f"max_social_influence: {self.max_social_influence}")
        print(f"seed: {self._seed}")          

    def draw_graphs(self, caption, distribution=False):
            
        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

        if self.nodes_positions:
            # Draw the kinship graph with default node color
            nx.draw_networkx(
                self.kins_net, ax=ax1, node_color=self.nodes_color , node_size=100, pos=self.nodes_positions,
                cmap=self.cmap, font_size=10
                )

            # Draw the geographical neighbors graph with green nodes
            nx.draw_networkx(
                self.geo_net, ax=ax2, node_color=self.nodes_color , node_size=100, pos=self.nodes_positions,
                cmap=self.cmap, font_size=10
            )

            # Draw the friendship graph with red nodes
            nx.draw_networkx(
                self.friends_net, ax=ax3, node_color=self.nodes_color , node_size=100, pos=self.nodes_positions,
                cmap=self.cmap, font_size=10
            )
        else:
            nx.draw_networkx(
                self.kins_net, ax=ax1, node_color=self.nodes_color , node_size=100,
                pos=nx.spring_layout(self.kins_net, seed=self._seed),
                cmap=self.cmap, font_size=10
                )

            # Draw the geographical neighbors graph with green nodes
            nx.draw_networkx(
                self.geo_net, ax=ax2, node_color=self.nodes_color , node_size=100,
                pos=nx.spring_layout(self.geo_net, seed=self._seed),
                cmap=self.cmap, font_size=10
            )

            # Draw the friendship graph with red nodes
            nx.draw_networkx(
                self.friends_net, ax=ax3, node_color=self.nodes_color , node_size=100,
                pos=nx.spring_layout(self.friends_net, seed=self._seed),
                cmap=self.cmap, font_size=10
            )

        # Set the titles for the subplots
        ax1.set_title("Kinship network")
        ax2.set_title("Geographical neighbors")
        ax3.set_title("Friendship network")

        # Common title
        plt.suptitle(caption + ' seed:' + str(self._seed), fontsize=16)

        if self.figures == 'show':
            # Show the figure
            plt.show()
        elif self.figures == 'save':
            folder_name = 'simulation_figures/figures_' + str(self._seed)
            # Create the folder if it doesn't exist
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            plt.savefig(os.path.join(folder_name, str(caption) + '.png'))

        if distribution:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            # Plot degree distribution for the first graph
            degree_sequence_1 = [d for n, d in self.kins_net.degree()]
            axs[0].hist(degree_sequence_1, bins=10)
            axs[0].set_title("Kinship Network Degree Distribution")
            axs[0].set_xlabel("Degree")
            axs[0].set_ylabel("Count")

            # Plot degree distribution for the second graph
            degree_sequence_2 = [d for n, d in self.geo_net.degree()]
            axs[1].hist(degree_sequence_2, bins=10)
            axs[1].set_title("Geographical Neighbors Degree Distribution")
            axs[1].set_xlabel("Degree")
            axs[1].set_ylabel("Count")

            # Plot in-degree and out-degree distribution for the third graph
            indegree_sequence = [d for n, d in self.friends_net.in_degree()]
            outdegree_sequence = [d for n, d in self.friends_net.out_degree()]
            axs[2].hist(indegree_sequence, bins=10, alpha=0.5, label="In-degree")
            axs[2].hist(outdegree_sequence, bins=10, alpha=0.5, label="Out-degree")
            axs[2].set_title("Frienship Network Degree Distribution")
            axs[2].set_xlabel("Degree")
            axs[2].set_ylabel("Count")
            axs[2].legend()

            # Adjust spacing between subplots
            plt.tight_layout()

            if self.figures == 'show':
                # Show the figure
                plt.show()
            elif self.figures == 'save':
                plt.savefig('/figures/' + caption + '_distribution.png')  

    def monitor_agents(self):
        monitoring_cost = self.monitoring_cost_weight * self.dict_fine_monitoring['monitoring'][-1]
        number_cheaters = 0
        total_fine = 0
        agents_fined = []
        num_of_agents = self.num_agents
        monitor_number = int((self.dict_fine_monitoring['monitoring'][-1] * num_of_agents)/100)
        for i in range(0, monitor_number):
            cand_agent = self.random.choice(self.schedule.agents)
            if cand_agent.cheated:
                number_cheaters += 1
                cand_agent.energy = cand_agent.energy - self.dict_fine_monitoring['fine'][-1]
                cand_agent.cheated_last = cand_agent.cheated
                cand_agent.cheated = False
                total_fine += self.dict_fine_monitoring['fine'][-1]
                agents_fined.append(cand_agent)
            # print("chosen", i)
            #chosen_agents = +1
        # print("Finishhhhhhhh")
        energy_balance = monitoring_cost - total_fine
        # if number_cheaters/num_of_agents > self.model.max_cheating_propension:
        #     self.model.max_cheating_propension = number_cheaters/num_of_agents
        for agent in self.schedule.agents:
            agent.energy -= (energy_balance/num_of_agents)
        self.list_tick_monitoring.append(self.stepcounter)

    def define_institution(self):
        # print("institutional emergence time reached: ", self.stepcounter + 1, '/', self.num_ticks, \
                    #   "current institution: ", self.list_institutions[-1])
        # self.draw_graphs(str(self.stepcounter + 1))
        count_energy_negative = 0
        for agent in self.schedule.agents:
            if agent.energy < 1000: #1000
                count_energy_negative += 1  # number of agents with negative energy (current)
        
        # print('Time:', self.stepcounter, "Number of agents with negative energy: ", count_energy_negative / self.num_agents\
        #       , '/', self.threshold_institutional_change)

        if (count_energy_negative / self.num_agents) > self.threshold_institutional_change:
            # records of voting result
            act_voting_dict = {action: 0 for action in self.list_actions} 
            fine_voting_dict = {fine: 0 for fine in range(self.max_fine + 1)}
            monitoring_voting_dict = {monitoring: 0 for monitoring in range(self.max_monitoring + 1)}
            voting_agents = self.get_voting_agents()
            self.ts_voting_agents[self.stepcounter] = voting_agents
            for agent in voting_agents:
                # voting on their action
                act_voting_dict[agent.action] += 1
                # voting on their fine
                fine_voting_dict[agent.own_idea_fine] += 1
                # voting on their monitoring
                monitoring_voting_dict[agent.own_idea_monitoring] += 1
                # cheating marker resets for a new institution
                agent.cheated = False

            # print ("Action voting results: ", act_voting_dict)
            new_institution = max(act_voting_dict, key=act_voting_dict.get)
            new_fine = max(fine_voting_dict, key=fine_voting_dict.get)
            new_monitoring = max(monitoring_voting_dict, key=monitoring_voting_dict.get)
            print("New institution: ", new_institution, " | New fine: ", new_fine, " | New monitoring: ", new_monitoring, "%")
            
            # Indicator is reset for a new institution
            for agent in self.schedule.agents:
                agent.energy = agent.initial_energy
                agent.cheated = False
            self.dict_institutions[(self.list_tick_emerge_institution[-1], self.stepcounter)] = self.list_institutions[-1]
            self.list_institutions.append(new_institution)  # list of institutions
            self.list_tick_emerge_institution.append(self.stepcounter)  # #the ticks when institutions define
            self.dict_fine_monitoring["fine"].append(new_fine)
            self.dict_fine_monitoring["monitoring"].append(new_fine)
        # else:
            # print("Institutional threshold not met: ", count_energy_negative, "/", \
                    # int(self.threshold_institutional_change * self.num_agents))


    def resource_grow(self):
        K = self.resource_first  # Integer from 15000 to 30000, endpoints included, random.randint(25000, 30000)
        r = self.r # self.random.uniform(self.r, self.r + 0.1)  #0.1# Random float 0.15, 0.30, random.uniform(0.20, 0.30)
        newresourcevalue = self.resource + (r * self.resource * (1 - (self.resource / K)))
        return newresourcevalue

    def fragmented_network(self, fragments_list, ba_m):

        if (sum(fragments_list) != self.num_agents):
            raise ValueError("Fragments sum and the total number of agents are different")
        
        kins_net = nx.Graph()
        n = 0 #counter

        for fragment in fragments_list:
            fragment_graph = nx.barabasi_albert_graph(fragment, ba_m, seed=self._seed)
            # Adjust node numbering to ensure consistency across fragments
            adjusted_edges = [
                (n + node_one, n + node_two) for node_one, node_two in fragment_graph.edges()
            ]
            adjusted_graph = nx.Graph(adjusted_edges)
            kins_net = nx.compose(kins_net, adjusted_graph)
            n += fragment
        
        return kins_net

    def get_voting_agents(self):
        agent_degrees = {}
        for agent in self.schedule.agents:
            agent.in_degree = len(agent.get_all_neighbors("in"))
            agent_degrees[agent] = agent.in_degree
        sorted_agent_degrees = dict(sorted(agent_degrees.items(), key=lambda item: item[1], reverse=True))
        agents_voting = list(sorted_agent_degrees.keys())[:self.num_voting_agents]
        for agent in agents_voting:
            agent.voting = True
        return agents_voting
    
    def update_combined_net(self):
        # Create a new directed graph
        self.combined_net = nx.DiGraph()
        # Set to keep track of added edges
        added_edges = set()
        # Add edges from the undirected networks
        for edge in self.kins_net.edges():
            added_edges.add((edge[0], edge[1]))
            added_edges.add((edge[1], edge[0]))  # Add the reverse edge

        for edge in self.geo_net.edges():
            added_edges.add((edge[0], edge[1]))
            added_edges.add((edge[1], edge[0]))  # Add the reverse edge

        for edge in self.friends_net.edges():
            added_edges.add((edge[0], edge[1]))

        # Add edges to the combined network
        for edge in added_edges:
            self.combined_net.add_edge(edge[0], edge[1])

        self.combined_evolution[self.stepcounter] = self.combined_net
        self.clustering_comb = nx.clustering(self.combined_net)

    def update_friends_net(self):
        # adjusting friendship ties
        backup_net = self.friends_net
        # to keep the density of the friends network consistent there is a count of
        # links that are broken to create the same amount of new links after
        num_rewired = 0
        # first check if the existing links are going to break
        num_edges1 = len(backup_net.edges()) 
        broken_links_temp = set()
        for node in backup_net:
            agent1 = self.schedule.agents[node]
            for another_node in backup_net.nodes():
                agent2 = self.schedule.agents[another_node]
                if node != another_node:
                    if (node, another_node) in backup_net.edges():
                        edge_weight = backup_net.get_edge_data(node, another_node, 0)['weight'] 
                        if self.random.random() < (1 - edge_weight):
                            self.friends_net.remove_edge(node, another_node)
                            broken_links_temp.add((node, another_node))
                            num_rewired += 1
                            # agent1.friends_out.remove(agent2.id)
                            # agent2.friends_in.remove(agent1.id)
        # then go through each pair again and determine new links based on their common friends in the old network
        # print("number of links broken: ", num_rewired)
        new_links_temp = set()
        num_edges2 = len(self.friends_net.edges())
        while num_rewired > 0:
            nodes = [node for node in range(self.num_agents)]
            edges_shuffled = [(node, another_node) for node, another_node in list(itertools.permutations(nodes, 2))]
            self.random.shuffle(edges_shuffled)
            for node, another_node in edges_shuffled:
                if not self.friends_net.has_edge(node, another_node):
                    # Code to handle the case when the edge doesn't exist
                    agent1 = self.schedule.agents[node]
                    agent2 = self.schedule.agents[another_node]
                    prob = agent1.rewiring_probability(agent2, backup_net)
                    # self.probabilities["total"].append(prob)
                    if self.random.random() < prob and num_rewired > 0:
                        self.friends_net.add_edge(node, another_node, weight=prob)
                        new_links_temp.add((node, another_node))
                        num_rewired -= 1
    
        # some links that were broken might have emergeda again, so true sets should be recalculated
        new_links = new_links_temp.difference(new_links_temp.intersection(broken_links_temp))
        broken_links = broken_links_temp.difference(new_links_temp.intersection(broken_links_temp))
        self.links_emerged[self.stepcounter] = new_links
        self.links_broke[self.stepcounter] = broken_links
        self.friends_evolution[self.stepcounter] = self.friends_net
        self.clustering_friends = nx.clustering(self.friends_net)
        num_edges3 = len(self.friends_net.edges())
        # print('# new links:', len(self.links_emerged[self.stepcounter]), 'density: ', nx.density(self.friends_net))