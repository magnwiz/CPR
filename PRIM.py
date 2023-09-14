from SALib.sample import (morris, latin, saltelli, fast_sampler)
from SALib.analyze import (morris, sobol, fast)
import crate_n_run as cnr
import pandas as pd
from ema_workbench import RealParameter, IntegerParameter, ScalarOutcome, Constant, Model, ArrayOutcome, TimeSeriesOutcome, \
MultiprocessingEvaluator, ema_logging, perform_experiments, Constraint, save_results
from ema_workbench.em_framework.optimization import epsilon_nondominated, to_problem
import csv
ema_logging.log_to_stderr(ema_logging.INFO)

uncertainty_dict = {
    # "num_ticks": (1000, 5000),
    # "n_actions": (2, 10),
    # "max_action": (10, 20),
    # "weight_triadic": (0, 0.99),
    # "weight_attribute": (0, 0.99),
    # "weight_geo": (0, .99),
    "n_fragments": (1, 4),
    # "rewiring_rate": (1, 100),
    "num_voting_agents": (1, 100),
    # "emergence_time": (100, 500),
    # "ba_m": (2, 5),
    # "ws_k": (2, 5),
    # "ws_p": (0, 1),
    # "k_0": (15000, 40000),
    # "r": (0, 1),
    # "energy_consumption": (2, 20),
    # "threshold_institutional_change": (0.2, 0.9),
    # "innovation_rate": (0, 1),
    # "max_social_influence": (0, 1)
}

constant_values = {
    "num_ticks": 2000,
    # "n_actions": 5,
    # "max_action": 12,
    # "weight_triadic": 0.33,
    # "weight_attribute": 0.33,
    # "weight_geo": 0.33,
    # "n_fragments": 2,
    # "rewiring_rate": 10,
    # "num_voting_agents": 50,
    "emergence_time": 200,
    "ba_m": 3,
    "ws_k": 2,
    "ws_p": 0.4,
    # "k_0": 30000,
    # "r": 0.25,
    # "energy_consumption": 5,
    "threshold_institutional_change": 0.3,
    "innovation_rate": 0.5,
    "max_social_influence": 0.5,
}

uncertainty_data_types = {
    "num_ticks": int,
    "n_actions": int,
    "max_action": int,
    "weight_triadic": float,
    "weight_attribute": float,
    "weight_geo": float,
    "n_fragments": int,
    "rewiring_rate": int,
    "num_voting_agents": int,
    "emergence_time": int,
    "ba_m": int,
    "ws_k": int,
    "ws_p": float,
    "k_0": int,
    "r": float,
    "energy_consumption": int,
    "threshold_institutional_change": float,
    "innovation_rate": float,
    "max_social_influence": float
}

if __name__ == '__main__':
    model = Model('CPR', function=cnr.create_n_run)

    model.uncertainties = [
        # IntegerParameter('n_actions', 2, 10),
        # IntegerParameter('max_action', 10, 20),
        IntegerParameter('n_fragments', 1, 4),
        # IntegerParameter('rewiring_rate', 1, 100),
        # IntegerParameter('num_voting_agents', 1, 100),
        # IntegerParameter('energy_consumption', 5, 20),
        # RealParameter('innovation_rate', 0, 1),
        
    ]

    model.constants = [
        Constant("num_ticks", 2000),
        Constant("emergence_time", 200),
        Constant("ba_m", 3),
        Constant("ws_k", 2),
        Constant("ws_p", 0.4),
        Constant("threshold_institutional_change", 0.3),
        Constant("max_social_influence", 0.5)
    ] 

    model.outcomes = [
            ScalarOutcome("num_institutions"), # kind=ScalarOutcome.MAXIMIZE), 
            ScalarOutcome("emergence_time"),     
            ScalarOutcome("mean_inst_age"), #kind=ScalarOutcome.MAXIMIZE),                        # Scalar outcome
            ScalarOutcome("variance_inst_age"), #kind=ScalarOutcome.MINIMIZE),
            ScalarOutcome("agents_avg_consumption"),                # Scalar outcome #kind=ScalarOutcome.MAXIMIZE),                
            ScalarOutcome("mean_in_degree"), #kind=ScalarOutcome.MAXIMIZE),                        # Scalar outcome
            ScalarOutcome("variance_in_degree"), #kind=ScalarOutcome.MINIMIZE), 
            ScalarOutcome("mean_out_degree"), #kind=ScalarOutcome.MAXIMIZE),                        # Scalar outcome
            ScalarOutcome("variance_out_degree"), #kind=ScalarOutcome.MINIMIZE),                   # Scalar outcome
            ScalarOutcome("mean_comb_degree"),# kind=ScalarOutcome.MAXIMIZE),                      # Scalar outcome
            ScalarOutcome("variance_comb_degree"),# kind=ScalarOutcome.MINIMIZE),  
            ScalarOutcome("friends_density"),                # Scalar outcome
            ScalarOutcome("resource"),
            ScalarOutcome("time_step"),
            ScalarOutcome("max_action"), # <=== Here and lower are initial parameters
            ScalarOutcome("n_actions"),
            ScalarOutcome("w_triadic"),
            ScalarOutcome("w_attribute"),
            ScalarOutcome("w_geo"),
            ScalarOutcome("n_fragments"),
            ScalarOutcome("max_fragment"),
            ScalarOutcome("rewiring_rate"),
            ScalarOutcome("num_voting_agents"),
            ScalarOutcome("emergence_rate"),
            ScalarOutcome("energy_consumption"),
            ScalarOutcome("initial_resource"),
            ScalarOutcome("r"),
            ScalarOutcome("threshold_institutional_change"),
            ScalarOutcome("innovation_rate"),
            ScalarOutcome("max_social_influence"),
            ScalarOutcome("max_fine"),
            ScalarOutcome("max_monitoring"),
            ScalarOutcome("monitoring_cost_weight"),
            ScalarOutcome("seed")
              #, kind=ScalarOutcome.MINIMIZE)
    ]

    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        res = evaluator.perform_experiments(scenarios=1000)
        experiments_df, outcomes_dict = res
        # for (experiments_df, outcomes_dict) in res:
        # if seed == 0: # first iteration
        data = experiments_df  
        data_input = pd.DataFrame
        for outcome in outcomes_dict.keys():
            data[outcome] = outcomes_dict[outcome]
        # save data frame   
        file_path = "network_results/" + "network_data.csv"
        data.to_csv(file_path, index=False)
        file_path = "network_results/" + "network_data.tar.gz"
        save_results(res, file_path)

        