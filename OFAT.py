from SALib.sample import (morris, latin, saltelli, fast_sampler)
from SALib.analyze import (morris, sobol, fast)
import crate_n_run as cnr
import pandas as pd
from ema_workbench import RealParameter, IntegerParameter, ScalarOutcome, Constant, Model, ArrayOutcome, TimeSeriesOutcome, \
MultiprocessingEvaluator, ema_logging, perform_experiments, Constraint, save_results
from ema_workbench.em_framework.optimization import epsilon_nondominated, to_problem

uncertainty_dict = {
    # "num_ticks": (1000, 5000),
    "n_actions": (2, 10),
    "max_action": (10, 20),
    "weight_triadic": (0, 0.99),
    "weight_attribute": (0, 0.99),
    "weight_geo": (0, 0.99),
    "n_fragments": (1, 4),
    "rewiring_rate": (1, 100),
    "num_voting_agents": (1, 100),
    "emergence_time": (100, 500),
    "ba_m": (2, 5),
    "ws_k": (2, 5),
    "ws_p": (0, 1),
    # "k_0": (15000, 40000),
    # "r": (0, 1),
    # "energy_consumption": (2, 20),
    "threshold_institutional_change": (0.2, 0.9),
    "innovation_rate": (0, 1),
    "max_social_influence": (0, 1)
}

constant_values = {
    "num_ticks": 2000,
    "n_actions": 5,
    "max_action": 12,
    # "weight_triadic": 0.33,
    # "weight_attribute": 0.33,
    # "weight_geo": 0.33,
    "n_fragments": 2,
    "rewiring_rate": 10,
    "num_voting_agents": 50,
    "emergence_time": 200,
    "ba_m": 3,
    "ws_k": 2,
    "ws_p": 0.4,
    "k_0": 30000,
    "r": 0.25,
    "energy_consumption": 5,
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

outcome_names = [
    "num_institutions",
    "agents_avg_consumption",
    "mean_in_degree",
    "variance_in_degree",
    "mean_comb_degree",
    "variance_comb_degree",
    "mean_inst_age",
    "variance_inst_age",
    "seed"
]

experiments = {}

# def sum_constraint(weight_triadic, weight_attribute, weight_geo):
#     return abs(weight_triadic + weight_attribute + weight_geo - 1)
if __name__ == '__main__':
    for uncertainty, bounds in uncertainty_dict.items():
        print(uncertainty, bounds)
        model = Model("CPR", function=cnr.create_n_run)
    
        if uncertainty_data_types[uncertainty] == int:
            model.uncertainties = [IntegerParameter(uncertainty, bounds[0], bounds[1])] 
        else:
            model.uncertainties = [RealParameter(uncertainty, bounds[0], bounds[1])]

        constants = constant_values.copy()
        constants.pop(uncertainty)
        
        model.constants = []

        for constant, value in constants.items():
            model.constants[constant] = Constant(constant, value)
        
        # constraint = [Constraint("sum_weights", outcome_names=None, function=sum_constraint)]
        
        model.outcomes = [
            ScalarOutcome("num_institutions"), # kind=ScalarOutcome.MAXIMIZE),                      # Scalar outcome
            ScalarOutcome("agents_avg_consumption"), #kind=ScalarOutcome.MAXIMIZE),                
            ScalarOutcome("mean_in_degree"), #kind=ScalarOutcome.MAXIMIZE),                        # Scalar outcome
            ScalarOutcome("variance_in_degree"), #kind=ScalarOutcome.MINIMIZE),                    # Scalar outcome
            ScalarOutcome("mean_comb_degree"),# kind=ScalarOutcome.MAXIMIZE),                      # Scalar outcome
            ScalarOutcome("variance_comb_degree"),# kind=ScalarOutcome.MINIMIZE),                  # Scalar outcome
            ScalarOutcome("mean_inst_age"), #kind=ScalarOutcome.MAXIMIZE),                        # Scalar outcome
            ScalarOutcome("variance_inst_age"), #kind=ScalarOutcome.MINIMIZE),
            ScalarOutcome("seed")#, kind=ScalarOutcome.MINIMIZE)
        ]

        experiments[uncertainty] = model

    results = {}
    # merged_archives = {}

    for uncertainty, model in experiments.items():
        # results[uncertainty] = []
        data = {}
        # with MultiprocessingEvaluator(model) as evaluator:
            # print(uncertainty, ', current seed iteration:', seed)
            # res = evaluator.perform_experiments(experiments[uncertainty], 1)
            # for seed in range(2):
        experiments_df, outcomes_dict = perform_experiments(experiments[uncertainty], 20)
        # for (experiments_df, outcomes_dict) in res:
        # if seed == 0: # first iteration
        data[uncertainty] = experiments_df  
        for outcome in outcomes_dict.keys():
            data[uncertainty][outcome] = outcomes_dict[outcome]
        # save data frame
        file_path = "OFAT_results/" + uncertainty + ".csv"
        data[uncertainty].to_csv(file_path, index=False)
                # else:
                #     new_data = {}
                #     new_data[uncertainty] = experiments_df
                #     for outcome in outcomes_dict.keys():
                #         new_data[uncertainty][outcome] = outcomes_dict[outcome]
                #     data[uncertainty] = pd.concat([data[uncertainty], new_data[uncertainty]], ignore_index=True)
                
            
            # Remove unwanted columns
            # unwanted_columns = ['scenario', 'policy', 'model']
            # data[uncertainty] = data[uncertainty].drop(columns=unwanted_columns)
            # shape = data[uncertainty].shape   
            # res_df = pd.DataFrame(res)
            # results[uncertainty].append(res)
            # shape = res_df.shape()
        
        # parameter is the same as uncertainty
        # for parameter, list_res in results.items():
        #     for experiments_df, outcomes_dict in list_res:         
        #         for outcome_name in outcome_names:
        #             data[parameter][outcome_name] = outcomes_dict[outcome_name]
                
        # results[uncertainty] = [pd.DataFrame.from_records(result) for result in results[uncertainty]]
        # problem = to_problem(model, searchover='uncertainties')
        # epsilons = [0.05] * len(model.outcomes)
        # merged_archives[uncertainty] = epsilon_nondominated(data[uncertainty], epsilons, problem)
        # file_path = "OFAT_results/" + uncertainty
        # save_results(data[uncertainty], file_path)





