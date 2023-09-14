from SALib.sample import (morris, latin, saltelli, fast_sampler)
from SALib.analyze import (morris, sobol, fast)
import crate_n_run as cnr
from ema_workbench import RealParameter, IntegerParameter, ScalarOutcome, Constant, Model, ArrayOutcome, TimeSeriesOutcome, \
MultiprocessingEvaluator, ema_logging, perform_experiments

model = Model("CPR", function=cnr.create_n_run)

# specify uncertainties
model.uncertainties = [
    IntegerParameter("num_ticks", 1000, 5000),
    IntegerParameter("n_actions", 5, 12),
    IntegerParameter("max_action", 1, 50),
    RealParameter("weight_triadic", 0, 1),
    RealParameter("weight_attribute", 0, 1),
    RealParameter("weight_geo", 0, 1),
    IntegerParameter("n_fragments", 1, 4),
    IntegerParameter("rewiring_rate", 1, 100),
    IntegerParameter("num_voting_agents", 1, 100),
    IntegerParameter("emergence_time", 100, 500),
    IntegerParameter("ba_m", 2, 5),
    IntegerParameter("ws_k", 2, 5),
    RealParameter("ws_p", 0, 1),
    IntegerParameter("k_0", 10000, 50000),
    RealParameter("r", 0, 1),
    IntegerParameter("energy_consumption", 1, 10),
    RealParameter("threshold_institutional_change", 0.2, 0.9)
]

model.outcomes = [
    ScalarOutcome("num_institutions"),
    ArrayOutcome("distribution_probabilities"),
    ArrayOutcome("list_institutions"),
    ArrayOutcome("list_ticks_inst"),
    ArrayOutcome("dict_inst"),
    ArrayOutcome("dict_fine_monitoring"),
    ScalarOutcome("degree_centrality['kins_net']"),
    ScalarOutcome("degree_centrality['geo_net']"),
    ScalarOutcome("degree_centrality['friends_net']"),
    ScalarOutcome("degree_centrality['friends_net (out)']"),
    ScalarOutcome("degree_centrality['friends_net (in)']"),
    ArrayOutcome("agents_consumption")
]

if __name__ == '__main__': 
    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(scenarios=1000)