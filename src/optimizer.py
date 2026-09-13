import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination

# Simulated evaluator for the Recommender Pipeline
def evaluate_pipeline(learning_rate, noise_multiplier):
    """
    In a real scenario, this function would trigger a federated training run,
    then evaluate the model to compute RMSE (accuracy), and Gini index (fairness).
    For demonstration, we use a synthetic function to represent the trade-off.
    Higher noise = lower privacy risk (better), but higher RMSE (worse).
    """
    # Simulated RMSE (Accuracy loss, minimize)
    rmse = 0.8 + (noise_multiplier * 0.2) + (abs(learning_rate - 0.05) * 5)
    
    # Simulated Popularity Bias (Fairness loss, minimize)
    # Higher noise can sometimes help exploration/fairness by disrupting the "rich get richer" dynamic
    fairness_loss = 0.5 - (noise_multiplier * 0.1)
    
    return rmse, fairness_loss

class RecommenderOptimizationProblem(ElementwiseProblem):
    def __init__(self):
        # 2 variables: learning_rate [0.001, 0.1], noise_multiplier [0.1, 1.0]
        # 2 objectives: minimize RMSE, minimize Fairness Loss
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=0, xl=np.array([0.001, 0.1]), xu=np.array([0.1, 1.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        learning_rate = x[0]
        noise_multiplier = x[1]
        
        rmse, fairness_loss = evaluate_pipeline(learning_rate, noise_multiplier)
        
        out["F"] = [rmse, fairness_loss]

def run_optimization():
    problem = RecommenderOptimizationProblem()

    algorithm = NSGA2(
        pop_size=20,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 10)

    print("Running Multi-Objective Optimization (NSGA-II) for Recommender Fairness Trade-offs...")
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    print(f"Pareto Front Found: {len(res.F)} solutions")
    for i, (f, x) in enumerate(zip(res.F, res.X)):
        print(f"Solution {i+1}: LR={x[0]:.4f}, Noise={x[1]:.4f} => RMSE={f[0]:.4f}, Fairness Loss={f[1]:.4f}")
        
    return res

if __name__ == "__main__":
    res = run_optimization()
    print("Optimization complete. Pareto front calculated.")
