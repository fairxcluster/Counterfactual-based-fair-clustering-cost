import numpy as np
from gurobipy import Model, GRB, quicksum
from scipy.spatial.distance import pdist, squareform
import time

def violating_lp_clustering(df, num_centers, alpha, beta, color_flag, clustering_method, violation):
    if clustering_method in ["kmeans", "kmedian"]:
        cost_fun_string = 'euclidean' if clustering_method == "kmedian" else 'sqeuclidean'
        problem, objective, variable_names = violating_clustering_lp_solver(df, num_centers, color_flag, alpha, beta, violation, cost_fun_string)

        t1 = time.monotonic()
        problem.optimize()
        t2 = time.monotonic()
        print("LP solving time = {}".format(t2 - t1))

        assignment = [problem.getVarByName(name).X for name in variable_names]
        objective_value = problem.ObjVal
        if clustering_method == "kmeans":
            objective_value = np.sqrt(objective_value)

        res = {
            "status": problem.Status,
            "success": problem.Status == GRB.OPTIMAL,
            "objective": objective_value,
            "assignment": assignment,
        }
        return res

    elif clustering_method == "kcenter":
        problem, objective, variable_names = violating_clustering_lp_solver(df, num_centers, color_flag, alpha, beta, violation, 'sqeuclidean')
        cost_ub = max(objective) + 1
        cost_lb = 0
        lowest_feasible_cost = cost_ub
        best_assignment = None

        while cost_ub > cost_lb + 0.1:
            cost_mid = (cost_ub + cost_lb) / 2.0
            new_problem, new_objective, new_variable_names = violating_clustering_lp_solver(df, num_centers, color_flag, alpha, beta, violation, 'sqeuclidean')

            for i, obj in enumerate(new_objective):
                if obj > cost_mid:
                    new_problem.getVarByName(new_variable_names[i]).UB = 0

            new_problem.optimize()

            if new_problem.Status == GRB.OPTIMAL:
                cost_ub = cost_mid
                lowest_feasible_cost = cost_mid
                best_assignment = [new_problem.getVarByName(name).X for name in new_variable_names]
            else:
                cost_lb = cost_mid

        res = {
            "status": GRB.OPTIMAL,
            "success": True,
            "objective": np.sqrt(lowest_feasible_cost),
            "assignment": best_assignment,
        }
        return res

    else:
        print("Invalid clustering method. Choose from 'kmeans', 'kmedian', or 'kcenter'.")
        return None

def violating_clustering_lp_solver(df, num_centers, color_flag, alpha, beta, violation, cost_fun_string):
    print("Initializing Gurobi model")
    problem = Model("ViolatingClustering")
    problem.setParam('OutputFlag', 0)
    problem.ModelSense = GRB.MINIMIZE

    objective, lower_bounds, upper_bounds, variable_names = prepare_to_add_variables(df, cost_fun_string)

    variables = {}
    for name, obj, lb, ub in zip(variable_names, objective, lower_bounds, upper_bounds):
        variables[name] = problem.addVar(obj=obj, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=name)

    problem.update()

    constraints_row, senses, rhs, constraint_names = prepare_to_add_constraints(df, num_centers, color_flag, beta, alpha, violation)

    for idx, ((var_names, coeffs), sense, rhs_val) in enumerate(zip(constraints_row, senses, rhs)):
        expr = quicksum(coeff * variables[var_name] for var_name, coeff in zip(var_names, coeffs))
        if sense == "E":
            problem.addConstr(expr == rhs_val, name=constraint_names[idx])
        elif sense == "L":
            problem.addConstr(expr <= rhs_val, name=constraint_names[idx])
        elif sense == "G":
            problem.addConstr(expr >= rhs_val, name=constraint_names[idx])

    return problem, objective, variable_names

def prepare_to_add_variables(df, cost_fun_string):
    num_points = len(df)
    variable_assn_names = ["x_{}_{}".format(i, j) for i in range(num_points) for j in range(num_points)]
    variable_facility_names = ["y_{}".format(i) for i in range(num_points)]
    variable_names = variable_assn_names + variable_facility_names
    total_variables = num_points * num_points + num_points
    lower_bounds = [0] * total_variables
    upper_bounds = [1] * total_variables
    objective = cost_function(df, cost_fun_string)
    return objective, lower_bounds, upper_bounds, variable_names

def cost_function(df, cost_fun_string):
    all_pair_distance = pdist(df.values, cost_fun_string)
    all_pair_distance = squareform(all_pair_distance)
    all_pair_distance = all_pair_distance.ravel().tolist()
    pad_for_facility = [0] * len(df)
    return all_pair_distance + pad_for_facility

def prepare_to_add_constraints(df, num_centers, color_flag, beta, alpha, violation):
    num_points = len(df)
    sum_constraints, sum_rhs = constraint_sums_to_one(num_points)
    validity_constraints, validity_rhs = constraint_validity(num_points)
    facility_constraints, facility_rhs = constraint_facility(num_points, num_centers)
    constraints_row = sum_constraints + validity_constraints + facility_constraints
    rhs = sum_rhs + validity_rhs + facility_rhs
    num_equality_constraints = len(sum_rhs)

    for var in color_flag:
        var_color_flag, var_beta, var_alpha = color_flag[var], beta[var], alpha[var]
        color_constraint, color_rhs = constraint_color(num_points, var_color_flag, var_beta, var_alpha, violation)
        constraints_row.extend(color_constraint)
        rhs.extend(color_rhs)

    num_inequality_constraints = len(rhs) - num_equality_constraints
    senses = ["E"] * num_equality_constraints + ["L"] * num_inequality_constraints
    constraint_names = ["c_{}".format(i) for i in range(len(rhs))]
    return constraints_row, senses, rhs, constraint_names

def constraint_sums_to_one(num_points):
    constraints = [[["x_{}_{}".format(j, i) for i in range(num_points)], [1] * num_points] for j in range(num_points)]
    rhs = [1] * num_points
    return constraints, rhs

def constraint_validity(num_points):
    constraints = [[["x_{}_{}".format(j, i), "y_{}".format(i)], [1, -1]] for j in range(num_points) for i in range(num_points)]
    rhs = [0] * (num_points * num_points)
    return constraints, rhs

def constraint_facility(num_points, num_centers):
    constraints = [[["y_{}".format(i) for i in range(num_points)], [1] * num_points]]
    rhs = [num_centers]
    return constraints, rhs

def constraint_color(num_points, color_flag, beta, alpha, violation):
    beta_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                         [beta[color] - 1 if color_flag[j] == color else beta[color] for j in range(num_points)]]
                        for i in range(num_points) for color in beta]

    alpha_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                          [np.round(1 - alpha[color], 3) if color_flag[j] == color else -alpha[color] for j in range(num_points)]]
                         for i in range(num_points) for color in alpha]

    constraints = beta_constraints + alpha_constraints
    rhs = [violation] * len(constraints)
    return constraints, rhs
