import numpy as np
from scipy.spatial.distance import cdist
from gurobipy import Model, GRB, quicksum
import time
from iterative_rounding import iterative_rounding_lp

def fair_partial_assignment(df, centers, alpha, beta, color_flag, clustering_method):
    print("MPHKA FAIR partial_assignment")
    if clustering_method in ["kmeans", "kmedian"]:
        cost_fun_string = 'euclidean' if clustering_method == "kmedian" else 'sqeuclidean'
        problem, objective, variable_names = fair_partial_assignment_lp_solver(df, centers, color_flag, alpha, beta, cost_fun_string)

        t1 = time.monotonic()
        problem.optimize()
        t2 = time.monotonic()
        print("LP solving time = {}".format(t2 - t1))
        if problem.Status == GRB.OPTIMAL:
            assignment = [problem.getVarByName(name).X for name in variable_names]
        else:
            print("Gurobi failed to solve the model. Status:", problem.Status)
            assignment = None
        assignment = [problem.getVarByName(name).X for name in variable_names]

        res = {
            "status": problem.Status,
            "success": problem.Status == GRB.OPTIMAL,
            "objective": problem.ObjVal,
            "assignment": assignment,
        }

        final_res = iterative_rounding_lp(df, centers, objective, color_flag, res)
        final_res["partial_assignment"] = res["assignment"]
        final_res["partial_objective"] = res["objective"]

        if clustering_method == "kmeans":
            final_res["partial_objective"] = np.sqrt(final_res["partial_objective"])
            final_res["objective"] = np.sqrt(final_res["objective"])

        return final_res

    elif clustering_method == "kcenter":
        problem, objective, variable_names = fair_partial_assignment_lp_solver(df, centers, color_flag, alpha, beta, 'sqeuclidean')
        problem.optimize()

        cost_ub = max(objective) + 1
        cost_lb = 0
        lowest_feasible_cost = cost_ub
        cheapest_feasible_assignment = None
        cheapest_feasible_obj = None

        while cost_ub > cost_lb + 0.1:
            cost_mid = (cost_ub + cost_lb) / 2.0
            new_problem, new_objective, new_variable_names = fair_partial_assignment_lp_solver(df, centers, color_flag, alpha, beta, 'sqeuclidean')
            for i, obj in enumerate(new_objective):
                if obj > cost_mid:
                    new_problem.getVarByName(new_variable_names[i]).UB = 0

            new_problem.optimize()

            if new_problem.Status == GRB.OPTIMAL:
                cost_ub = cost_mid
                lowest_feasible_cost = cost_mid
                cheapest_feasible_assignment = [new_problem.getVarByName(name).X for name in new_variable_names]
                cheapest_feasible_obj = new_objective
            elif new_problem.Status == GRB.INFEASIBLE:
                cost_lb = cost_mid
            else:
                raise ValueError("Solver status: {} at cost {}".format(new_problem.Status, cost_mid))

        res = {
            "status": GRB.OPTIMAL,
            "success": True,
            "objective": cost_ub,
            "assignment": cheapest_feasible_assignment,
        }

        final_res = iterative_rounding_lp(df, centers, cheapest_feasible_obj, color_flag, res)
        final_res["objective"] = np.sqrt(max([v * c for v, c in zip(final_res["assignment"], cheapest_feasible_obj)]))
        final_res["partial_objective"] = np.sqrt(lowest_feasible_cost)
        final_res["partial_assignment"] = cheapest_feasible_assignment

        return final_res

    else:
        print("Invalid clustering method. Choose from 'kmeans', 'kmedian', or 'kcenter'.")
        return None

def fair_partial_assignment_lp_solver(df, centers, color_flag, alpha, beta, cost_fun_string):
    print("Initializing Gurobi model")
    problem = Model("FairAssignment")
    problem.setParam('OutputFlag', 0)
    problem.ModelSense = GRB.MINIMIZE

    objective, lower_bounds, upper_bounds, variable_names = prepare_to_add_variables(df, centers, cost_fun_string)

    variables = {}
    for name, obj, lb, ub in zip(variable_names, objective, lower_bounds, upper_bounds):
        variables[name] = problem.addVar(obj=obj, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=name)

    problem.update()

    constraints_row, senses, rhs, constraint_names = prepare_to_add_constraints(df, centers, color_flag, beta, alpha)

    for idx, ((var_names, coeffs), sense, rhs_val) in enumerate(zip(constraints_row, senses, rhs)):
        expr = quicksum(coeff * variables[var_name] for var_name, coeff in zip(var_names, coeffs))
        if sense == "E":
            problem.addConstr(expr == rhs_val, name=constraint_names[idx])
        elif sense == "L":
            problem.addConstr(expr <= rhs_val, name=constraint_names[idx])
        elif sense == "G":
            problem.addConstr(expr >= rhs_val, name=constraint_names[idx])

    return problem, objective, variable_names

def prepare_to_add_variables(df, centers, cost_fun_string):
    num_points = len(df)
    num_centers = len(centers)
    variable_names = ["x_{}_{}".format(j, i) for j in range(num_points) for i in range(num_centers)]
    total_variables = num_points * num_centers
    lower_bounds = [0] * total_variables
    upper_bounds = [1] * total_variables
    objective = cost_function(df, centers, cost_fun_string)
    return objective, lower_bounds, upper_bounds, variable_names

def cost_function(df, centers, cost_fun_string):
    print("centers shape:", np.array(centers).shape)
    all_pair_distance = cdist(df.values, centers, cost_fun_string)
    return all_pair_distance.ravel().tolist()

def prepare_to_add_constraints(df, centers, color_flag, beta, alpha):
    num_points = len(df)
    num_centers = len(centers)
    constraints_row, rhs = constraint_sums_to_one(num_points, num_centers)
    sum_const_len = len(rhs)

    for var in color_flag:
        var_color_flag, var_beta, var_alpha = color_flag[var], beta[var], alpha[var]
        color_constraint, color_rhs = constraint_color(num_points, num_centers, var_color_flag, var_beta, var_alpha)
        constraints_row.extend(color_constraint)
        rhs.extend(color_rhs)

    senses = ["E"] * sum_const_len + ["L"] * (len(rhs) - sum_const_len)
    constraint_names = ["c_{}".format(i) for i in range(len(rhs))]
    return constraints_row, senses, rhs, constraint_names

def constraint_sums_to_one(num_points, num_centers):
    constraints = [[["x_{}_{}".format(j, i) for i in range(num_centers)], [1] * num_centers] for j in range(num_points)]
    rhs = [1] * num_points
    return constraints, rhs

def constraint_color(num_points, num_centers, color_flag, beta, alpha):
    beta_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                         [beta[color] - 1 if color_flag[j] == color else beta[color] for j in range(num_points)]]
                        for i in range(num_centers) for color in beta]

    alpha_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                          [np.round(1 - alpha[color], 3) if color_flag[j] == color else -alpha[color] for j in range(num_points)]]
                         for i in range(num_centers) for color in alpha]

    constraints = beta_constraints + alpha_constraints
    rhs = [0] * len(constraints)
    return constraints, rhs
