import utils
import utils as ut
import parse as ps
import networkx as nx
import numpy as np
import cplex
import docplex.cp.model as cp
from contextlib import redirect_stdout
import traceback
import sys

def solve_process(input_path):
    global G, S_MAX, N, R, S_mat, H_mat, idx, sol
    G, S_MAX, N, R, S_mat, H_mat, idx = process_input(input_path)
    sol = cp_model_and_solve(N)
    print(f"objective: {sol.get_objective_values()[0]}")
    sol_dict, actual_nbrooms = clean_solution(sol)
    if (is_valid(sol_dict, actual_nbrooms)):
        print("success!")
        print(sol_dict, actual_nbrooms)
        return sol_dict
#         print(f"nbrooms = {nbrooms} failed.")
#         new_nbrooms = np.round((nbrooms + actual_nbrooms) / 2)
#         nbrooms = new_nbrooms if new_nbrooms > nbrooms else nbrooms + 1
#         print(f"trying {nbrooms}")
#     print("FAILED to find good solution")
#     raise Exception('Could not find viable solution') 

def cp_model_and_solve(N): # N = # of students
    # ref to the mp sudoku example: https://ibmdecisionoptimization.github.io/docplex-doc/mp/creating_mdl.html
    global mdl, v
    mdl = cp.CpoModel("breakout_room_cp")
    R = range(N)
    # v is room assignment; key is (i, j, r) where i,j is a pair of students, and r is the room they are assigned to
    v = mdl.binary_var_dict(idx, name="v") 
    room_count = mdl.integer_var(1, N, "room_count")

    # constraint 1: everyone is assigned to one room. 
    for i in R:
            mdl.add(mdl.sum(v[i,k] for k in R) == 1)
    
    mdl.add(cp.sum(cp.sum(v[i,k] for i in R) != 0 for k in R) == room_count)

    constraints = []
    for r in R:
        S_room = cp.sum(S_mat[i,j] * v[i,r] * v[j,r] for i in R for j in range(i,N))
        constraints.append(mdl.add(S_room * room_count <= S_MAX ))
    # goal: maximize happiness
    mdl.maximize(cp.sum(H_mat[i,j] * v[i,k] * v[j,k] for i in R for j in range(i,N) for k in R))
    cpsol = mdl.solve( RelativeOptimalityTolerance=0.1)
    return cpsol

def clean_solution(sol):
    sol_mat = np.apply_along_axis(lambda x: sol.get_value(v[x[0],x[1]]), 0, np.indices((N,N)))
    cpsol_np = np.vstack((np.nonzero(sol_mat))).T
    cpsol_rooms = cpsol_np[:,1]
    cpsol_unique_rooms = np.unique(cpsol_rooms)
    number_of_rooms = len(cpsol_unique_rooms)
    for i in range(number_of_rooms):
        cpsol_rooms = np.where(cpsol_rooms==cpsol_unique_rooms[i],i,cpsol_rooms)
    cpsol_np[:,1]=cpsol_rooms
    cpsol_dict = {}
    cpsol_dict.update(dict(zip(cpsol_np[:,0], cpsol_np[:,1])))
    return cpsol_dict, number_of_rooms

def is_valid(cpsol_dict, number_of_rooms):
    return ut.is_valid_solution(cpsol_dict, G, S_MAX, number_of_rooms)

def process_input(filename):
    G, S_MAX = ps.read_input_file(filename)
    N = len(G)
    D = nx.to_dict_of_dicts(G)
    R = range(N)
    S_mat = np.zeros((N,N))
    H_mat = np.zeros((N,N))
    dic = D.copy()
    idx = [(i,k) for i in R for k in R]
    for i in R:
        dic.pop(i);
        for j in dic.keys():
            info = D[i][j]
            S_mat[i,j]=info['stress']
            H_mat[i,j]=info['happiness']
    return G, S_MAX, N, R, S_mat, H_mat, idx



# with open('log-med-1-50.txt', 'a+') as f:
#     with redirect_stdout(f):
#         print('MEDIUM LOG STARTS')
#         for i in range(1,50):
#             print(f"\n--------------\nCURRENT i = {i}\n")
#             sys.stdout.flush()
#             try:
#                 sol_dict = solve_process(f"inputs/small-{i}.in")
#                 sys.stdout.flush()
#                 ps.write_output_file(sol_dict, f"cp_outputs/small-{i}.out")
#                 sys.stdout.flush()
#             except: # catch *all* exceptions
#                 # e = sys.exc_info()[0]
#                 etype, value, t = sys.exc_info()
#                 # print('Error opening %s: %s' % (value.filename, value.strerror))
#                 print(f"ERROR OCCURED: i = {i}, {value}")
#                 traceback.print_exc()
#                 sys.stdout.flush()
#                 continue

solve_process("inputs/medium-1.in")