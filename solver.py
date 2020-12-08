# works with paid / edu CPLEX & docplex version. 
# has many constraints. 3D model, slow and complicated, but linear
import utils
import utils as ut
import parse as ps
import networkx as nx
from docplex.mp.model import Model
import numpy as np
import pandas
from contextlib import redirect_stdout
import sys


def process_input(filename):
    G, S_MAX = ps.read_input_file(filename)
    N = len(G)
    D = nx.to_dict_of_dicts(G)
    R = range(N)
    S_mat = np.zeros((N,N))
    H_mat = np.zeros((N,N))
    dic = D.copy()
    idx = []
    for i in R:
        dic.pop(i);
        idx.extend([(i, i, k) for k in R])
        for j in dic.keys():
            info = D[i][j]
            S_mat[i,j]=info['stress']
            H_mat[i,j]=info['happiness']
            idx.extend([(i, j, k) for k in R])
    #         D[i][j]['h'] = D[i][j].pop('happiness')
    #         D[i][j]['s'] = D[i][j].pop('stress')
    #         print("{}-{}: h={}, s={}".format(i, j, info['happiness'], info['stress']))
#     print(S_mat)
#     print(H_mat)
    print(len(idx))
    return G, S_MAX, N, R, S_mat, H_mat, idx

def init_model(N):
# ref to the mp sudoku example: https://ibmdecisionoptimization.github.io/docplex-doc/mp/creating_model.html
    global model, x
    model = Model("breakout_room_mp")
    R = range(N)
    # X is room assignment; key is (i, j, r) where i,j is a pair of students, and r is the room they are assigned to
    x = model.binary_var_dict(idx, name="X") 
    # room_count = model.integer_var(1, N, "room_count")

    # constraint 1: each pair can only be assigned to one room. 
    # -> for each pair, sum of all different room choices == 1
    for i in R:
    #     model.add_constraint(model.sum(x[j,i,k] for j in range(i) for k in R)
    #                     +    model.sum(x[i,j,k] for j in range(i,N) for k in R) == 1)
        for j in range(i, N):
            pair_sum = model.sum(x[i,j,k] for k in R)
            model.add_constraint(pair_sum <= 1)

    # constraint *2: generate "clique" for people appeared in a pair in a room. using triangle/rectangle, in each k layer.
    # constraint 3: everyone can only be assigned to one room.
    # -> other room sum of pairs including someone assigned == 0
    for k in R:
        for a in R: 
            for b in range(a + 1,N):
                model.add_constraint(model.indicator_constraint(
                    x[a,b,k], 
                    model.sum(x[a,i,r] for i in range(a+1,N) for r in R if r != k)+ 
                    model.sum(x[i,a,r] for i in range(a) for r in R if r != k)
                    ==0))
                model.add_constraint(model.indicator_constraint(
                    x[a,b,k], 
                    model.sum(x[b,i,r] for i in range(b+1,N) for r in R if r != k)+
                    model.sum(x[i,b,r] for i in range(b) for r in R if r != k)
                    == 0))
                for c in range(b + 1,N):
                    model.add_constraint(x[a,b,k] + x[a,c,k] + x[b,c,k] != 2)
                    for d in range(c + 1,N):
                        model.add_constraint(model.if_then((x[a,b,k] + x[c,d,k]) == 2, x[a,c,k] == 1))
    return model, x

    # constraint 2: rooms are using consecutive numbers starting from 0. 
    # if nobody is in one room, do not assign people to rooms with no > this one
    # also: calculate room count
    # now: count all partners in the assigned group!
    # for k in range(N-1):
    # #     room_sum = model.sum(x[i,j,k] for i in R for j in range(i,N))
    #     for i in R:
    #         for j in range(i, N):
    #             model.add_constraint(model.if_then(room_sum == 0, x[i,j,k+1]==0))
    #             model.add_constraint(model.if_then(room_sum == 0, room_count <= k))
    #             model.add_constraint(model.if_then(room_sum != 0, room_count >= k + 1))

def model_and_solve(NBROOMS):
# ref to the mp sudoku example: https://ibmdecisionoptimization.github.io/docplex-doc/mp/creating_model.html
    global model
        # constraint 3: stress level of each room is under the limit. 
    # how to find the number of rooms assigned? <- create a new variable
    constraints = []
    for r in R:
        S_room = model.sum(S_mat[i,j] * x[i,j,r] for i in R for j in range(i,N))
        constraints.append(model.add_constraint(S_room <= S_MAX / NBROOMS))
    # goal: maximize happiness
    model.maximize(model.sum(H_mat[i,j] * x[i,j,k] for i in R for j in range(i,N) for k in R))
    model.set_time_limit(5)
    msol = model.solve()
    model.remove_constraints(constraints)
    model.remove_objective()
    print(f"happiness: {msol.objective_value}")
#     print(msol)
    return msol

def clean_model_solution(msol, x, N):
    msol_dict = msol.get_value_dict(x)
    msol_np = np.concatenate((list(msol_dict.keys()), np.array([list(msol_dict.values())]).T), axis=1).astype(int)
    msol_nz = msol_np[msol_np[:,3].nonzero()]
    msol_rooms = msol_nz[:,2]
    msol_unique_rooms = np.unique(msol_rooms)
    msol_unique_rooms
    number_of_rooms = len(msol_unique_rooms)
    for i in range(number_of_rooms):
        msol_rooms = np.where(msol_rooms==msol_unique_rooms[i],i,msol_rooms)
    msol_rooms
    msol_nz[:,2]=msol_rooms
    for i in np.setdiff1d(np.arange(N), np.unique(np.concatenate([msol_nz[:,0],msol_nz[:,1]]))):
        msol_nz = np.append(msol_nz, [[i,i, number_of_rooms,1]],axis=0)
        msol_rooms = np.append(msol_rooms, number_of_rooms)
        number_of_rooms += 1
    print(msol_nz, number_of_rooms)
    return msol_nz, msol_rooms, number_of_rooms
    
def is_valid(msol_nz, number_of_rooms):
    S_room_max = S_MAX/number_of_rooms
    for i in range(number_of_rooms):
        room = msol_nz[np.argwhere(msol_nz[:,2] == i).flatten()]
        S_r = S_mat[room[:,0], room[:,1]].sum()
        if (S_r > S_room_max):
            print(f"room {i} stress = {S_r}, exceeded {S_room_max}")
            return False
    return True

def to_dict(msol_nz, msol_rooms, number_of_rooms):
    sol_dict = {}
    for i in range(number_of_rooms):
        sel = msol_nz[np.argwhere(msol_rooms == i).flatten()]
        sol_dict.update(dict(zip(sel[:,0], sel[:,2])))
        sol_dict.update(dict(zip(sel[:,1], sel[:,2])))
    return sol_dict

# processes to get valid solution
def solve_process(input_path):
    global G, S_MAX, N, R, S_mat, H_mat, idx, msol, x
    G, S_MAX, N, R, S_mat, H_mat, idx = process_input(input_path)
#     model, x = init_model(N)
    nbrooms = 1
    while(nbrooms < N):
        msol = model_and_solve(nbrooms)
        sol_nz, msol_rooms, number_of_rooms = clean_model_solution(msol, x, N)
        if (number_of_rooms != N and is_valid(sol_nz, number_of_rooms)):
            print(f"success! nbrooms = {nbrooms}")
            return to_dict(sol_nz, msol_rooms, number_of_rooms)
        print(f"nbrooms = {nbrooms} failed. Trying {nbrooms + 1}")
        nbrooms += 1
    print("FAILED to find good solution")
    return {}
# sol_dict, number_of_rooms, sol_nz = solve_process("inputs_outputs/20.in")

G, S_MAX, N, R, S_mat, H_mat, idx = process_input("inputs/small-1.in")
init_model(10)
print('init model complete')
with open('log-10-200.txt', 'a+') as f:
    with redirect_stdout(f):
        print('LOG STARTS')
        for i in range(1,2):
            print(f"\n--------------\nCURRENT i = {i}\n")
            sys.stdout.flush()
            try:
                sol_dict = solve_process(f"inputs/small-{i}.in")
                sys.stdout.flush()
                ps.write_output_file(sol_dict, f"outputs/small-{i}.out")
                sys.stdout.flush()
            except: # catch *all* exceptions
                e = sys.exc_info()[0]
                print(f"ERROR OCCURED: i = {i}, {repr(e)}")
                sys.stdout.flush()
                continue