# CS 170 Fall 2020 Final Project - Zoom Breakout Rooms

[Project Spec](https://drive.google.com/file/d/1gVRdr8cV3oXGI6lyRvB0CuJocENiIGyw/view?usp=sharing)

Constrained optimization problem. Each pair of students i, j have corresponding s_i,j and h_i,j for stress and happiness. The goal is to assign them to k rooms and maximize the total happiness value H while keeping total stress below S_max/k. 

Check out `breakout-rooms.ipynb` for code at the exploration stage including outputs with longer comments related to breaking down and modeling the problem. `cp_optimize.ipynb` has the final code with some inline comments. 

The optimizer used was IBM's CPLEX, more specifically the python package DOcplex. Ref: [IBM Decision Optimization CPLEX Python library (DOcplex) documentation](https://ibmdecisionoptimization.github.io/docplex-doc/) 

## Original Readme

Requirements:

Python 3.6+

You'll only need to install networkx to work with the starter code. For installation instructions, follow: https://networkx.github.io/documentation/stable/install.html

If using pip to download, run `python3 -m pip install networkx`


Files:
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `utils.py`: contains functions to compute cost and validate NetworkX graphs

When writing inputs/outputs:
- Make sure you use the functions `write_input_file` and `write_output_file` provided
- Run the functions `read_input_file` and `read_output_file` to validate your files before submitting!
  - These are the functions run by the autograder to validate submissions
