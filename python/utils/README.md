# Utilities for optimization

### Structure of this directory

* `datasets`: Utilities to load datasets available in the `data` folder.


* `elasticity.py`: Boundary conditions and a linear elastic model which 
  loads a precomputed stiffness matrix. The linear elastic model is passed 
  as a parameter to the `AdjointControlProblem` class.


* `objective_functions.py`: Terms of the cost function, including a data 
  attachment term and a L2 penalty term on the forces. Those terms are 
  passed as parameters to the `AdjointControlProblem` class.


* `optimization_problems.py`: Contains the `AdjointControlProblem` class 
  which anages the adjoint procedure, and a wrapper to print iterations and 
  save results when using a Scipy optimization solver.


* `results_processing.py`: Utilities to produce deformed meshes for 
  viewing with Paraview.


* `sofa_models.py`: Linear elastic model where the stiffness matrix is 
  computed using SOFA at the beginning of the procedure.
