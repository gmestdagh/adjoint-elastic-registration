"""Estimate a sequence of local forces applied on the liver.

Data generation and surface forces reconstruction both involve a
Neo-Hookean elastic model, resulting in a much larger execution time
than liver_sequence.py.
"""
import os
import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b
from time import time

from utils.optimization_problems import AdjointControlProblem, ScipyWrapper
from utils.objective_functions import DataAttachment
from utils.elasticity import DirichletBoundaryCondition
from utils.fenicsx_models import NeoHookeanModel
from utils.datasets import liver

# Load dataset
# Available sequences are s1, ..., s5.
# Available meshes are 'default' (used for data generation) and 'remeshed'.
seq_name = 's1'
ds = liver.load_sequence(seq_name, which_mesh='default', model='neohookean',
                         mesh_format='xdmf')

# Prepare output saving
save_folder = os.path.join(liver.RESULTS_FOLDER + '_sequences_nhk', seq_name)

###############################################################################
# CREATE PROBLEM
###############################################################################
# Initialize elastic problem
bc = DirichletBoundaryCondition(ds.dirichlet_vertices)
elastic_model = NeoHookeanModel(
    mesh_filename=ds.mesh_filename,
    young_modulus=4500.,
    poisson_ratio=0.49,
    bc=DirichletBoundaryCondition(ds.dirichlet_vertices),
    verbose=False
)

from dolfinx import fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from time import time

np.random.seed(0)

tao = elastic_model.tao
tao.solution.array[:] = 0.
# tao.solution.array[:] = np.random.rand(tao.solution.array.shape[0])

forces = np.zeros_like(elastic_model.rhs)
forces[:] = 0.01
start_time = time()
position = elastic_model.solve_direct(forces)
end_time = time()
print('Elapsed : ', end_time - start_time)
print('Reason : ', tao.reason)

displacement = position - elastic_model.initial_position
cons_violation = np.linalg.norm(displacement[bc.indices], np.inf)
print(f'Constraint violation : {cons_violation:10.2e}')

# Test Petsc Tao solver

# def objval(tao, u):
#     f = 0.5 * (u[0]**2 + 10. * u[1]**2 + 100. * u[2]**3)
#     nit = tao.getIterationNumber()
#     print(f'{nit:3d}  {f:14.9e}')
#     return f

# def objgrad(tao: PETSc.TAO, u: PETSc.Vec, g: PETSc.Vec):
#     g[0] = u[0]
#     g[1] = 10. * u[1]
#     g[2] = 100. * u[2]
#     nit = tao.getIterationNumber()
#     print(f'{nit:3d}  {"":14s}  {g.norm():14.9e}')
#     return None

# def objvalgrad(tao: PETSc.TAO, u: PETSc.Vec, g: PETSc.Vec):
#     f = objval(tao, u)
#     objgrad(tao, u, g)
#     nit = tao.getIterationNumber()
#     print(f'{nit:3d}  {f:14.9e}  {g.norm():14.9e}')
#     print("u_storage : ", u_storage)
#     print("u         : ", u)
#     print("g_storage : ", g_storage)
#     print("g         : ", g)
#     return f

# # def hessian(tao, u, H, P):
# #     print('Calling Hessian')
# #     return None

# # Prepare initial guess and gradient
# u_storage = PETSc.Vec().createSeq(3)
# u_storage[0] = 1.0
# u_storage[1] = 2.0
# u_storage[2] = 3.0
# g_storage = PETSc.Vec().createSeq(3)

# # Prepare bounds
# lower = PETSc.Vec().createSeq(3)
# upper = PETSc.Vec().createSeq(3)
# lower[:] = np.array([PETSc.NINFINITY, 1., PETSc.NINFINITY])
# upper[:] = np.array([PETSc.INFINITY, PETSc.INFINITY, PETSc.INFINITY])

# # Prepare Hessian matrix
# d = PETSc.Vec().createSeq(3)
# d[:] = [1., 10., 100.]
# H = PETSc.Mat().createDiagonal(d)
# # H.assemble()

# tao = PETSc.TAO()
# tao.create()
# tao.setType('lmvm')
# tao.setObjective(objval)
# tao.setGradient(objgrad, g_storage)
# # tao.setObjectiveGradient(objvalgrad, g_storage)
# # tao.setHessian(hessian, H)
# tao.setInitial(u_storage)
# # tao.setVariableBounds(lower, upper)
# tao.setFromOptions()
# tao.solve()
# print("Solution  : ", tao.solution[:])
# print("u_storage : ", u_storage[:])
# print("g_storage : ", g_storage[:])

# # Launch again
# u_storage[0] = 1.0
# u_storage[1] = 2.0
# u_storage[2] = 3.0
# tao.solve()
# print("Solution  : ", tao.solution[:])
# print("u_storage : ", u_storage[:])
# print("g_storage : ", g_storage[:])

# # s = tao.computeObjectiveGradient(u, g)

# # print(s)
# # print(type(s))
# # print(g[:])

# # Create optimization problem and solver
# obj = DataAttachment(ds.matching_triangles, ds.pointcloud[0])
# problem = AdjointControlProblem(obj, ds.forces_vertices, elastic_model)

# wrapper = ScipyWrapper(problem, save_folder=None, verbose=False)

# ###############################################################################
# # OPTIMIZATION PROCESS
# ###############################################################################
# # Load ground truth forces
# true_forces = np.load(
#     os.path.join(liver.DATA_FOLDER, 'sequences', 'neohookean', seq_name,
#                  'true-forces.npy')
# )

# # Initialize optimization
# x0 = np.zeros(ds.forces_vertices.shape[0] * 3, dtype=np.float64)
# wrapper.first_eval(x0)
# gnorm0 = np.linalg.norm(wrapper.pb.g.reshape(-1), np.inf)

# nruns = 50
# times = np.zeros(nruns)
# fcalls = np.zeros(nruns, dtype=int)
# errs = np.zeros(nruns)

# # Print header
# print(f"  time  nit  fcalls  Error (%)  Termination message")

# for i in range(nruns):
#     # Update point cloud
#     obj.update_point_cloud(ds.pointcloud[i])

#     # Run optimization process
#     start_time = time()
#     x_final, f_final, stats = fmin_l_bfgs_b(
#         wrapper.fgobj,
#         x0,
#         m=x0.shape[0],
#         factr=10.,
#         pgtol=5e-4 * gnorm0,
#         maxiter=1000
#     )
#     times[i] = time() - start_time
#     x0 = x_final

#     # Evaluate estimation error
#     estimated_force = x_final.reshape(-1, 3).sum(axis=0)
#     diff = norm(estimated_force - true_forces[i]) / norm(true_forces[i])
#     errs[i] = diff * 100
#     fcalls[i] = stats["funcalls"]
#     message = stats['task']
#     nit = stats['nit']

#     # Print log
#     print(f'{times[i]:6.4f}  {nit:3d}  {fcalls[i]:6d}  {errs[i]:6.2f} %  '
#           + message)

# print('-----------------------------------------------------------------')
# print(f'Average time : {times[1:].mean():.4f}')
# print(f'Average fcalls : {fcalls[1:].mean():.2f}')
# print(f'Average err : {errs[1:].mean():.3f}')
