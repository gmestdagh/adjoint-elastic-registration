"""Elastic model where the stiffnes matrix is computed using SOFA.

To use this module SOFA, SofaPython3 and Caribou should be installed.
"""

import numpy as np
from scipy.sparse.linalg import factorized

import Sofa.Simulation
import Sofa.Core
import SofaCaribou
import SofaRuntime

from .elasticity import DirichletBoundaryCondition


class LinearElasticModel:
    """Linear elastic model with constant stiffness matrix."""

    def __init__(self, mesh_filename: str, young_modulus: float,
                 poisson_ratio: float, bc: DirichletBoundaryCondition):

        # Import necessary plugins
        SofaRuntime.importPlugin('SofaComponentAll')
        SofaRuntime.importPlugin('SofaGeneralLoader')
        SofaRuntime.importPlugin('SofaGeneralLinearSolver')
        SofaRuntime.importPlugin('SofaPython3')

        # Main node and mesh loading
        self.root = Sofa.Core.Node()
        self.root.addObject(
            'MeshGmshLoader',
            name='meshLoaderCoarse',
            filename=mesh_filename
        )

        # Mechanical node
        self.meca = self.root.addChild('meca')
        self.meca.addObject('TetrahedronSetTopologyContainer',
                            name='topo',
                            src='@../meshLoaderCoarse')
        self.meca.addObject('TetrahedronSetGeometryAlgorithms',
                            template='Vec3d')

        # CG Solver to assemble matrix
        self.CG = self.meca.addObject('ConjugateGradientSolver')

        # Mechanical object to manage mesh position
        self.mo = self.meca.addObject('MechanicalObject',
                                      template='Vec3d',
                                      showObject='1',
                                      showObjectScale='3')

        # Elastic model
        self.meca.addObject('TetrahedronElasticForce',
                            topology_container='@topo',
                            youngModulus=young_modulus,
                            poissonRatio=poisson_ratio,
                            corotated=False)

        # Init simulation
        Sofa.Simulation.init(self.root)

        # Prepare stiffness matrix
        self.initial_position = np.copy(self.mo.position.value).reshape(-1)
        self.CG.assemble(0., 0., -1)
        self.stiffness_matrix = self.CG.A().tocsc()
        bc.add_penalization_to_matrix(self.stiffness_matrix)
        self.solve_system = factorized(self.stiffness_matrix)

    def solve_adjoint(self, position: np.ndarray, rhs: np.ndarray):
        """Solve the adjoint system, return a nx3 vector and a flag.

        :param position Position where the adjoint system is evaluated
        :param rhs Right-hand side of the system (nx3 vector)
        """
        return self.solve_system(rhs)

    def solve_direct(self, forces: np.ndarray):
        """Compute displacement and return positions."""
        sol = self.solve_system(forces)
        return sol + self.initial_position
