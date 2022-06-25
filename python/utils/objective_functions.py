"""Data attachment and penalty terms that compose the objective function"""

import numpy as np
from trimesh import Trimesh
from trimesh.triangles import points_to_barycentric


class DataAttachment:
    """Measures discrepancy between a deformation and observed data.

    The function evaluates the average square distance between points
    from the point cloud.
    Points are projected onto triangles and the resulting gradient is 
    distributed over triangle vertices.
    """
    def __init__(self, matching_triangles, point_cloud):
        """Create a instance associated to a point cloud.
        
        :param matching_triangles: Array of shape (ntri, 3) containing
            vertex indices for each triangle concerned by the projection
        :param point_cloud: Array of shape (npoints, 3) containing the
            points coordinates
        """
        self.matching_triangles = matching_triangles
        self.point_cloud = point_cloud

    def update_point_cloud(self, point_cloud):
        """Load new point cloud and update normalizing factor."""
        self.point_cloud = point_cloud

    def fobj(self, position: np.ndarray):
        """Evaluate objective value.

        :param position: Vector of mesh vertices positions
        """
        surface = Trimesh(position.reshape(-1, 3), self.matching_triangles)
        _, distances, _ = surface.nearest.on_surface(self.point_cloud)
        return 0.5 * (distances ** 2).mean()

    def fgobj(self, position: np.ndarray):
        """Evaluate objective value and gradient.

        :param position: Vector of mesh vertices positions
        """
        surface = Trimesh(position.reshape(-1, 3), self.matching_triangles)
        closest, distances, tri_idx = \
            surface.nearest.on_surface(self.point_cloud)
        f = 0.5 * (distances ** 2).mean()

        # Compute barycentric coordinates of projection points
        barycentric = \
            points_to_barycentric(surface.triangles[tri_idx], closest) \
            .reshape(-1, 3, 1)
        projections = (closest - self.point_cloud) / self.point_cloud.shape[0]

        # Objective gradient
        g = np.zeros_like(position).reshape(-1, 3)
        for (triangle, proj, bary) in zip(self.matching_triangles[tri_idx], 
                                          projections, barycentric):
            # g[triangle] is a 3x3 array
            g[triangle] += bary * proj

        return f, g.reshape(-1)


class L2Penalty:
    """A penalty term based on the l2 norm.

    This penalty represents a function of type

            R(forces) = alpha / 2 * || forces ||^2,

    where ||.|| is the Euclidean norm in R^n.
    """
    def __init__(self, penalty_coefficient, forces_vertices):
        """Initialize penalty term.

        :param penalty_coefficient: Weight of the penalty
        :param forces_vertices: Vector containing indices of vertices 
            where forces apply
        """
        self.penalty_coefficient = penalty_coefficient
        self.forces_vertices = np.unique(forces_vertices)
        self.penalty_coefficient /= np.size(self.forces_vertices)

    def update_coefficient(self, coefficient):
        """Set a new penalty coefficient."""
        self.penalty_coefficient = coefficient / np.size(self.forces_vertices)

    def fobj(self, forces):
        """Evaluate objective value.

        :param forces: Vector representing nodal forces applied on
            force vertices
        """
        return 0.5 * self.penalty_coefficient * (forces ** 2).sum()

    def fgobj(self, forces):
        """Evaluate objective value and gradient.

        :param forces: Vector representing nodal forces applied on
            force vertices
        """
        g = self.penalty_coefficient * forces
        f = 0.5 * self.penalty_coefficient * (forces ** 2).sum()
        return f, g
