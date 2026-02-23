"""Mesh loading and property computation utilities."""

from __future__ import annotations

import warnings
import numpy as np
import open3d as o3d
from numpy.typing import NDArray

_EPS = 1.0e-12


class Mesh:
    """
    Mesh initializer using Open3D.

    Loads a mesh from file or accepts an existing Open3D TriangleMesh, computes
    normals, basic geometric properties, and provides helpers for volume, center
    of mass, and inertia tensor evaluation.

    Notes
    -----
    - The loader assumes the provided mesh is already centered at its COM in the
      material frame. Geometry is left untouched; computed properties reflect the
      input coordinates.
    - When passing a raw Open3D TriangleMesh directly, ensure it is already
      COM-centered and that its normals are valid.
    """

    def __init__(
        self,
        mesh_or_path: str | o3d.geometry.TriangleMesh,
        warn_if_not_watertight: bool = True,
    ) -> None:
        if isinstance(mesh_or_path, str):
            self.mesh = o3d.io.read_triangle_mesh(mesh_or_path)
        else:
            self.mesh = mesh_or_path

        if self.mesh.is_empty():
            raise ValueError("Loaded mesh is empty.")

        # Clean up mesh geometry.
        self.mesh.merge_close_vertices(1e-6)
        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_unreferenced_vertices()

        self.mesh.compute_vertex_normals()
        self.mesh.compute_triangle_normals()

        self.is_watertight = bool(self.mesh.is_watertight())
        if warn_if_not_watertight and not self.is_watertight:
            warnings.warn(
                "Mesh is not watertight; using oriented bounding box fallback for "
                "volume, center of mass, and inertia tensor.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.vertices = np.asarray(self.mesh.vertices, dtype=np.float64)
        self.triangles = np.asarray(self.mesh.triangles, dtype=np.int32)
        self.triangle_normals = np.asarray(self.mesh.triangle_normals, dtype=np.float64)
        self.n_triangles = int(self.triangles.shape[0])
        self.obb = self.mesh.get_oriented_bounding_box()

    def compute_volume(self) -> float:
        """
        Compute mesh volume using the divergence theorem. Falls back to oriented
        bounding box volume for non-watertight or degenerate meshes.
        """
        if self.is_watertight:
            v0 = self.vertices[self.triangles[:, 0]]
            v1 = self.vertices[self.triangles[:, 1]]
            v2 = self.vertices[self.triangles[:, 2]]
            signed = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
            vol = signed.sum() / 6.0
            if abs(vol) > _EPS:
                return float(abs(vol))
        extent = np.asarray(self.obb.extent, dtype=np.float64)
        return float(np.prod(extent))

    def compute_center_of_mass(self, density: float = 1.0) -> NDArray[np.float64]:
        """
        Compute COM assuming uniform density. Uses watertight tetrahedralization
        when available; otherwise falls back to oriented bounding box center.
        """
        if self.is_watertight:
            v0 = self.vertices[self.triangles[:, 0]]
            v1 = self.vertices[self.triangles[:, 1]]
            v2 = self.vertices[self.triangles[:, 2]]
            cross = np.cross(v1, v2)
            signed_vol = np.einsum("ij,ij->i", v0, cross) / 6.0
            total_signed_vol = signed_vol.sum()
            if abs(total_signed_vol) > _EPS:
                com_num = (signed_vol[:, None] * (v0 + v1 + v2)).sum(axis=0)
                com = com_num / (4.0 * total_signed_vol)
                return com.astype(np.float64)
        return np.asarray(self.obb.center, dtype=np.float64)

    def compute_inertia_tensor(
        self,
        density: float = 1.0,
        com: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute 3x3 inertia tensor about the provided COM. If COM is not
        specified, the origin is assumed to be the COM.

        Falls back to oriented bounding box inertia if mesh is non-watertight
        or degenerate.
        """
        if com is None:
            com_target = np.zeros(3, dtype=np.float64)
        else:
            com_target = np.asarray(com, dtype=np.float64)
        if self.is_watertight:
            v0 = self.vertices[self.triangles[:, 0]]
            v1 = self.vertices[self.triangles[:, 1]]
            v2 = self.vertices[self.triangles[:, 2]]
            cross = np.cross(v1, v2)
            signed_vol = np.einsum("ij,ij->i", v0, cross) / 6.0
            total_signed_vol = signed_vol.sum()
            volume = abs(total_signed_vol)
            if volume > _EPS:
                sign = 1.0 if total_signed_vol >= 0.0 else -1.0
                # second moment matrix S = ∫ r r^T dV over the mesh
                outer_v0 = np.einsum("ni,nj->nij", v0, v0)
                outer_v1 = np.einsum("ni,nj->nij", v1, v1)
                outer_v2 = np.einsum("ni,nj->nij", v2, v2)
                outer_v0_v1 = np.einsum("ni,nj->nij", v0, v1) + np.einsum(
                    "ni,nj->nij", v1, v0
                )
                outer_v0_v2 = np.einsum("ni,nj->nij", v0, v2) + np.einsum(
                    "ni,nj->nij", v2, v0
                )
                outer_v1_v2 = np.einsum("ni,nj->nij", v1, v2) + np.einsum(
                    "ni,nj->nij", v2, v1
                )

                S = signed_vol[:, None, None] * (
                    (outer_v0 + outer_v1 + outer_v2) / 10.0
                    + (outer_v0_v1 + outer_v0_v2 + outer_v1_v2) / 20.0
                )
                S_sum = S.sum(axis=0) * sign
                I_origin = density * (
                    np.trace(S_sum) * np.eye(3, dtype=np.float64) - S_sum
                )
                mass = density * volume
                com_vec = np.asarray(com_target, dtype=np.float64)
                shift = mass * (
                    np.dot(com_vec, com_vec) * np.eye(3, dtype=np.float64)
                    - np.outer(com_vec, com_vec)
                )
                return I_origin - shift

        # Oriented bounding box fallback
        extent = np.asarray(self.obb.extent, dtype=np.float64)
        mass = density * float(np.prod(extent))
        I_local = (
            mass
            / 12.0
            * np.diag(
                [
                    extent[1] ** 2 + extent[2] ** 2,
                    extent[0] ** 2 + extent[2] ** 2,
                    extent[0] ** 2 + extent[1] ** 2,
                ]
            )
        )
        R = self._obb_rotation()
        I_world = R @ I_local @ R.T
        com_world = np.asarray(self.obb.center, dtype=np.float64)
        com_vec = com_target
        delta = com_vec - com_world
        shift = mass * (
            np.dot(delta, delta) * np.eye(3, dtype=np.float64) - np.outer(delta, delta)
        )
        return I_world - shift

    def compute_bounding_box(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return axis-aligned bounding box (min_bound, max_bound)."""
        return (
            np.asarray(self.mesh.get_min_bound(), dtype=np.float64),
            np.asarray(self.mesh.get_max_bound(), dtype=np.float64),
        )

    def _obb_rotation(self) -> NDArray[np.float64]:
        """Return the rotation matrix of the oriented bounding box."""
        if hasattr(self.obb, "R"):
            R = np.asarray(self.obb.R, dtype=np.float64)
        else:
            try:
                R = np.asarray(self.obb.get_rotation_matrix(), dtype=np.float64)
            except Exception:
                R = np.eye(3, dtype=np.float64)
        if R.shape != (3, 3):
            return np.eye(3, dtype=np.float64)
        return R
