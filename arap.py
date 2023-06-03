import time
import os
import sys
from pathlib import Path
from copy import deepcopy
from logger import Logger

import numpy as np
from scipy.sparse import coo_matrix, linalg
import open3d as o3d


pyexample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Open3D/examples/python')
sys.path.append(pyexample_path)
import open3d_example as o3dex
from geometry.triangle_mesh_deformation import (
    problem0 as pick,
    problem1 as plain,
    problem2 as armadillo
)


class MeshNp:
    def __init__(self, mesh):
        self.mesh_o3d = deepcopy(mesh)
        self.vertices = np.array(deepcopy(mesh.vertices))
        self.triangles = np.array(deepcopy(mesh.triangles))

    def set_v_t(self):
        self.mesh_o3d.vertices = o3d.utility.Vector3dVector(self.vertices)
        self.mesh_o3d.triangles = o3d.utility.Vector3iVector(self.triangles)
        self.mesh_o3d.compute_vertex_normals()

    def get_o3d(self):
        self.set_v_t()
        mesh = deepcopy(self.mesh_o3d)
        return mesh
    
    def _get_adjacency_list(self):
        n_vert = len(self.vertices)
        adj_list = [[] for _ in range(n_vert)]
        for triangle in Logger.tqdm_debug(self.triangles, desc='Generating adjacency list'):
            i, j, k = triangle
            adj_list[i].extend([j, k])
            adj_list[j].extend([i, k])
            adj_list[k].extend([i, j])
        adj_list = [set(adj) for adj in adj_list]
        return adj_list
    
    @staticmethod
    def _get_ordered_edge(vidx0, vidx1):
            return min(vidx0, vidx1), max(vidx0, vidx1)

    def _get_edge_to_vert_map(self):
        def add_edge(edge2verts, edge, vert):
            if edge in edge2verts:
                edge2verts[edge].append(vert)
            else:
                edge2verts[edge] = [vert]

        edge2verts = dict()
        for v1, v2, v3 in Logger.tqdm_debug(self.triangles, desc='Generating edge to vert map'):
            add_edge(edge2verts, MeshNp._get_ordered_edge(v1,v2), v3)
            add_edge(edge2verts, MeshNp._get_ordered_edge(v2,v3), v1)
            add_edge(edge2verts, MeshNp._get_ordered_edge(v3,v1), v2)
        return edge2verts

    def _get_edge_weights(self, edge_to_verts: dict, min_weight=0):
        ''' return cotangent weights for each edge '''
        weights = dict()
        for edge, verts0 in Logger.tqdm_debug(edge_to_verts.items(), total=len(edge_to_verts), desc='Generating edge weights'):
            weight_sum = 0.
            if verts0:
                for v_id in verts0:
                    a = self.vertices[edge[0]] - self.vertices[v_id]
                    b = self.vertices[edge[1]] - self.vertices[v_id]
                    weight = a @ b / np.linalg.norm(np.cross(a, b))
                    weight_sum += weight
                weight_sum /= len(verts0)
                weight_sum = max(weight_sum, min_weight)
            weights[edge] = weight_sum
        return weights
    
    def _get_sparse_solver(self, constraints, adjacency_list, edge_weights):

        triplets = []
        n_verts = len(self.vertices)
        for i in Logger.tqdm_debug(range(n_verts), total=n_verts, desc='Generating constraint triplets'):
            if i in constraints:
                triplets.append((i, i, 1))
            else:
                W = 0
                for j in adjacency_list[i]:
                    w = edge_weights[MeshNp._get_ordered_edge(i, j)]
                    triplets.append((i, j, -w))
                    W += w
                if W > 0:
                    triplets.append((i, i, W))

        triplet_rows, triplet_cols, triplet_vals = zip(*triplets)
        L = coo_matrix((triplet_vals, (triplet_rows, triplet_cols)), shape=(n_verts, n_verts))
        L = L.tocsc()

        solver = linalg.splu(L)
        Logger.log_debug("[DeformAsRigidAsPossible] done setting up sparse solver")
        return solver
    
    def _get_surface_areas(self):
        def get_area(triangle, verts):
            tr_verts = [verts[v_id] for v_id in triangle]
            a = tr_verts[1] - tr_verts[0]
            b = tr_verts[2] - tr_verts[0]
            area = 0.5 * np.linalg.norm(np.cross(a, b))
            return area

        triangle_areas = np.zeros([len(self.triangles)], dtype=float)
        for i, triangle in Logger.tqdm_debug(enumerate(self.triangles), total=len(self.triangles), desc='Caculating triangles area'):
            triangle_area = get_area(triangle, self.vertices)
            triangle_areas[i] = triangle_area
        return triangle_areas.sum(), triangle_areas

    def deform_arap(
        self,
        constraint_ids: np.array,
        constraint_pos: list[np.array],
        max_iter: int = 50,
        energy_model: str = 'Spokes',
        smoothed_alpha:float = 0.01,
        save_steps_dir: str = ''
    ) -> o3d.geometry.TriangleMesh:
        Logger.set_prefisx('ARAP')
        tic = time.time()

        if save_steps_dir:
            Logger.log_debug(f'Saving intermediate steps into {save_steps_dir}')
            save_steps_dir = Path(save_steps_dir)
            save_steps_dir.mkdir(parents=True, exist_ok=True)
        
        prime = self.get_o3d()
        prime = MeshNp(prime)

        n_verts = len(self.vertices)
        adjacency_list = prime._get_adjacency_list()
        edge_to_verts = prime._get_edge_to_vert_map()
        edge_weights = prime._get_edge_weights(edge_to_verts)

        constraints = dict()
        for c_id, pos in zip(constraint_ids, constraint_pos):
            constraints[c_id] = pos
        
        solver = prime._get_sparse_solver(constraints, adjacency_list, edge_weights)

        b = np.zeros([3, n_verts])

        surface_area = -1.
        Rs = np.zeros([n_verts, 3, 3], dtype=float)
        if energy_model == 'Smoothed':
            Rs_old = np.zeros([n_verts], dtype=float)
            surface_area, triangle_areas = prime._get_surface_areas()

        # Update rotations
        # http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
        for iter in range(max_iter):
            tic_step = time.time()
            if energy_model == 'Smoothed':
                Rs, Rs_old = Rs_old, Rs

            for i in range(n_verts):
                S = np.zeros((3,3), dtype=float)
                R = np.zeros((3,3), dtype=float)
                n_nbs = 0
                for j in adjacency_list[i]:
                    e0 = self.vertices[i] - self.vertices[j]
                    e1 = prime.vertices[i] - prime.vertices[j]
                    w = edge_weights[MeshNp._get_ordered_edge(i, j)]
                    S += w * np.outer(e0, e1)
                    if energy_model == 'Smoothed':
                        R += Rs_old[j]
                    n_nbs += 1
                if energy_model == 'Smoothed' and iter > 0 and n_nbs > 0:
                    S = 2*S + (4 * smoothed_alpha * surface_area / n_nbs) * R.T
                
                U, D, Vt = np.linalg.svd(S)
                D = np.array([1, 1, np.linalg.det(Vt.T @ U.T)])
                Rs[i] = Vt.T @ np.diag(D) @ U.T

                if np.linalg.det(Rs[i]) <= 0:
                    Logger.log_warning("something went wrong with updating R")
            
            # Update Positions
            for i in range(n_verts):
                bi = np.zeros((3), dtype=float)
                if i in constraints:
                    bi = constraints[i]
                else:
                    for j in adjacency_list[i]:
                        w = edge_weights[MeshNp._get_ordered_edge(i, j)]
                        bi += w/2 * ((Rs[i] + Rs[j]) @ (self.vertices[i] - self.vertices[j]))
                b[:,i] = bi

            for comp in range(3):
                p_prime = solver.solve(b[comp])
                prime.vertices[:,comp] = p_prime
            
            # Compute energy and log
            energy = 0.0
            reg = 0.0
            for i in range(n_verts):
                for j in adjacency_list[i]:
                    w = edge_weights[MeshNp._get_ordered_edge(i, j)]
                    e0 = self.vertices[i] - self.vertices[j]
                    e1 = prime.vertices[i] - prime.vertices[j]
                    diff = e1 - Rs[i] * e0
                    energy += w * np.linalg.norm(diff) ** 2
                    
                    if energy_model == 'Smoothed':
                        reg += np.linalg.norm(Rs[i] - Rs[j]) ** 2
            if energy_model == 'Smoothed':
                energy += smoothed_alpha * surface_area * reg
            
            # save intermeadiate steps
            if save_steps_dir:
                mehs_path = save_steps_dir / f'iter_{iter:03d}.ply'
                mehs_path = str(mehs_path.absolute())
                mash_now = prime.get_o3d()
                o3d.io.write_triangle_mesh(mehs_path, mash_now)
                s = f', saved to {mehs_path}'
            else:
                s = ''
            
            Logger.log_debug(f'iter={iter}, energy={energy:e}, time={time.time() - tic_step:.1f}s{s}')

        Logger.log_debug(f'deform took {time.time() - tic:.1f}[s]')
        Logger.reset_prefisx()
        return prime.get_o3d()


def o3d_deform_as_rigid_as_possible(mesh, constraint_ids: np.array, constraint_pos: np.array):
    constraint_ids = o3d.utility.IntVector(constraint_ids)
    constraint_pos = o3d.utility.Vector3dVector(constraint_pos)
    tic = time.time()
    mesh_prime = mesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter=50)
    print(f'deform took {time.time() - tic:.2f}[s]')

    return mesh_prime

def show_transformation(mesh, mesh_prime, constraint_pos):
    mesh_prime.compute_vertex_normals()
    mesh.paint_uniform_color((1, 0, 0))
    handles = o3d.geometry.PointCloud()
    handles.points = o3d.utility.Vector3dVector(constraint_pos)
    handles.paint_uniform_color((0, 1, 0))
    o3d.visualization.draw_geometries([mesh, mesh_prime, handles])


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.Debug)
    Logger.set_log_level(Logger.Mode.DEBUG)

    for mesh, constraint_ids, constraint_pos in [
            # pick(),
            # plain(),
            armadillo()
    ]:
        constraint_ids = np.array(constraint_ids, dtype=np.int32)

        # mesh_np = MeshNp(mesh)
        # mesh_prime_my = mesh_np.deform_arap(constraint_ids, constraint_pos, save_steps_dir='data/pick')
        # show_transformation(mesh, mesh_prime_my, constraint_pos)
        mesh_prime_o3d = o3d_deform_as_rigid_as_possible(mesh, constraint_ids, constraint_pos)
        show_transformation(mesh, mesh_prime_o3d, constraint_pos)

    o3d.utility.set_verbosity_level(o3d.utility.Info)
