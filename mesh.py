import torch
import trimesh
import kaolin as kal


class Mesh:
    def __init__(self, path, device = "cuda"):
        self.device = device
        self._load_mesh(path, device)
        self.mesh = kal.rep.SurfaceMesh(self.vertices, self.faces)


    def _load_mesh(self, path, device):
        mesh_np = trimesh.load(path)
        vertices = torch.tensor(mesh_np.vertices, device=device, dtype=torch.float)
        faces = torch.tensor(mesh_np.faces, device=device, dtype=torch.long)
        
        # Normalize
        vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
        scale = 1.8 / torch.max(vmax - vmin).item()
        vertices = vertices - (vmax + vmin) / 2 # Center mesh on origin
        vertices = vertices * scale # Rescale to [-0.9, 0.9]

        self.vertices = vertices
        self.faces = faces
    

    