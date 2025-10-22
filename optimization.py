import torch
import kaolin as kal
import trimesh

from mesh import Mesh
from camera import Camera
from render import MeshRenderer
from tqdm import tqdm
import loss

device="cuda"

# 1. Load mesh
reference_mesh_path = "./data/bunny.obj"
ref_mesh = Mesh(reference_mesh_path).mesh

target_mesh_path = "./data/sphere.obj"
init_mesh = Mesh(target_mesh_path).mesh

# 2. Set camera
camera = Camera()
cameras = camera.get_random_camera_batch(8)

# 3. Set render
mesh_render = MeshRenderer()

# 4. Set optimization
learning_rate = 1e-3
parameter = torch.nn.Parameter(init_mesh.vertices.clone().detach(), requires_grad=True)
def lr_schedule(iter):
    return max(0.0, 10 ** (-(iter) * 0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    
optimizer = torch.optim.Adam([parameter], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x)) 

# 5. Run optimization
iter = 1000
batch = 8
train_res =  (392,518)
for it in tqdm(range(iter)): 
    optimizer.zero_grad()
    cameras = camera.get_random_camera_batch(batch, iter_res=train_res, device=device)

    # render gt mesh at sampled views
    ref = mesh_render.render_batch(ref_mesh, cameras, train_res)

    optimized_vertices = parameter
    optimized_mesh = kal.rep.SurfaceMesh(optimized_vertices, init_mesh.faces)
    buffers = mesh_render.render_batch(optimized_mesh, cameras, train_res)

    # loss compute
    mask_loss = loss.mask_loss(buffers["mask"], ref["mask"])
    depth_loss = loss.depth_loss(buffers["depth"], ref["depth"], ref["mask"]) * 10

    total_loss = mask_loss + depth_loss #+ reg_loss + smooth_loss
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    if (it + 1) % 20 == 0: # save intermediate results every 100 iters
        with torch.no_grad():
            print(f"total_loss: {total_loss}, mask_loss: {mask_loss}, depth_loss: {depth_loss}")

# 6. Save results
mesh_np = trimesh.Trimesh(vertices = optimized_mesh.vertices.detach().cpu().numpy(), faces=optimized_mesh.faces.detach().cpu().numpy(), process=False)
mesh_np.export(f'./results/output_mesh.obj')
print('result saved!')

