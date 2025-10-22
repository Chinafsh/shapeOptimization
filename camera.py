import torch
import kaolin as kal
import math
import numpy as np
import util

class Camera:

    def __init__(self, extrinsics=None, intrinsics=None, device="cuda"):
        self.device = device

    def get_random_camera_batch(self,
                                batch_size, 
                                fovy = np.deg2rad(45), 
                                iter_res=[512,512], 
                                cam_near_far=[0.1, 1000.0], 
                                cam_radius=3.0, 
                                device="cuda", 
                                use_kaolin=True):
        if use_kaolin:
            camera_pos = torch.stack(kal.ops.coords.spherical2cartesian(
                *kal.ops.random.sample_spherical_coords((batch_size,), azimuth_low=0., azimuth_high=math.pi * 2,
                                                        elevation_low=-math.pi / 2., elevation_high=math.pi / 2., device='cuda'),
                cam_radius
            ), dim=-1)
            return kal.render.camera.Camera.from_args(
                eye=camera_pos + torch.rand((batch_size, 1), device='cuda') * 0.5 - 0.25,
                at=torch.zeros(batch_size, 3),
                up=torch.tensor([[0., 1., 0.]]),
                fov=fovy,
                near=cam_near_far[0], far=cam_near_far[1],
                height=iter_res[0], width=iter_res[1],
                device='cuda'
            )
        else:
            def get_random_camera():
                proj_mtx = util.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])
                mv     = util.translate(0, 0, -cam_radius) @ util.random_rotation_translation(0.25)
                mvp    = proj_mtx @ mv
                return mv, mvp
            mv_batch = []
            mvp_batch = []
            for i in range(batch_size):
                mv, mvp = get_random_camera()
                mv_batch.append(mv)
                mvp_batch.append(mvp)
            return torch.stack(mv_batch).to(device), torch.stack(mvp_batch).to(device)
        

    def get_rotate_camera(self, itr, fovy = np.deg2rad(45), iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=3.0, device="cuda", use_kaolin=True):
        if use_kaolin:
            ang = (itr / 10) * np.pi * 2
            camera_pos = torch.stack(kal.ops.coords.spherical2cartesian(torch.tensor(ang), torch.tensor(0.4), -torch.tensor(cam_radius)))
            return kal.render.camera.Camera.from_args(
                eye=camera_pos,
                at=torch.zeros(3),
                up=torch.tensor([0., 1., 0.]),
                fov=fovy,
                near=cam_near_far[0], far=cam_near_far[1],
                height=iter_res[0], width=iter_res[1],
                device='cuda'
            )
        else:
            proj_mtx = util.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])

            # Smooth rotation for display.
            ang    = (itr / 10) * np.pi * 2
            mv     = util.translate(0, 0, -cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
            mvp    = proj_mtx @ mv
            return mv.to(device), mvp.to(device)
        


