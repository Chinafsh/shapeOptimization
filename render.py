import torch
import nvdiffrast.torch as dr
import kaolin as kal
import torch.nn.functional as F




def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret

class MeshRenderer:
    """
    Renderer for the Mesh representation.

    Args:
        rendering_options (dict): Rendering options.
        glctx (nvdiffrast.torch.RasterizeGLContext): RasterizeGLContext object for CUDA/OpenGL interop.
    """
    def __init__(self, near=1.0, far = 1000.0 , ssaa=1, device='cuda'):
        self.glctx = dr.RasterizeCudaContext(device=device)
        # self.glctx = dr.RasterizeGLContext(device=device)
        self.near = near
        self.far = far
        self.ssaa = ssaa
        self.device=device

    def render(self, 
               mesh, 
               extrinsics: torch.Tensor, 
               intrinsics: torch.Tensor, 
               resolution: tuple, 
               return_types=["mask", "normal", "depth"]):
        """
        Render the mesh.

        Args:
            mesh : mesh model
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            return_types (list): list of return types, can be "mask", "depth", "normal_map", "normal", "color"

        Returns:
            edict based on return_types containing:
                color (torch.Tensor): [3, H, W] rendered color image
                depth (torch.Tensor): [H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] rendered normal image
                normal_map (torch.Tensor): [3, H, W] rendered normal map image
                mask (torch.Tensor): [H, W] rendered mask image
        """
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32, device=self.device)
            ret_dict = {k : default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types}
            return ret_dict


        perspective = intrinsics_to_projection(intrinsics, self.near, self.far)

        RT = extrinsics.unsqueeze(0)
        full_proj = (perspective @ extrinsics).unsqueeze(0)

        vertices = mesh.vertices.unsqueeze(0)

        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))

        faces_int = mesh.faces.int()

        rast, _ = dr.rasterize(
            self.glctx, vertices_clip, faces_int, (resolution * self.ssaa))

        out_dict = {}
        for type in return_types:
            img = None
            if type == "mask" :
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif type == "depth":
                img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "normal" :
                img = dr.interpolate(
                    mesh.face_normal.reshape(1, -1, 3), rast,
                    torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3)
                )[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
                # normalize norm pictures
                img = (img + 1) / 2
            elif type == "normal_map" :
                img = dr.interpolate(mesh.vertex_attrs[:, 3:].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "color" :
                img = dr.interpolate(mesh.vertex_attrs[:, :3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)

            if self.ssaa > 1:
                img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                img = img.squeeze()
            else:
                img = img.permute(0, 3, 1, 2).squeeze()
            out_dict[type] = img
        return out_dict
    
    def render_batch(self, 
                     mesh, 
                     camera, 
                     resolution: tuple, 
                     return_types=["mask", "normal", "depth"]):
        """
        Returns:
            edict based on return_types containing:
                color (torch.Tensor): [B, H, W, 3] rendered color image
                depth (torch.Tensor): [B, H, W, 1] rendered depth image
                normal (torch.Tensor): [B, H, W, 3] rendered normal image
                normal_map (torch.Tensor): [B, H, W, 3] rendered normal map image
                mask (torch.Tensor): [B, H, W, 1] rendered mask image
        """
        
        vertices_camera = camera.extrinsics.transform(mesh.vertices)
        # Projection: nvdiffrast take clip coordinates as input to apply barycentric perspective correction.
        # Using `camera.intrinsics.transform(vertices_camera) would return the normalized device coordinates.
        proj = camera.projection_matrix().unsqueeze(1)
        proj[:, :, 1, 1] = -proj[:, :, 1, 1]
        homogeneous_vecs = kal.render.camera.up_to_homogeneous(
            vertices_camera
        )
        vertices_clip = (proj @ homogeneous_vecs.unsqueeze(-1)).squeeze(-1)
        faces_int = mesh.faces.int()

        rast, _ = dr.rasterize(
            self.glctx, vertices_clip, faces_int, resolution)

        out_dict = {}
        for type in return_types:
            if type == "mask" :
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif type == "depth":
                img = dr.interpolate(homogeneous_vecs, rast, faces_int)[0]
            elif type == "normals" :
                img = dr.interpolate(
                    mesh.face_normals.reshape(len(mesh), -1, 3), rast,
                    torch.arange(mesh.faces.shape[0] * 3, device='cuda', dtype=torch.int).reshape(-1, 3)
                )[0]
            out_dict[type] = img
        return out_dict
        