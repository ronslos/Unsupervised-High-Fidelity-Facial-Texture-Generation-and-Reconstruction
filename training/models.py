import torch
import torch.nn as nn
import pickle
import numpy as np
from scipy.io import loadmat
from pytorch3d.structures import Meshes
import torchvision
import torch.nn.functional as F
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PerspectiveCameras,
    PointLights,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesUV,
    blending
)


class DFRModel(nn.Module):
    def __init__(self, layer_size=512 * 2):
        super(DFRModel, self).__init__()
        self.input_size = 18 * 512
        self.output_size = 257
        self.intermediate_size = layer_size

        self.L1 = nn.Linear(self.input_size, self.intermediate_size)
        self.L2 = nn.Linear(self.intermediate_size, self.intermediate_size)
        self.L3 = nn.Linear(self.intermediate_size, self.output_size)
        self.activation = nn.ELU()

    def forward(self, input):
        input = input.flatten(start_dim=1)
        out = self.activation(self.L1(input))
        out = self.activation(self.L2(out))
        output = self.L3(out)
        return output


class RigNet(nn.Module):
    def __init__(self):
        super(RigNet, self).__init__()
        self.encoder_input_size = 512
        self.encoder_output_size = 32
        self.input_dimension = 18
        self.params_size = 257
        self.decoder_output_size = 512

        self.encoder = nn.ModuleList(
            [nn.Linear(self.encoder_input_size, self.encoder_output_size) for i in range(self.input_dimension)])
        self.decoder = nn.ModuleList(
            [nn.Linear(self.encoder_output_size + self.params_size, self.decoder_output_size) for i in
             range(self.input_dimension)])

    def forward(self, w, p):
        # l = [i for i in range(self.input_dimension)]
        l = [self.encoder[i](w[:, i, :]) for i in range(self.input_dimension)]
        l_concat_p = torch.stack([torch.cat((l_i, p), dim=1) for l_i in l], dim=1)
        d = torch.stack([self.decoder[i](l_concat_p[:, i, :]) for i in range(self.input_dimension)], dim=1)
        return d


class ReconModelMP(nn.Module):
    def __init__(self, device, img_size=1024):
        super(ReconModelMP, self).__init__()
        self.device = device
        self.img_size = img_size
        self.renderer = self.get_renderer()
        model = loadmat('BFM/canonical_face_model.mat')
        self.tri = torch.from_numpy(model['tri']).to(device)
        # self.uv_map = torch.from_numpy(model['uv']).type(torch.float32).to(device)
        self.uv_map = torch.from_numpy(np.load('BFM/uv_map_mp_open_mouth.npy')).float().to(device)
        self.tri_uv = torch.from_numpy(model['tri_uv']).type(torch.float32).to(device)
        self.tri_mouth = torch.from_numpy(model['mouth_tri']).type(torch.float32).to(device)

    def get_renderer(self):
        R, T = look_at_view_transform(-1, 0, 0)
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, znear=0.01, zfar=10000, min_x=1, max_x=0,
                                         min_y=1,
                                         max_y=0)
        # lights = PointLights(device=self.device, location=[[0.0, 0.0, 1e5]], ambient_color=[[1, 1, 1]],
        #                      specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])
        lights = AmbientLights(device=self.device)
        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )
        return renderer

    def forward(self, landmarks):
        batch_num = landmarks.shape[0]
        landmarks = landmarks.type(torch.float32)
        # landmarks[:, :, 2] = 10-landmarks[:, :, 2]
        # tri = self.tri.repeat(batch_num, 1, 1).long()
        # tri_uv = self.tri_uv.repeat(batch_num, 1, 1).long()
        tri_mouth = self.tri_mouth.repeat(batch_num, 1, 1).long()
        # face_color = TexturesUV(maps=tex, faces_uvs=tri_uv,
        #                         verts_uvs=self.uv_map.repeat(batch_num, 1, 1))

        face_color_mouth = TexturesVertex(torch.ones_like(landmarks))
        # mesh = Meshes(landmarks, tri, face_color)
        mesh_mouth = Meshes(landmarks, tri_mouth, face_color_mouth)
        rendered_mouth = self.renderer(mesh_mouth)
        # rendered_img = self.renderer(mesh)
        # mask = rendered_img.permute(0, 3, 1, 2)[:, 3, :, :].detach() > 0
        mask_mouth = rendered_mouth.permute(0, 3, 1, 2)[:, 3, :, :].detach() > 0
        # mask[mask_mouth] = 0

        return mask_mouth


class ReconModelOrig(nn.Module):
    def __init__(self, face_model,
                 focal=[1015, 1015], img_size=[1024, 1024], device='cuda:0'):
        super(ReconModelOrig, self).__init__()
        self.facemodel = face_model

        self.focal = focal
        self.img_size = img_size

        self.device = torch.device(device)

        self.renderer = self.get_renderer(self.device)

        uv_map = nn.Parameter(torch.from_numpy(np.load('BFM/uv_map_closed_mouth.npy')).float(), requires_grad=False)
        # uv_map = nn.Parameter(torch.from_numpy(loadmat('BFM/uv_mp_to_ours.mat')['new_uv']).float(), requires_grad=False)
        self.register_parameter('uv_map', uv_map)
        self.kp_inds = torch.tensor(self.facemodel['keypoints'] - 1).squeeze().long()

        meanshape = nn.Parameter(torch.from_numpy(self.facemodel['meanshape'], ).float(), requires_grad=False)
        self.register_parameter('meanshape', meanshape)

        idBase = nn.Parameter(torch.from_numpy(self.facemodel['idBase']).float(), requires_grad=False)
        self.register_parameter('idBase', idBase)

        exBase = nn.Parameter(torch.from_numpy(self.facemodel['exBase']).float(), requires_grad=False)
        self.register_parameter('exBase', exBase)

        meantex = nn.Parameter(torch.from_numpy(self.facemodel['meantex']).float(), requires_grad=False)
        self.register_parameter('meantex', meantex)

        texBase = nn.Parameter(torch.from_numpy(self.facemodel['texBase']).float(), requires_grad=False)
        self.register_parameter('texBase', texBase)

        tri = nn.Parameter(torch.from_numpy(self.facemodel['tri']).float(), requires_grad=False)
        self.register_parameter('tri', tri)

        point_buf = nn.Parameter(torch.from_numpy(self.facemodel['point_buf']).float(), requires_grad=False)
        self.register_parameter('point_buf', point_buf)

    def similar_transform(self, pts3d, roi_box, size):
        pts3d_x = pts3d[:, :, 0]
        pts3d_y = pts3d[:, :, 1]
        pts3d_z = pts3d[:, :, 2]
        pts3d_y = size - pts3d_y + 1
        # pts3d_z -= 1
        pts3d_x += 1

        sx, sy, ex, ey = roi_box[:, 0], roi_box[:, 1], roi_box[:, 2], roi_box[:, 3]
        scale_x = 0.97 * (ex - sx) / size
        scale_y = 0.97 * (ey - sy) / size
        pts3d_x = pts3d_x * scale_x[:, None] + sx[:, None]
        pts3d_y = pts3d_y * scale_y[:, None] + sy[:, None]
        s = (scale_x[:, None] + scale_y[:, None]) / 2
        pts3d_z = pts3d_z * s
        pts3d_z = torch.min(pts3d_z, dim=1)[0][:, None] - pts3d_z
        pts3d = torch.cat((pts3d_x[:, :, None], pts3d_y[:, :, None], pts3d_z[:, :, None]), dim=2)
        return pts3d

    def get_renderer(self, device):
        R, T = look_at_view_transform(10, 0, 0)
        # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.001, zfar=1000,
        #                                 fov=2 * np.arctan(self.img_size // 2 / self.focal) * 180. / np.pi)
        self.cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=((self.focal[0], self.focal[1]),),
                                          principal_point=((self.img_size[0] / 2, self.img_size[1] / 2),),
                                          image_size=((self.img_size[0], self.img_size[1]),))
        lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]], ambient_color=[[1., 1., 1.]],
                             specular_color=[[0, 0, 0]], diffuse_color=[[0.0, 0.0, 0.0]])
        # lights = AmbientLights(device=device)
        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=self.cameras,
                lights=lights,
                blend_params=blend_params
            )
        )
        return renderer

    def Split_coeff(self, coeff):
        id_coeff = coeff[:, :80]  # identity(shape) coeff of dim 80
        ex_coeff = coeff[:, 80:144]  # expression coeff of dim 64
        tex_coeff = coeff[:, 144:224]  # texture(albedo) coeff of dim 80
        angles = coeff[:, 224:227]  # ruler angles(x,y,z) for rotation of dim 3
        gamma = coeff[:, 227:254]  # lighting coeff for 3 channel SH function of dim 27
        translation = coeff[:, 254:]  # translation coeff of dim 3

        return id_coeff, ex_coeff, tex_coeff, angles, gamma, translation

    def Shape_formation(self, id_coeff, ex_coeff):
        n_b = id_coeff.size(0)

        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
                     torch.einsum('ij,aj->ai', self.exBase, ex_coeff) + self.meanshape

        face_shape = face_shape.view(n_b, -1, 3)
        face_shape = face_shape - self.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)

        return face_shape

    def Texture_formation(self, tex_coeff):
        n_b = tex_coeff.size(0)
        face_texture = torch.einsum('ij,aj->ai', self.texBase, tex_coeff) + self.meantex

        face_texture = face_texture.view(n_b, -1, 3)
        return face_texture

    def Compute_norm(self, face_shape):

        face_id = self.tri.long() - 1
        point_id = self.point_buf.long() - 1
        v1 = face_shape[:, face_id[:, 0], :]
        v2 = face_shape[:, face_id[:, 1], :]
        v3 = face_shape[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
        vertex_norm = torch.sum(face_norm[:, point_id, :], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def Projection_block(self, face_shape):
        half_image_width = self.img_size // 2
        batchsize = face_shape.shape[0]
        camera_pos = torch.tensor([0.0, 0.0, 10.0], device=face_shape.device).reshape(1, 1, 3)
        # tensor.reshape(constant([0.0,0.0,10.0]),[1,1,3])
        p_matrix = np.array([self.focal, 0.0, half_image_width, \
                             0.0, self.focal, half_image_width, \
                             0.0, 0.0, 1.0], dtype=np.float32)

        p_matrix = np.tile(p_matrix.reshape(1, 3, 3), [batchsize, 1, 1])
        reverse_z = np.tile(np.reshape(np.array([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0], dtype=np.float32), [1, 3, 3]),
                            [batchsize, 1, 1])

        p_matrix = torch.tensor(p_matrix, device=face_shape.device)
        reverse_z = torch.tensor(reverse_z, device=face_shape.device)
        face_shape = torch.matmul(face_shape, reverse_z) + camera_pos
        aug_projection = torch.matmul(face_shape, p_matrix.permute((0, 2, 1)))

        face_projection = aug_projection[:, :, :2] / \
                          torch.reshape(aug_projection[:, :, 2], [batchsize, -1, 1])
        return face_projection

    @staticmethod
    def Compute_rotation_matrix(angles):
        n_b = angles.size(0)
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1, 1).view(3, n_b, 3, 3)

        if angles.is_cuda: rotXYZ = rotXYZ.to(angles.device)

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)

    @staticmethod
    def Rigid_transform_block(face_shape, rotation, translation):
        face_shape_r = torch.matmul(face_shape, rotation)
        face_shape_t = face_shape_r + translation.view(-1, 1, 3)

        return face_shape_t

    @staticmethod
    def Illumination_layer(face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        # v_num = face_texture.shape[1]
        a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        c = [1 / np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]
        init_lit = torch.Tensor([0.8, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([1, 1, -1]).type(torch.float32).to(
            face_norm.device)
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
            a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(face_norm.device),
            -a[1] * c[1] * face_norm[..., 1:2],
            a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
            a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2 - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        lighting = torch.cat([r, g, b], dim=-1)
        return lighting

    # @staticmethod
    # def Illumination_layer(norm, gamma):
    #
    #     n_b, num_vertex, _ = norm.size()
    #     n_v_full = n_b * num_vertex
    #     gamma = gamma.view(-1, 3, 9).clone()
    #     gamma[:, :, 0] += 0.8
    #
    #     gamma = gamma.permute(0, 2, 1)
    #
    #     a0 = np.pi
    #     a1 = 2 * np.pi / np.sqrt(3.0)
    #     a2 = 2 * np.pi / np.sqrt(8.0)
    #     c0 = 1 / np.sqrt(4 * np.pi)
    #     c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
    #     c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
    #     d0 = 0.5 / np.sqrt(3.0)
    #
    #     Y0 = torch.ones(n_v_full).to(gamma.device).float() * a0 * c0
    #     norm = norm.view(-1, 3)
    #     nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
    #     arrH = []
    #
    #     arrH.append(Y0)
    #     arrH.append(-a1 * c1 * ny)
    #     arrH.append(a1 * c1 * nz)
    #     arrH.append(-a1 * c1 * nx)
    #     arrH.append(a2 * c2 * nx * ny)
    #     arrH.append(-a2 * c2 * ny * nz)
    #     arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
    #     arrH.append(-a2 * c2 * nx * nz)
    #     arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))
    #
    #     H = torch.stack(arrH, 1)
    #     Y = H.view(n_b, num_vertex, 9)
    #     lighting = Y.bmm(gamma)
    #
    #     return lighting

    def get_lms(self, face_shape, kp_inds):
        lms = face_shape[:, kp_inds, :]
        return lms

    def inference(self, coeff, texture_map=None):
        batch_num = coeff.shape[0]

        id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = self.Split_coeff(coeff)

        face_shape = self.Shape_formation(id_coeff, ex_coeff)
        face_texture = self.Texture_formation(tex_coeff)
        rotation = self.Compute_rotation_matrix(angles)
        face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)
        # face_shape_t = self.similar_transform(face_shape_t, bboxs, 224)

        face_lms_t = self.get_lms(face_shape_t, self.kp_inds)
        # lms = self.Projection_block(face_lms_t)
        # lms = torch.stack([lms[:, :, 0], self.img_size - lms[:, :, 1]], dim=2)
        tri = self.tri - 1
        tri_rep = tri.repeat(batch_num, 1, 1)

        if texture_map is None:
            face_norm = self.Compute_norm(face_shape)
            face_norm_r = face_norm.bmm(rotation)
            face_color = self.Illumination_layer(face_texture, face_norm_r, gamma)
            face_color = TexturesVertex((face_color - 127.5) / 128)
        else:
            face_color = TexturesUV(maps=texture_map, faces_uvs=tri.repeat(batch_num, 1, 1).long(),
                                    verts_uvs=self.uv_map.repeat(batch_num, 1, 1))
            color = torch.Tensor([64, 224, 208]).to(self.device)
            face_color_white = TexturesVertex(torch.ones_like(face_shape_t) * color[None, None, :])

        mesh = Meshes(face_shape_t, tri_rep, face_color)
        mesh_white = Meshes(face_shape_t, tri_rep, face_color_white)
        # self.cameras.focal_length = 1015 * 1024 / 224
        self.cameras.focal_length[:, 0] = 0.98 * self.focal[0]
        self.cameras.focal_length[:, 1] = self.focal[1]
        rendered_img = self.renderer(mesh)
        self.renderer.shader.lights.diffuse_color = self.renderer.shader.lights.diffuse_color * 0 + 0.5
        self.renderer.shader.lights.ambient_color = self.renderer.shader.lights.diffuse_color * 0 + 0.1
        rendered_img_white = self.renderer(mesh_white)
        self.renderer.shader.lights.diffuse_color = self.renderer.shader.lights.diffuse_color * 0
        self.renderer.shader.lights.ambient_color = self.renderer.shader.lights.diffuse_color * 0 + 1
        mask = rendered_img[:, :, :, 3].detach() > 0
        return rendered_img, None, face_texture, mask, face_shape_t, tri, self.uv_map, rendered_img_white

    def forward(self, coeff, texture_map=None, bboxs=None, use_lighting=False):

        batch_num = coeff.shape[0]

        id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = self.Split_coeff(coeff)

        face_shape = self.Shape_formation(id_coeff, ex_coeff)
        face_texture = self.Texture_formation(tex_coeff)
        rotation = self.Compute_rotation_matrix(angles)
        face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)
        # face_shape_t = self.similar_transform(face_shape_t, bboxs, 224)

        face_lms_t = self.get_lms(face_shape_t, self.kp_inds)
        # lms = self.Projection_block(face_lms_t)
        # lms = torch.stack([lms[:, :, 0], self.img_size - lms[:, :, 1]], dim=2)
        tri = self.tri - 1
        tri_rep = tri.repeat(batch_num, 1, 1)
        if bboxs is not None:
            left, up, scale = bboxs[:, 0], bboxs[:, 1], 102. / bboxs[:, 2]
        if texture_map is None:
            face_norm = self.Compute_norm(face_shape)
            face_norm_r = face_norm.bmm(rotation)
            face_color = self.Illumination_layer(face_texture, face_norm_r, gamma)
            face_color = TexturesVertex((face_color - 127.5) / 128)
        else:
            face_color = TexturesUV(maps=texture_map, faces_uvs=tri.repeat(batch_num, 1, 1).long(),
                                    verts_uvs=self.uv_map.repeat(batch_num, 1, 1))
        if use_lighting:
            face_norm = self.Compute_norm(face_shape)
            face_norm_r = face_norm.bmm(rotation)
            lighting = self.Illumination_layer(face_norm_r, gamma)
            # lighting = (torch.mean(lighting, dim=2, keepdim=True)).repeat((1, 1, 3))
            lighting = TexturesVertex(lighting)
            mesh_lighting = Meshes(face_shape_t, tri_rep, lighting)
        mesh = Meshes(face_shape_t, tri_rep, face_color)
        scale = scale
        center_hor = left + 112 / scale
        center_ver = up + 112 / scale
        all_renders, all_lightings = [], []
        for i in range(face_shape_t.shape[0]):
            self.cameras.focal_length[:, 0] = 0.98 * self.focal[0] / scale[i]
            self.cameras.focal_length[:, 1] = self.focal[1] / scale[i]
            self.cameras.principal_point = ((center_hor[i], center_ver[i]),)
            all_renders.append(self.renderer(mesh[i]).permute(0, 3, 1, 2))
            if use_lighting:
                all_lightings.append(self.renderer(mesh_lighting[i]).permute(0, 3, 1, 2))
        rendered_img = torch.stack(all_renders).squeeze()
        mask = rendered_img[ 3, :, :].detach() > 0
        if use_lighting:
            rendered_lighting = torch.stack(all_lightings).squeeze()
            # rendered_img = (((rendered_img * 128 + 127.5) * rendered_lighting) - 127.5) / 128.0
            rendered_img = ((rendered_img / 2 + 0.5) * rendered_lighting) * 2 - 1
        return rendered_img, None, face_texture, coeff, mask  # , rendered_lighting.permute(0, 3, 1, 2)[:, :3, :, :]

    def forward_recon(self, coeff, texture_map=None, bboxs=None, use_lighting=False):

        batch_num = coeff.shape[0]

        id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = self.Split_coeff(coeff)

        face_shape = self.Shape_formation(id_coeff, ex_coeff)
        face_texture = self.Texture_formation(tex_coeff)
        rotation = self.Compute_rotation_matrix(angles)
        face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)
        # face_shape_t = self.similar_transform(face_shape_t, bboxs, 224)

        face_lms_t = self.get_lms(face_shape_t, self.kp_inds)
        # lms = self.Projection_block(face_lms_t)
        # lms = torch.stack([lms[:, :, 0], self.img_size - lms[:, :, 1]], dim=2)
        tri = self.tri - 1
        tri_rep = tri.repeat(batch_num, 1, 1)
        if bboxs is not None:
            left, up, scale = bboxs[:, 0], bboxs[:, 1], 102. / bboxs[:, 2]
        if texture_map is None:
            face_norm = self.Compute_norm(face_shape)
            face_norm_r = face_norm.bmm(rotation)
            face_color = self.Illumination_layer(face_texture, face_norm_r, gamma)
            face_color = TexturesVertex((face_color - 127.5) / 128)
        else:
            face_color = TexturesUV(maps=texture_map, faces_uvs=tri.repeat(batch_num, 1, 1).long(),
                                    verts_uvs=self.uv_map.repeat(batch_num, 1, 1))
        if use_lighting:
            face_norm = self.Compute_norm(face_shape)
            face_norm_r = face_norm.bmm(rotation)
            lighting = self.Illumination_layer(face_norm_r, gamma)
            # lighting = (torch.mean(lighting, dim=2, keepdim=True)).repeat((1, 1, 3))
            lighting = TexturesVertex(lighting)
            mesh_lighting = Meshes(face_shape_t, tri_rep, lighting)
            rendered_lighting = self.renderer(mesh_lighting).permute(0, 3, 1, 2)
        mesh = Meshes(face_shape_t, tri_rep, face_color)
        self.cameras.focal_length[:, 0] = 0.98 * self.focal[0]
        self.cameras.focal_length[:, 1] = self.focal[1]
        rendered_img = self.renderer(mesh).permute(0, 3, 1, 2)
        # rendered_img = rendered_img * rendered_lighting
        mask = rendered_img[:, 3, :, :].detach() > 0
        return rendered_img, None, face_texture, coeff, mask


class ReconModel(nn.Module):
    def __init__(self, face_model,
                 focal=1015 * 4, img_size=1024, device='cuda:0'):
        super(ReconModel, self).__init__()
        self.facemodel = face_model

        self.focal = focal
        self.img_size = img_size

        self.device = torch.device(device)

        self.renderer = self.get_renderer(self.device)
        self.lighting_renderer = self.get_renderer(self.device, lighting=False)

        uv_map = nn.Parameter(torch.from_numpy(np.load('BFM/uv_map_closed_mouth.npy')).float(), requires_grad=False)
        exp_base = loadmat('BFM/w_exp')['converted_exp']
        self.register_parameter('uv_map', uv_map)
        self.kp_inds = torch.tensor(self.facemodel['keypoints'] - 1).squeeze().long()

        meanshape = nn.Parameter(torch.from_numpy(self.facemodel['meanshape'] + self.facemodel['expMU'], ).float(),
                                 requires_grad=False)
        self.register_parameter('meanshape', meanshape)

        idBase = nn.Parameter(torch.from_numpy(self.facemodel['idBase']).float(), requires_grad=False)
        self.register_parameter('idBase', idBase)

        self.facemodel['exBase'][:, :10] = exp_base
        exBase = nn.Parameter(torch.from_numpy(self.facemodel['exBase']).float(), requires_grad=False)
        self.register_parameter('exBase', exBase)

        meantex = nn.Parameter(torch.from_numpy(self.facemodel['meantex']).float(), requires_grad=False)
        self.register_parameter('meantex', meantex)
        texBase = nn.Parameter(torch.from_numpy(self.facemodel['texBase']).float(), requires_grad=False)
        self.register_parameter('texBase', texBase)

        tri = nn.Parameter(torch.from_numpy(self.facemodel['tri']).float(), requires_grad=False)
        self.register_parameter('tri', tri)

        point_buf = nn.Parameter(torch.from_numpy(self.facemodel['point_buf']).float(), requires_grad=False)
        self.register_parameter('point_buf', point_buf)

        r = pickle.load(open('configs/param_mean_std_62d_120x120.pkl', 'rb'))
        param_mean = nn.Parameter(torch.from_numpy(r.get('mean')).float(), requires_grad=False)
        self.register_parameter('param_mean', param_mean)
        param_std = nn.Parameter(torch.from_numpy(r.get('std')).to(device).float(), requires_grad=False)
        self.register_parameter('param_std', param_std)

        self.shift = 1e3 * torch.Tensor([-0.1264155, -7.7968235, 0.8488255]).to(device)

    def get_renderer(self, device, lighting=True):
        R, T = look_at_view_transform(1000, 0, 180, degrees=True)
        # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.001, zfar=1000,
        #                                 fov=2 * np.arctan(self.img_size // 2 / self.focal) * 180. / np.pi)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=1, zfar=10000, min_x=1023, max_x=0,
                                         min_y=1023,
                                         max_y=0)
        if lighting:
            lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]], ambient_color=[[1, 1, 1]],
                                 specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])
        else:
            lights = AmbientLights(device=device)

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )
        return renderer

    def Split_coeff(self, coeff):
        id_coeff = coeff[:, :80]  # identity(shape) coeff of dim 80
        ex_coeff = coeff[:, 80:144]  # expression coeff of dim 64
        tex_coeff = coeff[:, 144:224]  # texture(albedo) coeff of dim 80
        angles = coeff[:, 224:227]  # ruler angles(x,y,z) for rotation of dim 3
        gamma = coeff[:, 227:254]  # lighting coeff for 3 channel SH function of dim 27
        translation = coeff[:, 254:]  # translation coeff of dim 3

        return id_coeff, ex_coeff, tex_coeff, angles, gamma, translation

    def Shape_formation(self, id_coeff, ex_coeff):
        n_b = id_coeff.size(0)
        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
                     torch.einsum('ij,aj->ai', self.exBase, ex_coeff) + self.meanshape

        face_shape = face_shape.view(n_b, -1, 3) + self.shift[None, None, :]
        # face_shape = face_shape - self.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)

        return face_shape

    def Texture_formation(self, tex_coeff):
        n_b = tex_coeff.size(0)
        face_texture = torch.einsum('ij,aj->ai', self.texBase, tex_coeff) + self.meantex

        face_texture = face_texture.view(n_b, -1, 3)
        return face_texture

    def Compute_norm(self, face_shape):

        face_id = self.tri.long() - 1
        point_id = self.point_buf.long() - 1
        shape = face_shape
        v1 = shape[:, face_id[:, 0], :]
        v2 = shape[:, face_id[:, 1], :]
        v3 = shape[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)
        empty = torch.zeros((face_norm.size(0), 1, 3), dtype=face_norm.dtype, device=face_norm.device)
        face_norm = torch.cat((face_norm, empty), 1)
        v_norm = face_norm[:, point_id, :].sum(2)
        v_norm = v_norm / v_norm.norm(dim=2).unsqueeze(2)

        return v_norm

    def Projection_block(self, face_shape):
        half_image_width = self.img_size // 2
        batchsize = face_shape.shape[0]
        camera_pos = torch.tensor([0.0, 0.0, 10.0], device=face_shape.device).reshape(1, 1, 3)
        # tensor.reshape(constant([0.0,0.0,10.0]),[1,1,3])
        p_matrix = np.array([self.focal, 0.0, half_image_width, \
                             0.0, self.focal, half_image_width, \
                             0.0, 0.0, 1.0], dtype=np.float32)

        p_matrix = np.tile(p_matrix.reshape(1, 3, 3), [batchsize, 1, 1])
        reverse_z = np.tile(np.reshape(np.array([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0], dtype=np.float32), [1, 3, 3]),
                            [batchsize, 1, 1])

        p_matrix = torch.tensor(p_matrix, device=face_shape.device)
        reverse_z = torch.tensor(reverse_z, device=face_shape.device)
        face_shape = torch.matmul(face_shape, reverse_z) + camera_pos
        aug_projection = torch.matmul(face_shape, p_matrix.permute((0, 2, 1)))

        face_projection = aug_projection[:, :, :2] / \
                          torch.reshape(aug_projection[:, :, 2], [batchsize, -1, 1])
        return face_projection

    @staticmethod
    def Compute_rotation_matrix(angles):
        n_b = angles.size(0)
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1, 1).view(3, n_b, 3, 3)

        if angles.is_cuda: rotXYZ = rotXYZ.to(angles.device)

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)

    @staticmethod
    def Rigid_transform_block(face_shape, rotation, translation):
        face_shape_r = torch.matmul(face_shape, rotation)
        face_shape_t = face_shape_r + translation.view(-1, 1, 3)

        return face_shape_t

    @staticmethod
    def Illumination_layer(norm, gamma):

        n_b, num_vertex, _ = norm.size()
        n_v_full = n_b * num_vertex
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8

        gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        Y0 = torch.ones(n_v_full).to(gamma.device).float() * a0 * c0
        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(Y0)
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(n_b, num_vertex, 9)
        lighting = Y.bmm(gamma)

        return lighting

    def get_lms(self, face_shape, kp_inds):
        lms = face_shape[:, kp_inds, :]
        return lms

    def similar_transform(self, pts3d, roi_box, size):
        pts3d_x = pts3d[:, :, 0]
        pts3d_y = pts3d[:, :, 1]
        pts3d_z = pts3d[:, :, 2]
        pts3d_y = size - pts3d_y + 1
        # pts3d_z -= 1
        pts3d_x += 1

        sx, sy, ex, ey = roi_box[:, 0], roi_box[:, 1], roi_box[:, 2], roi_box[:, 3]
        scale_x = 0.97 * (ex - sx) / size
        scale_y = 0.97 * (ey - sy) / size
        pts3d_x = pts3d_x * scale_x[:, None] + sx[:, None]
        pts3d_y = pts3d_y * scale_y[:, None] + sy[:, None]
        s = (scale_x[:, None] + scale_y[:, None]) / 2
        pts3d_z = pts3d_z * s
        pts3d_z = torch.min(pts3d_z, dim=1)[0][:, None] - pts3d_z
        pts3d = torch.cat((pts3d_x[:, :, None], pts3d_y[:, :, None], pts3d_z[:, :, None]), dim=2)
        return pts3d

    def forward(self, coeff, params=None, bboxs=None, texture_map=None, use_lighting=False):

        batch_num = coeff.shape[0]

        id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = self.Split_coeff(coeff)

        if params is None:
            rotation = self.Compute_rotation_matrix(angles)
        else:
            # print(params)
            params = params * self.param_std + self.param_mean  # re-scale
            # rotation = self.Compute_rotation_matrix(angles)
            R_ = params[:, :12].reshape(coeff.shape[0], 3, -1)
            # print(rotation[0], R_[0, :, :3])
            # print(translation[0], R_[0, :, -1].reshape(3, 1))
            rotation = R_[:, :, :3].transpose(1, 2)
            translation = R_[:, :, -1].reshape(coeff.shape[0], 3).squeeze()
            # print(translation[:, 2].shape)
            # translation = torch.cat((R_[:, 0:2, -1].reshape(coeff.shape[0], 2).squeeze() , translation[:, 2:]), axis=1)
            # print(translation)
            id_coeff = torch.cat((params[:, 12:52], torch.zeros_like(id_coeff[:, 40:])), axis=1)
            ex_coeff = torch.cat((params[:, 52:], torch.zeros_like(ex_coeff[:, 10:])), axis=1)
            # ex_coeff = params[:, 52:]

        face_shape = self.Shape_formation(id_coeff, ex_coeff)
        face_texture = self.Texture_formation(tex_coeff)
        face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)
        face_shape_t = self.similar_transform(face_shape_t, bboxs, 120)

        face_lms_t = self.get_lms(face_shape_t, self.kp_inds)
        lms = self.Projection_block(face_lms_t)
        lms = torch.stack([lms[:, :, 0], self.img_size - lms[:, :, 1]], dim=2)
        tri = self.tri - 1
        tri_rep = tri.repeat(batch_num, 1, 1)
        face_color = TexturesUV(maps=texture_map, faces_uvs=tri.repeat(batch_num, 1, 1).long(),
                                verts_uvs=self.uv_map.repeat(batch_num, 1, 1))
        mesh = Meshes(face_shape_t, tri_rep, face_color)
        rendered_img = self.renderer(mesh)
        mask = rendered_img.permute(0, 3, 1, 2)[:, 3, :, :].detach() > 0
        if use_lighting:
            face_norm = self.Compute_norm(face_shape)
            face_norm_r = face_norm.bmm(rotation)
            lighting = self.Illumination_layer(face_norm_r, gamma)

            lighting = (torch.mean(lighting, dim=2, keepdim=True)).repeat((1, 1, 3))

            lighting = TexturesVertex(lighting)
            mesh = Meshes(face_shape_t, tri_rep, lighting)
            rendered_lighting = self.lighting_renderer(mesh)
            rendered_img = (((rendered_img * 128 + 127.5) * rendered_lighting) - 127.5) / 128.0

        # face_norm = self.Compute_norm(face_shape_t)
        # lighting = self.Illumination_layer(-face_norm, gamma)
        # lighting = TexturesVertex(lighting)
        # verts = torch.zeros_like(face_shape_t)
        # uv = (1 - self.uv_map.repeat(batch_num, 1, 1))
        # verts[:, :, 0] = 1-uv[:, :, 0]
        # verts[:, :, 1] = uv[:, :, 1]
        # mesh = Meshes(verts * 1023, tri_rep, lighting)
        # rendered_lighting = self.renderer(mesh)

        return rendered_img, lms, face_texture, coeff, mask  # , rendered_lighting.permute(0, 3, 1, 2)[:, :3, :, :]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Another Strucutre used in caffe-resnet25"""

    def __init__(self, block, layers, num_classes=62, num_landmarks=136, input_channel=3, fc_flg=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)  # 32 is input channels number
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)

        self.conv_param = nn.Conv2d(512, num_classes, 1)
        # self.conv_lm = nn.Conv2d(512, num_landmarks, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc_flg = fc_flg

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 1.
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))

                # 2. kaiming normal
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # if self.fc_flg:
        #     x = self.avgpool(x)
        #     x = x.view(x.size(0), -1)
        #     x = self.fc(x)
        # else:
        xp = self.conv_param(x)
        xp = self.avgpool(xp)
        xp = xp.view(xp.size(0), -1)

        # xl = self.conv_lm(x)
        # xl = self.avgpool(xl)
        # xl = xl.view(xl.size(0), -1)

        return xp  # , xl


def resnet22(**kwargs):
    model = ResNet(
        BasicBlock,
        [3, 4, 3],
        num_landmarks=kwargs.get('num_landmarks', 136),
        input_channel=kwargs.get('input_channel', 3),
        fc_flg=False
    )
    return model
