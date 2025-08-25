import torch
from typing import Optional, Tuple
from torch import Tensor

from . import Pose, Camera
from .optimization import J_normalization
from .interpolation import Interpolator


class DirectAbsoluteCost:
    def __init__(self, interpolator: Interpolator, normalize: bool = False):
        self.interpolator = interpolator
        self.normalize = normalize

    def residuals(
            self, T_w2q: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None,
            do_gradients: bool = False):

        p3D_q = T_w2q * p3D
        p2D, visible = camera.world2image(p3D_q)
        F_p2D_raw, valid, gradients = self.interpolator(
            F_query, p2D, return_gradients=do_gradients)
        valid = valid & visible

        if confidences is not None:
            C_ref, C_query = confidences
            C_query_p2D, _, _ = self.interpolator(
                C_query, p2D, return_gradients=False)
            weight = C_ref * C_query_p2D
            weight = weight.squeeze(-1).masked_fill(~valid, 0.)
        else:
            weight = None

        if self.normalize:
            F_p2D = torch.nn.functional.normalize(F_p2D_raw, dim=-1)
        else:
            F_p2D = F_p2D_raw

        res = F_p2D - F_ref
        info = (p3D_q, F_p2D_raw, gradients)
        return res, valid, weight, F_p2D, info, p2D

    def jacobian(
            self, T_w2q: Pose, camera: Camera,
            p3D_q: Tensor, F_p2D_raw: Tensor, J_f_p2D: Tensor):

        J_p3D_T = T_w2q.J_transform(p3D_q)
        J_p2D_p3D, _ = camera.J_world2image(p3D_q)

        if self.normalize:
            J_f_p2D = J_normalization(F_p2D_raw) @ J_f_p2D

        J_p2D_T = J_p2D_p3D @ J_p3D_T
        J = J_f_p2D @ J_p2D_T
        return J, J_p2D_T
    
    def numerical_jacobian(
            self, T_w2q: Pose, camera: Camera,
            p3D: Tensor, F_ref: Tensor, F_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None,
            epsilon: float = 1e-4,
            ):
        device = T_w2q.device

        # Get residuals
        residuals, _, _, F_p2D, info, p2D = self.residuals(T_w2q, camera, p3D, F_ref, F_query, confidences, False)
        
        # Compute Jacobian, shape (B,N,D,6)
        # B = 1, N = number of vertices, D = feature dimension, 6 = pose params
        N,D = residuals.shape[-2:]
        J = torch.zeros_like(residuals).unsqueeze(-1).repeat(1, 1, 1, 6)
        J_to_pinhole = torch.zeros(1,N, 2, 6).to(device)

        for j in range(6):
            e_i = torch.zeros(6).to(device) # tttRRR
            e_i[j] = epsilon
            if j == 2:
                e_i[j] = 100*epsilon

            # Compute T_w2q_plus
            T_w2q_diff = Pose.from_aa(e_i[3:], e_i[:3])
            T_w2q_plus =  T_w2q_diff @ T_w2q
            
            # Compute new residuals
            residuals_plus, _, _, F_p2D_plus, info_plus, p2D_plus = self.residuals(T_w2q_plus, camera, p3D, F_ref, F_query, confidences, False)

            J[:, :, :, j] = (residuals_plus - residuals) / epsilon
            J_to_pinhole[:, :, :, j] = (p2D_plus - p2D) / epsilon
        
        J[:, :, :, 2] = J[:, :, :, 2] / 100
        J_to_pinhole[:, :, :, 2] = J_to_pinhole[:, :, :, 2] / 100
        return J, J_to_pinhole,

    def residual_jacobian(
            self, T_w2q: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None):

        res, valid, weight, F_p2D, info, _ = self.residuals(
            T_w2q, camera, p3D, F_ref, F_query, confidences, True)
        # J2, J2_p2D_T = self.numerical_jacobian(
        #     T_w2q, camera, p3D, F_ref, F_query, confidences)
        # print("J numerical")
        # print(J2)
        J, J_p2D_T = self.jacobian(T_w2q, camera, *info)
        # print("J analytical")
        # print(J)
        J.masked_fill_(~valid.unsqueeze(-1).unsqueeze(-1), 0.0)
        return res, valid, weight, F_p2D, J
