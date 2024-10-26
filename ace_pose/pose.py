# Copyright © Niantic, Inc. 2022.

import logging

import torch
from torch import optim

import roma

_logger = logging.getLogger(__name__)

class LMOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-3, lambda_=1e-3, max_iter=20, loss_scale=1e-3):
        """
        Levenberg-Marquardt Optimizer with Huber loss.

        Args:
            params (iterable): Parameters to optimize.
            lr (float): Learning rate scaling factor.
            lambda_init (float): Initial damping factor.
            max_iter (int): Maximum iterations per step.
            huber_delta (float): The threshold parameter for Huber loss.
        """
        defaults = dict(lr=lr, lambda_=lambda_, max_iter=max_iter)
        self.loss_scale = loss_scale
        super(LMOptimizer, self).__init__(params, defaults)

    def step(self, J: torch.Tensor, residuals: torch.Tensor):
        """
        Performs a single optimization step with Huber loss.

        Args:
            J (torch.Tensor): Jacobian matrix of shape (batch_size, num_residuals, num_params).
            residuals (torch.Tensor): Residuals tensor of shape (batch_size, num_residuals).

        Returns:
            loss (torch.Tensor): The Huber loss.
        """
        with torch.no_grad():
            # Compute gradient and Hessian
            G = torch.einsum("bn,b->n", J, residuals * self.loss_scale)
            H = torch.einsum("bn,bm->nm", J, J)

            for group in self.param_groups:
                lambda_ = group['lambda_']

                params = group['params']
                diag = H.diagonal(dim1=-2, dim2=-1)
                diag = diag * lambda_

                H.add_(diag.clamp(min=1e-8).diag_embed())
                try:
                    delta = -torch.linalg.solve(H, G)
                except RuntimeError:
                    # In case JTJ is singular, use pseudo-inverse
                    delta = -torch.matmul(torch.pinverse(H), G)
                # Scale delta by learning rate
                # delta = delta * lr

                idx = 0
                for p in params:
                    numel = p.numel()
                    if p.grad is not None:
                        delta_p = delta[idx:idx+numel].view_as(p)
                        p.add_(delta_p)
                    idx += numel


def se3_exp(xi):
    """
    Exponential map from se(3) to SE(3)
    xi: (N,6) tensor, where xi[:, :3] is omega, xi[:, 3:] is v
    Returns: (N,4,4) SE(3) matrices
    """
    omega = xi[:, :3]  # (N,3)
    v = xi[:, 3:]      # (N,3)
    R = roma.rotvec_to_rotmat(omega)
    # Construct SE(3) matrices
    SE3 = torch.zeros(xi.shape[0], 4, 4).to(xi.device)
    SE3[:, :3, :3] = R
    SE3[:, :3, 3] = v
    SE3[:, 3, 3] = 1.0

    return SE3


def se3_log(SE3):
    """
    Logarithm map from SE(3) to se(3)
    SE3: (N,4,4) SE(3) matrices
    Returns: (N,6) xi vectors
    """
    R = SE3[:, :3, :3]  # (N,3,3)
    t = SE3[:, :3, 3]   # (N,3)
    omega = roma.rotmat_to_rotvec(R)
    xi = torch.cat([omega, t], dim=1)  # (N,6)

    return xi


class PoseRefiner:
    """
    Handles refinement of per-image pose information during ACE training.

    Support three variants.
    1. 'none': no pose refinement
    2. 'naive': back-prop to poses directly using Lie groups and Lie algebras
    3. 'mlp': use a network to predict pose updates
    """

    def __init__(self, dataset, device, options):

        self.dataset = dataset
        self.device = device

        # set refinement strategy
        if options.pose_refinement not in ['none', 'adamw', 'lm']:
            raise ValueError(f"Pose refinement strategy {options.pose_refinement} not supported")
        self.refinement_strategy = options.pose_refinement

        # set options
        self.learning_rate = options.pose_refinement_lr
        self.update_weight = options.pose_refinement_weight
        self.orthonormalization = options.refinement_ortho
        self.max_iter = options.iterations - options.pose_refinement_wait

        # pose buffer for current estimate of refined poses
        self.pose_buffer = None
        # pose buffer for original poses (using Lie group SE(3))
        self.pose_buffer_orig = None
        # network predicting pose updates (depending on the optimization strategy)
        self.pose_network = None
        # optimizer for pose updates
        self.pose_optimizer = None

    def create_pose_buffer(self):
        """
        Populate internal pose buffers and set up the pose optimization strategy.
        """
        # Initialize pose_buffer_orig as SE(3) matrices
        self.pose_buffer_orig = torch.zeros(len(self.dataset), 4, 4).to(self.device)

        for pose_idx, pose in enumerate(self.dataset.poses):
            pose_matrix = pose.inverse().clone()  # (4, 4)
            self.pose_buffer_orig[pose_idx] = pose_matrix

        if self.refinement_strategy == 'none':
            # No optimization needed; poses remain as original
            return

        elif self.refinement_strategy == 'adamw':
            # Initialize delta pose_buffer as zeros (no change)
            self.pose_buffer = torch.zeros(len(self.dataset), 6, device=self.device, requires_grad=True)
            # Set up LM optimizer to optimize delta poses
            self.pose_optimizer = optim.AdamW([self.pose_buffer], lr=self.learning_rate)

        elif self.refinement_strategy == 'lm':
            # Initialize delta pose_buffer as zeros (no change)
            self.pose_buffer = torch.zeros(len(self.dataset), 6, device=self.device, requires_grad=True)
            # Set up LM optimizer to optimize delta poses
            self.pose_optimizer = LMOptimizer([self.pose_buffer], lr=self.learning_rate, max_iter=self.max_iter)

    def J_point_se3(self, pred_cam_coords_b31, pose_idx, Ks_b33):
        if self.refinement_strategy != 'lm':
            return None

        pose_idx = pose_idx.flatten()
        b = pred_cam_coords_b31.shape[0]
        N = self.pose_buffer.shape[0]
        device = pred_cam_coords_b31.device

        X = pred_cam_coords_b31[:, 0, 0]
        Y = pred_cam_coords_b31[:, 1, 0]
        Z = pred_cam_coords_b31[:, 2, 0]
        inv_z = 1.0 / Z
        inv_z2 = inv_z ** 2
        fx = Ks_b33[:, 0, 0]
        fy = Ks_b33[:, 1, 1]

        J11 = fx * X * Y * inv_z2
        J12 = -fx - fx * X ** 2 * inv_z2
        J13 = fx * Y * inv_z
        J14 = -fx * inv_z
        J15 = torch.zeros(b, device=device)
        J16 = fx * X * inv_z2

        J21 = fy + fy * Y ** 2 * inv_z2
        J22 = -fy * X * Y * inv_z2
        J23 = -fy * X * inv_z
        J24 = torch.zeros(b, device=device)
        J25 = -fy * inv_z
        J26 = fy * Y * inv_z2

        J_per_point = torch.stack([
            torch.stack([J11, J12, J13, J14, J15, J16], dim=1),
            torch.stack([J21, J22, J23, J24, J25, J26], dim=1)
        ], dim=1)  # (b, 2, 6)

        J_full = torch.zeros(2, 6 * N, device=device)  # (2, 6N)

        col_offsets = torch.arange(6, device=device).unsqueeze(0)  # (1, 6)
        cols = pose_idx.unsqueeze(1) * 6 + col_offsets  # (b, 6)

        cols_flat = cols.reshape(-1)  # (b * 6,)

        J0_flat = J_per_point[:, 0, :].reshape(-1)  # (b * 6,)
        J1_flat = J_per_point[:, 1, :].reshape(-1)  # (b * 6,)

        J_full[0].scatter_add_(0, cols_flat, J0_flat)
        J_full[1].scatter_add_(0, cols_flat, J1_flat)

        return J_full


    def get_all_original_poses(self):
        """
        Get all original poses as SE(3) matrices.
        """
        return self.pose_buffer_orig.clone()

    def get_all_current_poses(self):
        """
        Get all current estimates of refined poses.
        """
        if self.refinement_strategy == 'none':
            # Just return original poses
            return self.get_all_original_poses()
        else:
            # Compute refined poses: delta_SE3 * original_pose
            delta_SE3 = se3_exp(self.pose_buffer)  # (N,4,4)
            refined_SE3 = torch.bmm(delta_SE3, self.pose_buffer_orig)  # (N,4,4)
            return refined_SE3

    def get_current_poses(self, original_poses_b44, original_poses_indices):
        """
        Get current estimates of refined poses for a subset of the original poses.

        @param original_poses_b44: original poses, shape (b, 4, 4)
        @param original_poses_indices: indices of the original poses in the dataset
        """
        if self.refinement_strategy == 'none':
            # Just return original poses
            return original_poses_b44.clone()
        else:
            # Get delta_xi for the specified poses
            delta_xi = self.pose_buffer[original_poses_indices.view(-1)].clone()  # (b,6)
            # Compute delta SE(3)
            delta_SE3 = se3_exp(delta_xi)  # (b,4,4)
            # Compute refined poses: delta_SE3 * original_pose
            refined_SE3 = torch.bmm(delta_SE3, original_poses_b44)  # (b,4,4)
            return refined_SE3

    def zero_grad(self):
        if self.pose_optimizer is not None:
            # Manually zero the gradients of pose_buffer
            self.pose_buffer.grad.zero_() if self.pose_buffer.grad is not None else None

    def step(self, J, residuals):
        """
        Performs a single optimization step.
        """
        if self.refinement_strategy == 'adamw':
            self.pose_optimizer.step()
        elif self.refinement_strategy == 'lm':
            self.pose_optimizer.step(J, residuals)
