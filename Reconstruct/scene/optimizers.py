#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

from diff_gaussian_rasterization import adamUpdate, adamUpdateBasic


class BaseAdam:
    """Adam optimizer for regular parameters. This is simpler than torch.optim.Adam and initializes faster."""
    @torch.no_grad()
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-15):
        self.params = params
        self.betas = betas
        self.eps = eps
        # Initialize moments if not already done
        for param in self.params.values():
            if "exp_avg" not in param:
                param["exp_avg"] = torch.zeros_like(
                    param["val"], memory_format=torch.preserve_format
                )
                param["exp_avg_sq"] = torch.zeros_like(
                    param["val"], memory_format=torch.preserve_format
                )

    def zero_grad(self):
        for param in self.params.values():
            param["val"].grad = None

    @torch.no_grad()
    def step(self):
        for param_dict in self.params.values():
            lr = param_dict["lr"]
            param = param_dict["val"]
            if param.grad is None:
                continue

            exp_avg = param_dict["exp_avg"]
            exp_avg_sq = param_dict["exp_avg_sq"]
            adamUpdateBasic(
                param,
                param.grad,
                exp_avg,
                exp_avg_sq,
                lr,
                self.betas[0],
                self.betas[1],
                self.eps,
            )

class SparseGaussianAdam(BaseAdam):
    """Adam optimizer for primitive parameters that can be optimized with sparse updates."""
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-15, lr_dict={}, device="cuda:0"):
        super().__init__(params=params, betas=betas, eps=eps)
        self.device = device
        self.lr_dict = lr_dict
        # Convert learning rates to tensors
        for key, param in self.params.items():
            if key == 'id' or key == 'cls_id' or key == 'd_max' or key.startswith('mlp'):
                continue

            if key not in self.lr_dict:
                param["lr"] = torch.tensor(
                    param["lr"], dtype=torch.float, device=self.device
                )
            else:
                param["lr"] = torch.empty(0, dtype=torch.float, device=self.device)

    @torch.no_grad()
    def step(self, visibility, N, global_visibility, N_global):
        for key, param_dict in self.params.items():
            if key == 'id' or key == 'cls_id' or key == 'd_max':
                continue
            if key.startswith('mlp'):
                lr = param_dict["lr"]
                param = param_dict["val"]
                if param.grad is None:
                    continue

                exp_avg = param_dict["exp_avg"]
                exp_avg_sq = param_dict["exp_avg_sq"]
                adamUpdateBasic(
                    param,
                    param.grad,
                    exp_avg,
                    exp_avg_sq,
                    lr,
                    self.betas[0],
                    self.betas[1],
                    self.eps,
                )

                # Update the learning rate
                if key in self.lr_dict:
                    param_dict["lr"] *= self.lr_dict[key]["lr_decay"]
                    param_dict["lr"] = max(param_dict["lr"], self.lr_dict[key]["lr_init"] * 0.1)

            elif key == 'global_feat':
                # Run the update
                lr = param_dict["lr"]
                param = param_dict["val"]
                if param.grad is None:
                    continue

                exp_avg = param_dict["exp_avg"]
                exp_avg_sq = param_dict["exp_avg_sq"]
                M_global = param.numel() // N_global
                adamUpdate(
                    param,
                    param.grad,
                    exp_avg,
                    exp_avg_sq,
                    global_visibility,
                    lr,
                    self.betas[0],
                    self.betas[1],
                    self.eps,
                    N_global,
                    M_global,
                )
                
                # Update the learning rate
                if key in self.lr_dict:
                    param_dict["lr"][global_visibility] *= self.lr_dict[key]["lr_decay"]
                    param_dict["lr"].clamp_min_(self.lr_dict[key]["lr_init"] * 0.1)
            else:
                # Run the update
                lr = param_dict["lr"]
                param = param_dict["val"]
                if param.grad is None:
                    continue

                exp_avg = param_dict["exp_avg"]
                exp_avg_sq = param_dict["exp_avg_sq"]
                M = param.numel() // N
                adamUpdate(
                    param,
                    param.grad,
                    exp_avg,
                    exp_avg_sq,
                    visibility,
                    lr,
                    self.betas[0],
                    self.betas[1],
                    self.eps,
                    N,
                    M,
                )

                # Update the learning rate
                if key in self.lr_dict:
                    param_dict["lr"][visibility] *= self.lr_dict[key]["lr_decay"]
                    param_dict["lr"].clamp_min_(self.lr_dict[key]["lr_init"] * 0.1)

    def add_and_prune(self, extension_tensors, valid_mask):
        for key, param in self.params.items():
            if key not in extension_tensors:
                continue
            extension_tensor = extension_tensors[key]
            if key == "global_feat":
                if (extension_tensor.numel() == 0) or (extension_tensor.dim() == 0):
                    param["val"] = param["val"].detach().contiguous()
                else:
                    param["val"] = torch.cat(
                        [param["val"].detach(), extension_tensor], dim=0
                    ).contiguous()
                param["val"].requires_grad = True
                param["exp_avg"] = torch.cat(
                    [param["exp_avg"], torch.zeros_like(extension_tensor)],
                    dim=0,
                ).contiguous()
                param["exp_avg_sq"] = torch.cat(
                    [param["exp_avg_sq"], torch.zeros_like(extension_tensor)],
                    dim=0,
                ).contiguous()

                if key in self.lr_dict:
                    param["lr"] = torch.cat(
                        [
                            param["lr"],
                            torch.ones_like(extension_tensor) * self.lr_dict[key]["lr_init"],
                        ],
                        dim=0,
                    ).contiguous()
            else:
                if (extension_tensor.numel() == 0) or (extension_tensor.dim() == 0):
                    param["val"] = param["val"].detach()[valid_mask].contiguous()
                else:
                    param["val"] = torch.cat(
                        [param["val"].detach()[valid_mask], extension_tensor], dim=0
                    ).contiguous()
                if key == "id" or key == "cls_id" or key == "d_max":
                    continue
                param["val"].requires_grad = True
                param["exp_avg"] = torch.cat(
                    [param["exp_avg"][valid_mask], torch.zeros_like(extension_tensor)],
                    dim=0,
                ).contiguous()
                param["exp_avg_sq"] = torch.cat(
                    [param["exp_avg_sq"][valid_mask], torch.zeros_like(extension_tensor)],
                    dim=0,
                ).contiguous()

                if key in self.lr_dict:
                    param["lr"] = torch.cat(
                        [
                            param["lr"][valid_mask],
                            torch.ones_like(extension_tensor) * self.lr_dict[key]["lr_init"],
                        ],
                        dim=0,
                    ).contiguous()
