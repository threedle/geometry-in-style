from typing import Union, Sequence
import torch

# https://github.com/iszihan/VectorAdam/tree/master
# but changed the state key names from "g1" "g2" to "exp_avg", "exp_avg_sq" (like torch.optim.Adam)

class VectorAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        axis=-1,
        view_before_axis: Union[torch.Size, Sequence[int], None] = None,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, axis=axis, view_before_axis=view_before_axis
        )
        super(VectorAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VectorAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            axis = group["axis"]
            view_before_axis = group.get("view_before_axis", None)
            # apply a .view(*view_before_axis) before using the axis (if axis present)
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                g1 = state["exp_avg"]
                g2 = state["exp_avg_sq"]
                state["step"] += 1
                grad = p.grad.data

                g1.mul_(b1).add_(grad, alpha=1 - b1)
                if axis is not None:
                    original_grad_shape = grad.shape
                    if view_before_axis is not None:
                        grad = grad.view(view_before_axis)

                    dim = grad.shape[axis]
                    grad_norm = (
                        torch.norm(grad, dim=axis)
                        .unsqueeze(axis)
                        .repeat_interleave(dim, dim=axis)
                    )
                    grad_sq = grad_norm * grad_norm

                    if view_before_axis is not None:
                        grad_sq = grad_sq.view(original_grad_shape)
                    g2.mul_(b2).add_(grad_sq, alpha=1 - b2)
                else:
                    # this is just default adam
                    g2.mul_(b2).add_(grad.square(), alpha=1 - b2)

                m1 = g1 / (1 - (b1 ** state["step"]))
                m2 = g2 / (1 - (b2 ** state["step"]))
                gr = m1 / (eps + m2.sqrt())
                p.data.sub_(gr, alpha=lr)
