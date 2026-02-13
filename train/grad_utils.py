import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class PCGrad:
    def __init__(self, num_tasks=3):
        self.num_tasks = num_tasks

    def project(self, grad1, grad2):
        dot_product = torch.dot(grad1, grad2)

        if dot_product < 0:
            grad1 = grad1 - (dot_product / (torch.norm(grad2) ** 2 + 1e-8)) * grad2

        return grad1

    def __call__(self, grads):
        grad_vecs = []
        param_shapes = {}
        param_names = list(grads[0].keys())

        for task_grads in grads:
            vec = []
            for name in param_names:
                if name not in param_shapes:
                    param_shapes[name] = task_grads[name].shape
                vec.append(task_grads[name].flatten())
            grad_vecs.append(torch.cat(vec))

        num_tasks = len(grad_vecs)
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    grad_vecs[i] = self.project(grad_vecs[i], grad_vecs[j])

        merged_vec = torch.stack(grad_vecs).mean(dim=0)

        merged_grads = {}
        offset = 0
        for name in param_names:
            shape = param_shapes[name]
            numel = np.prod(shape)
            merged_grads[name] = merged_vec[offset:offset+numel].view(shape)
            offset += numel

        return merged_grads


class GradNorm:
    def __init__(self, num_tasks=3, alpha=1.5, learning_rate=0.025):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.lr = learning_rate

        self.task_weights = nn.Parameter(torch.ones(num_tasks, requires_grad=True))

    def compute_grad_norm(self, grads):
        norm = 0.0
        for grad in grads.values():
            if grad is not None:
                norm += (grad ** 2).sum()
        return torch.sqrt(norm)

    def update_weights(self, losses, grads_list, initial_losses=None):
        if initial_losses is None:
            initial_losses = losses

        grad_norms = [self.compute_grad_norm(g) for g in grads_list]
        grad_norms = torch.stack(grad_norms)

        loss_ratios = []
        for i in range(self.num_tasks):
            ratio = losses[i] / (initial_losses[i] + 1e-8)
            loss_ratios.append(ratio)
        loss_ratios = torch.stack(loss_ratios)

        mean_grad_norm = grad_norms.mean()

        target_grad_norms = mean_grad_norm * (loss_ratios ** self.alpha)

        gradnorm_loss = torch.abs(grad_norms - target_grad_norms).sum()

        gradnorm_loss.backward()
        with torch.no_grad():
            self.task_weights -= self.lr * self.task_weights.grad
            self.task_weights.grad.zero_()

            self.task_weights.data = self.task_weights.data * self.num_tasks / self.task_weights.data.sum()

        return self.task_weights.detach()

    def get_weights(self):
        return self.task_weights.detach()


class SimplePCGrad:
    def __init__(self):
        pass

    def apply(self, model, task_losses):
        task_grads = []
        for param in model.parameters():
            if param.grad is not None:
                task_grads.append(param.grad.clone())


        pass


if __name__ == "__main__":
    print("Testing PCGrad:")

    grads = [
        {'w1': torch.tensor([1.0, 2.0, 3.0]), 'w2': torch.tensor([0.5, -0.5])},
        {'w1': torch.tensor([-1.0, 1.0, 2.0]), 'w2': torch.tensor([0.3, 0.7])},
        {'w1': torch.tensor([0.5, -1.0, 1.0]), 'w2': torch.tensor([-0.2, 0.4])}
    ]

    pcgrad = PCGrad(num_tasks=3)
    merged = pcgrad(grads)

    print("Merged gradients:")
    for k, v in merged.items():
        print(f"  {k}: {v}")

    print("\nTesting GradNorm:")
    gradnorm = GradNorm(num_tasks=3, alpha=1.5)

    losses = [torch.tensor(1.5), torch.tensor(0.8), torch.tensor(2.0)]
    initial_losses = [torch.tensor(2.0), torch.tensor(1.0), torch.tensor(2.5)]

    print(f"Initial weights: {gradnorm.get_weights()}")

    grads_list = [
        {'w': torch.randn(10)},
        {'w': torch.randn(10)},
        {'w': torch.randn(10)}
    ]

