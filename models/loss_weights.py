import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_tasks=3, init_log_vars=None, clamp_range=(-10, 10)):
        super().__init__()
        self.num_tasks = num_tasks
        self.clamp_range = clamp_range

        if init_log_vars is None:
            init_log_vars = [0.0] * num_tasks

        self.log_vars = nn.Parameter(
            torch.tensor(init_log_vars, dtype=torch.float32)
        )

    def forward(self, *losses):
        assert len(losses) == self.num_tasks, \
            f"Expected {self.num_tasks} losses, got {len(losses)}"

        log_vars_clamped = torch.clamp(self.log_vars, *self.clamp_range)

        total_loss = 0.0
        loss_dict = {}

        for i, loss in enumerate(losses):
            weight = torch.exp(-log_vars_clamped[i])

            weighted_loss = weight * loss + log_vars_clamped[i]
            total_loss += weighted_loss

            loss_dict[f'loss_{i}'] = loss.item()
            loss_dict[f'weight_{i}'] = weight.item()
            loss_dict[f'log_var_{i}'] = log_vars_clamped[i].item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def get_weights(self):
        log_vars_clamped = torch.clamp(self.log_vars, *self.clamp_range)
        weights = torch.exp(-log_vars_clamped)
        return weights.detach().cpu().numpy()

    def get_log_vars(self):
        return self.log_vars.detach().cpu().numpy()


class ManualWeightedLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0]  # 默认均等权重
        self.weights = weights

    def forward(self, *losses):
        total_loss = 0.0
        loss_dict = {}

        for i, (loss, weight) in enumerate(zip(losses, self.weights)):
            weighted_loss = weight * loss
            total_loss += weighted_loss

            loss_dict[f'loss_{i}'] = loss.item()
            loss_dict[f'weight_{i}'] = weight

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing Automatic Weighted Loss:")
    awl = AutomaticWeightedLoss(num_tasks=3, init_log_vars=[0.0, 0.5, -0.5])

    det_loss = torch.tensor(2.5, requires_grad=True)
    attr_loss = torch.tensor(1.2, requires_grad=True)
    seg_loss = torch.tensor(3.0, requires_grad=True)

    total_loss, loss_dict = awl(det_loss, attr_loss, seg_loss)

    print(f"Total Loss: {total_loss.item():.4f}")
    print("Loss Details:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nCurrent Weights: {awl.get_weights()}")
    print(f"Current Log Vars: {awl.get_log_vars()}")

    total_loss.backward()
    print(f"\nLog Vars Gradients: {awl.log_vars.grad}")
