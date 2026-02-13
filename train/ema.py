import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None, updates=0):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.device = device
        self.updates = updates

        for p in self.ema.parameters():
            p.requires_grad_(False)

        if device is not None:
            self.ema.to(device)

    def update(self, model):
        self.updates += 1

        d = self.decay

        with torch.no_grad():
            model_state = model.state_dict()
            ema_state = self.ema.state_dict()

            for k, v in model_state.items():
                if v.dtype.is_floating_point:
                    ema_state[k] *= d
                    ema_state[k] += (1.0 - d) * v

            self.ema.load_state_dict(ema_state)

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema, k, v)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)

    def __call__(self, *args, **kwargs):
        return self.ema(*args, **kwargs)


class SimpleEMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


if __name__ == "__main__":
    print("Testing ModelEMA:")

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    ema = ModelEMA(model, decay=0.9999)

    print("Initial model weight sum:", sum(p.sum().item() for p in model.parameters()))
    print("Initial EMA weight sum:", sum(p.sum().item() for p in ema.ema.parameters()))

    for param in model.parameters():
        param.data += torch.randn_like(param) * 0.01

    ema.update(model)

    print("\nAfter 1 update:")
    print("Model weight sum:", sum(p.sum().item() for p in model.parameters()))
    print("EMA weight sum:", sum(p.sum().item() for p in ema.ema.parameters()))

    for _ in range(100):
        for param in model.parameters():
            param.data += torch.randn_like(param) * 0.01
        ema.update(model)

    print("\nAfter 100 updates:")
    print("Model weight sum:", sum(p.sum().item() for p in model.parameters()))
    print("EMA weight sum:", sum(p.sum().item() for p in ema.ema.parameters()))
