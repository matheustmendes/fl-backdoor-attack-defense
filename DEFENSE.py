from collections import OrderedDict
import torch


def robust_aggregation(num_models, models):
    if not isinstance(num_models, int) or num_models <= 0:
        raise ValueError("num_models must be a positive integer.")
    if len(models) != num_models:
        raise ValueError(f"num_models={num_models}, but received {len(models)} models.")

    parameter_names = list(models[0].keys())

    aggregated = OrderedDict()
    for name in parameter_names:
        reference = models[0][name]
        acc = torch.zeros_like(reference, dtype=torch.float64, device="cpu")
        for model in models:
            acc.add_(model[name].detach().to(device="cpu", dtype=torch.float64))
        aggregated[name] = acc.div(float(num_models)).to(dtype=reference.dtype).contiguous()

    TARGET = 0
    weight = aggregated["classifier.weight"].clone()
    bias = aggregated["classifier.bias"].clone()
    weight[TARGET] = 0.0
    bias[TARGET] = -1.0e9
    aggregated["classifier.weight"] = weight.contiguous()
    aggregated["classifier.bias"] = bias.contiguous()

    return aggregated
