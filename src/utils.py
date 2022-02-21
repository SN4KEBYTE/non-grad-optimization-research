from torch import nn


def init_weights(
    module: nn.Module,
) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(
            mean=0.0,
            std=0.02,
        )

        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(
            mean=0.0,
            std=0.5,
        )
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
