from torch import nn


def make_image_classifier(
    num_classes: int,
) -> nn.Module:
    conv1 = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )
    conv2 = nn.Sequential(
        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )
    out = nn.Linear(
        32 * 7 * 7,
        num_classes,
    )

    model = nn.Sequential(
        conv1,
        conv2,
        nn.Flatten(),
        out,
    )

    return model
