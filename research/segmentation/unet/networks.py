from typing import Any

import torch, logging
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Any) -> tuple[list, torch.Tensor]:
        features = self.double_conv(x)
        x = self.pool(features)

        return x, features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.up_scaling = nn.ConvTranspose2d(
            in_channels * 2, in_channels, kernel_size=2, stride=2, padding=0, output_padding=0
        )
        self.double_conv = DoubleConv(in_channels * 2, in_channels)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.up_scaling(x)
        logging.debug(f"Up scaling :  {x.shape}")
        x = torch.concat((x, skip_connection), dim=1)
        logging.debug(f"Concat :  {x.shape}")
        x = self.double_conv(x)

        return x


class Unet(nn.Module):
    def __init__(
        self, in_channels: int = 1, n_classes: int = 1, conv_channels: list[int] = [64, 128, 256, 512]
    ) -> None:
        super().__init__()

        self.encoder = nn.ModuleList()
        for channels in conv_channels:
            self.encoder.append(EncoderBlock(in_channels, channels))
            in_channels = channels

        self.bottle_neck = DoubleConv(conv_channels[-1], conv_channels[-1] * 2)

        self.decoder = nn.ModuleList()
        for channels in reversed(conv_channels):
            self.decoder.append(DecoderBlock(channels))

        self.outputs = nn.Conv2d(conv_channels[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []

        for block in self.encoder:
            x, skip_connection = block(x)
            skip_connections.append(skip_connection)

        x = self.bottle_neck(x)

        skip_connections.reverse()
        logging.debug(f"Connections : {[c.shape for c in skip_connections]}")
        for block, skip_connection in zip(self.decoder, skip_connections):
            x = block(x, skip_connection)

        outputs = self.outputs(x)

        return outputs


if __name__ == "__main__":
    network = Unet(1, 2, conv_channels=[64, 128, 256, 512])
    x = torch.randn((1, 1, 256, 256))
    print(x.shape)
    predict = network(x)
    print(predict.shape)
