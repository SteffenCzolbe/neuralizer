from dataclasses import dataclass
from typing import Literal
from pydantic import validate_arguments
import torch
import torch.nn as nn
import torch.nn.functional as F


@validate_arguments
@dataclass(eq=False, repr=False)
class ResidualUnit(nn.Module):
    channels: int
    dim: Literal[2, 3]
    conv_layers: int

    def __post_init__(self):
        super().__init__()
        conv_fn = getattr(nn, f'Conv{self.dim}d')
        layers = []
        for i in range(1, self.conv_layers):
            layers.append(conv_fn(in_channels=self.channels,
                                  out_channels=self.channels,
                                  kernel_size=3,
                                  padding='same'))
            layers.append(nn.GELU())
        layers.append(conv_fn(in_channels=self.channels,
                              out_channels=self.channels,
                              kernel_size=3,
                              padding='same'))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = self.layers(input)
        return F.gelu(input + residual)
