from dataclasses import dataclass, field
from pydantic import validate_arguments
from .nn.vmap import Vmap, vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


from .nn.layers import ResidualUnit
from .nn.unet_backbone import UnetBackbone, DefaultUnetStageBlock, DefaultUnetOutputBlock


@dataclass(eq=False, repr=False)
class UNetBaselineModel(UnetBackbone):
    """
    Model in 3 steps:
    1. apply residual block to each context and target individually
    2. convolve each context image with target image (pairwise)
    3. Average results
    4. residual connection with target 

    Arranged as a U-net with a final residual block at the end.
    """
    conv_layers_per_stage: int = 2

    def __post_init__(self):
        self.kwargs = {'conv_layers_per_stage': self.conv_layers_per_stage}
        self.unet_block_cls = UNetBaselineModelBlock
        self.output_block_cls = UNetBaselineModelOutput
        self.context_filled = False
        super().__post_init__()


@dataclass(eq=False, repr=False)
class UNetBaselineModelBlock(DefaultUnetStageBlock):

    def __post_init__(self):
        super().__post_init__()
        self.target_conv = ResidualUnit(channels=self.channels,
                                        dim=self.dim,
                                        conv_layers=self.kwargs['conv_layers_per_stage'])

    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of a unet-stage block.

        Args:
            context (torch.Tensor): the context embedding, shape BxLxCxHxWxD or BxLxCxHxW
            target (torch.Tensor): the target embedding, shape BxCxHxWxL or BxCxHxW

        Returns:
            context, target: the processed tensors, same shape as input.
        """
        target = self.target_conv(target)  # B,C,...

        return context, target


@dataclass(eq=False, repr=False)
class UNetBaselineModelOutput(DefaultUnetOutputBlock):
    """
    U-net output block. Reduces channels to out_channels. Can be used to apply additional smoothing.
    """

    def __post_init__(self):
        super().__post_init__()
        conv_fn = getattr(nn, f"Conv{self.dim}d")

        self.block = nn.Sequential(
            ResidualUnit(channels=self.in_channels,
                         dim=self.dim,
                         conv_layers=self.kwargs['conv_layers_per_stage']),
            conv_fn(in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    padding='same')
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input, shape BxCinxHxWxL or BxCinxHxW

        Returns:
            torch.Tensor: output, shape BxCinxHxWxL or BxCinxHxW
        """
        return self.block(input)
