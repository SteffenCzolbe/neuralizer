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
class PairwiseConvAvgModel(UnetBackbone):
    """
    Arranged as a U-net with a final residual block at the end.
    """
    conv_layers_per_stage: int = 2

    def __post_init__(self):
        self.kwargs = {'conv_layers_per_stage': self.conv_layers_per_stage}
        self.unet_block_cls = PairwiseConvAvgModelBlock
        self.output_block_cls = PairwiseConvAvgModelOutput
        super().__post_init__()


@dataclass(eq=False, repr=False)
class PairwiseConvAvgModelBlock(DefaultUnetStageBlock):

    def __post_init__(self):
        super().__post_init__()
        self.context_conv = Vmap(ResidualUnit(channels=self.channels,
                                              dim=self.dim,
                                              conv_layers=self.kwargs['conv_layers_per_stage'])
                                 )
        self.target_conv = ResidualUnit(channels=self.channels,
                                        dim=self.dim,
                                        conv_layers=self.kwargs['conv_layers_per_stage'])
        conv_fn = getattr(nn, f'Conv{self.dim}d')
        self.combine_conv_target = Vmap(conv_fn(in_channels=2*self.channels,
                                                out_channels=self.channels,
                                                kernel_size=1,
                                                padding='same')
                                        )
        self.combine_conv_context = Vmap(conv_fn(in_channels=2*self.channels,
                                         out_channels=self.channels,
                                         kernel_size=1,
                                         padding='same')
                                         )

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
        # do single convs on input
        context = self.context_conv(context)  # B,L,C,...
        target = self.target_conv(target)  # B,C,...

        # concat on channels
        context_target = torch.concat(
            [context, target.unsqueeze(1).expand_as(context)], dim=2)  # B,L,2C,...

        # conv query with support
        target_update = self.combine_conv_target(context_target)  # B,L,C,...
        context_update = self.combine_conv_context(context_target)

        # average
        target_update = target_update.mean(dim=1, keepdim=False)  # B,C,...

        # resudual and activation
        target = F.gelu(target + target_update)
        context = F.gelu(context + context_update)

        # return augmented inputs
        return context, target


@dataclass(eq=False, repr=False)
class PairwiseConvAvgModelOutput(DefaultUnetOutputBlock):
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


if __name__ == '__main__':
    device = 'cuda'
    unet2d = PairwiseConvAvgModel(dim=2, stages=4, in_channels=2,
                                  out_channels=3, inner_channels=32, conv_layers_per_stage=2).to(device)
    context_in = torch.rand(7, 8, 2, 64, 64).to(device)
    context_out = torch.rand(7, 8, 3, 64, 64).to(device)
    target_in = torch.rand(7, 2, 64, 64).to(device)
    target_out = unet2d(context_in, context_out, target_in)
    print(unet2d)
    print('2d ok')

    unet3d = PairwiseConvAvgModel(dim=3,
                                  stages=4,
                                  in_channels=2,
                                  out_channels=3,
                                  inner_channels=32,
                                  conv_layers_per_stage=2).to(device)
    context_in = torch.rand(7, 8, 2, 64, 64, 32).to(device)
    context_out = torch.rand(7, 8, 3, 64, 64, 32).to(device)
    target_in = torch.rand(7, 2, 64, 64, 32).to(device)
    target_out = unet3d(context_in, context_out, target_in)
    print(unet3d)
    print('3d ok')
