from typing import *
from .vmap import Vmap, vmap
import torch
import torch.nn as nn
from collections.abc import Iterable
from pydantic import validate_arguments
from dataclasses import dataclass, field


@dataclass(eq=False, repr=False)
class DefaultUnetStageBlock(nn.Module):
    channels: int
    kwargs: Optional[Dict[str, Any]]
    dim: Literal[2, 3] = 2

    def __post_init__(self):
        super().__init__()

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
        return context, target


@dataclass(eq=False, repr=False)
class DefaultUnetOutputBlock(nn.Module):
    """
    U-net output block. Reduces channels to out_channels. Can be used to apply additional smoothing.
    """

    in_channels: int
    out_channels: int
    kwargs: Optional[Dict[str, Any]]
    dim: Literal[2, 3] = 2

    def __post_init__(self):
        super().__init__()

        conv_fn = getattr(nn, f"Conv{self.dim}d")

        self.block = nn.Sequential(
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


@dataclass(eq=False, repr=False)
class UnetBackbone(nn.Module):
    """
    Customizable U-Net class. Implement individual stages by passing an 
    implementation of the DefaultUnetStageBlock and DefaultUnetOutputBlock classes.
    This class takes care of channels, shortcuts, and up/downsampling.
    This class uses exclusively 1x1 convolutions without activation functions for channel mapping.

    Structure, channels, size

        in_channels, size H
    Embedding
        inner_channels[0], size H
    UnetStageBlock
        inner_channels[0], size H
    UnetDownsampleAndCreateShortcutBlock
        inner_channels[1], size H/2
    ...
        inner_channels[1], size H/2
    UnetUpsampleAndConcatShortcutBlock
        inner_channels[0], size H
    UnetStageBlock
        inner_channels[0], size H
    UnetOutputBlock
        out_channels, size H


    Args:
        nn (_type_): _description_
    """
    stages: int
    in_channels: int
    out_channels: int
    inner_channels: Union[int, List[int]]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dim: Literal[2, 3] = 2
    unet_block_cls: nn.Module = DefaultUnetStageBlock
    output_block_cls: nn.Module = DefaultUnetOutputBlock
    context_filled: bool = True

    def __post_init__(self):
        super().__init__()

        self.inner_channels = self.parse_channels(self.inner_channels)
        assert len(self.inner_channels) == self.stages
        self._build()

    @validate_arguments
    def parse_channels(self, inner_channels: Union[int, List[int]]) -> List[int]:
        if isinstance(inner_channels, int):
            # single int given. Expand to Iterable over stages.
            return [inner_channels] * self.stages
        elif isinstance(inner_channels, Iterable):
            if len(inner_channels) == 1:
                # single int given as Iterable. Expand to Iterable over stages.
                return inner_channels * self.stages
            else:
                return inner_channels

    def _build(self):
        conv_fn = getattr(nn, f"Conv{self.dim}d")
        self.target_embedding = conv_fn(in_channels=self.in_channels,
                                        out_channels=self.inner_channels[0],
                                        kernel_size=1,
                                        padding='same')
        if self.context_filled:
            self.context_embedding = Vmap(conv_fn(in_channels=self.in_channels+self.out_channels,
                                                  out_channels=self.inner_channels[0],
                                                  kernel_size=1,
                                                  padding='same')
                                          )

        self.enc_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.output_block = self.output_block_cls(in_channels=self.inner_channels[0],
                                                  out_channels=self.out_channels,
                                                  dim=self.dim,
                                                  kwargs=self.kwargs)

        for i in range(self.stages):
            self.enc_blocks.append(
                self.unet_block_cls(channels=self.inner_channels[i],
                                    dim=self.dim,
                                    kwargs=self.kwargs)
            )
            if i < self.stages - 1:
                self.dec_blocks.append(
                    self.unet_block_cls(channels=self.inner_channels[-(i+2)],
                                        dim=self.dim,
                                        kwargs=self.kwargs)
                )
                self.downsample_blocks.append(
                    UnetDownsampleAndCreateShortcutBlock(
                        in_channels=self.inner_channels[i],
                        out_channels=self.inner_channels[i+1],
                        dim=self.dim,
                        context_filled=self.context_filled,
                    )
                )
                self.upsample_blocks.append(
                    UnetUpsampleAndConcatShortcutBlock(
                        in_channels=self.inner_channels[-(i+1)],
                        in_shortcut_channels=self.inner_channels[-(i+2)],
                        out_channels=self.inner_channels[-(i+2)],
                        dim=self.dim,
                        context_filled=self.context_filled,
                    )
                )

    def forward(self,
                context_in: torch.Tensor,
                context_out: torch.Tensor,
                target_in: torch.Tensor,
                ) -> torch.Tensor:
        """Performs the forward pass

        Args:
            context_in (torch.Tensor): Context input, shape BxCinxHxW or BxCinxHxWxD
            context_out (torch.Tensor): Context output, shape BxCoutxHxW or BxCoutxHxWxD
            target_in (torch.Tensor): Target input, shape BxCinxHxW or BxCinxHxWxD

        Returns:
            torch.Tensor: Target output, shape BxCoutxHxW or BxCoutxHxWxD
        """
        # concat context
        context = torch.cat([context_in, context_out],
                            dim=2)if self.context_filled else context_in  # BLCHW
        # apply input embeddings
        context = self.context_embedding(
            context) if self.context_filled else context  # BCHW1
        target = self.target_embedding(target_in)  # BCHWL

        # feed through encoder
        shortcuts = []
        for i in range(self.stages):
            # run block
            context, target = self.enc_blocks[i](context, target)

            # downsample
            if i < self.stages - 1:
                context, target, shortcut = self.downsample_blocks[i](
                    context, target)
                shortcuts.append(shortcut)

        # feed through decoder
        shortcuts = shortcuts[::-1]
        for i in range(self.stages - 1):
            # upsample and add shortcut
            context, target = self.upsample_blocks[i](
                context, target, shortcuts[i])

            # run block
            context, target = self.dec_blocks[i](context, target)

        # apply output block
        return self.output_block(target)


@dataclass(eq=False, repr=False)
class UnetDownsampleAndCreateShortcutBlock(nn.Module):
    in_channels: int
    out_channels: int
    dim: Literal[2, 3]
    context_filled: bool = True

    def __post_init__(self):
        super().__init__()
        self.needs_channel_asjustment = self.in_channels != self.out_channels
        if self.needs_channel_asjustment:
            conv_fn = getattr(nn, f"Conv{self.dim}d")
            self.context_linear_layer = Vmap(conv_fn(in_channels=self.in_channels,
                                                     out_channels=self.out_channels,
                                                     kernel_size=4,
                                                     stride=2,
                                                     padding=1)
                                             ) if self.context_Filled else None
            self.target_linear_layer = conv_fn(in_channels=self.in_channels,
                                               out_channels=self.out_channels,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1)
        else:
            pool_fn = getattr(nn, f"MaxPool{self.dim}d")
            self.context_pooling_layer = Vmap(
                pool_fn(kernel_size=2)) if self.context_filled else None
            self.target_pooling_layer = pool_fn(kernel_size=2)

    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # make shortcut
        shortcut = (context, target)
        # downsample
        if self.needs_channel_asjustment:
            context = self.context_linear_layer(
                context) if self.context_filled else context
            target = self.target_linear_layer(target)
        else:
            context = self.context_pooling_layer(
                context) if self.context_filled else context
            target = self.target_pooling_layer(target)

        return context, target, shortcut


@dataclass(eq=False, repr=False)
class UnetUpsampleAndConcatShortcutBlock(nn.Module):
    in_channels: int
    in_shortcut_channels: int
    out_channels: int
    dim: Literal[2, 3]
    context_filled: bool = True

    def __post_init__(self):
        super().__init__()
        self.upsampling_layer = nn.Upsample(
            scale_factor=2, mode='trilinear' if self.dim == 3 else 'bilinear', align_corners=False)

        conv_fn = getattr(nn, f"Conv{self.dim}d")
        self.context_conv_layer = Vmap(conv_fn(in_channels=self.in_channels + self.in_shortcut_channels,
                                               out_channels=self.out_channels,
                                               kernel_size=1,
                                               padding='same')
                                       ) if self.context_filled else None
        self.target_conv_layer = conv_fn(in_channels=self.in_channels + self.in_shortcut_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=1,
                                         padding='same')

    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor,
                shortcut: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # upsample
        context = vmap(self.upsampling_layer,
                       context) if self.context_filled else context
        target = self.upsampling_layer(target)

        # concat with shortcut
        ctx_short, tgt_short = shortcut
        # B L C ...
        context = torch.cat([context, ctx_short],
                            dim=2) if self.context_filled else context
        target = torch.cat([target, tgt_short], dim=1)  # B C ...

        # reduce dim
        context = self.context_conv_layer(
            context) if self.context_filled else context
        target = self.target_conv_layer(target)

        return context, target


if __name__ == '__main__':
    device = 'cuda'
    unet2d = UnetBackbone(dim=2, stages=4, in_channels=2,
                          out_channels=3, inner_channels=32).to(device)
    context_in = torch.rand(7, 8, 2, 64, 64).to(device)
    context_out = torch.rand(7, 8, 3, 64, 64).to(device)
    target_in = torch.rand(7, 2, 64, 64).to(device)
    target_out = unet2d(context_in, context_out, target_in)
    print(unet2d)
    print('2d ok')

    unet3d = UnetBackbone(dim=3, stages=4, in_channels=2,
                          out_channels=3, inner_channels=32).to(device)
    context_in = torch.rand(7, 8, 2, 64, 64, 32).to(device)
    context_out = torch.rand(7, 8, 3, 64, 64, 32).to(device)
    target_in = torch.rand(7, 2, 64, 64, 32).to(device)
    target_out = unet3d(context_in, context_out, target_in)
    print(unet3d)
    print('3d ok')
