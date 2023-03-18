import torch
import torch.nn as nn
import numpy as np
import kornia
from ..util.shapecheck import ShapeChecker


class TaskSpecificActivationFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self.dummy_param.device

    @property
    def dtype(self):
        return self.dummy_param.dtype

    def forward(self, y_pred_no_activation, x, activation):
        sc = ShapeChecker()
        sc.check(y_pred_no_activation, "B Cout H W D")
        sc.check(x, "B Cin H W D")

        # since we train with small batches (or no batches at all!), we iterate over batch
        y_pred = []
        for i in range(len(x)):
            y_pred.append(self.apply_activation_fun(
                activation[i], x[[i]], y_pred_no_activation[[i]]))

        y_pred = torch.cat(y_pred, dim=0)
        sc.check(y_pred, "B Cout H W D")

        return y_pred

    def apply_activation_fun(self, activation_fun, x, y_pred_no_activation):
        """Applies the loss and activation functions

        Args:
            lossfun (str): The loss function to apply
            y_pred_no_activation (torch.Tensor): Tensor 1xCxHxWxD

        Returns:
            torch.Tensor:: Tensor 1xCxHxWxD
        """
        if activation_fun == 'none':
            return y_pred_no_activation
        elif activation_fun == 'sigmoid':
            return self.sigmoid(y_pred_no_activation)
        elif activation_fun == 'sigmoid+threshold':
            return self.sigmoid_thresholded(y_pred_no_activation)

    def sigmoid(self, y_pred_no_activation):
        return torch.sigmoid(y_pred_no_activation)

    def sigmoid_thresholded(self, y_pred_no_activation):
        # sigmoid and thresholding
        return torch.sigmoid(y_pred_no_activation).round()

    def deformable_registration(self, y_pred_no_activation, x):
        # we interpret the prediction as a 2d displacement field. We add an empty 3rd channel to use the torchreg package.
        transform = torch.cat(
            [y_pred_no_activation, torch.zeros_like(y_pred_no_activation)], dim=1)[:, :3].unsqueeze(-1)
        # first channel of the input is the moving image
        moving = x[:, [0]]

        # the torchreg package expects a dummy 3rd channel.
        morphed = self.transformer(moving.unsqueeze(-1), transform).squeeze(-1)

        # expand morphed back to 2 channels
        zeros = torch.zeros_like(x)
        zeros[:, [0]] = morphed
        return zeros


class TaskSpecificLossModule(nn.Module):

    def __init__(self, lam=0.2, ncc_win_size=9):
        """_summary_

        Args:
            lam (_type_): regularization for the deformable registration
            ncc_win_size (_type_): NCC window size
        """
        super().__init__()
        self.lam = lam
        self.dummy_param = nn.Parameter(torch.empty(0))

        self.activation_function = TaskSpecificActivationFunction()
        self.bce = nn.BCELoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
        self.ncc = NCC(window=ncc_win_size, reduction='mean')
        self.ssim = kornia.metrics.SSIM(
            window_size=11, max_val=1.0, eps=1e-12, padding='same')

    @property
    def device(self):
        return self.dummy_param.device

    @property
    def dtype(self):
        return self.dummy_param.dtype

    def forward(self, x, y_pred, y_true, lossfun):
        """Calculates the loss. Expects batched inputs, with potentially different loss functions applied to each member of the batch.

        Args:
            y_pred (torch.Tensor): The prediction, without activation function
            y_true (torch.Tensor): The target
            lossfun (list[str]): List of loss functions to applied to each member of the batch.
        """
        sc = ShapeChecker()
        sc.check(y_pred, "B Cout H W D")
        sc.check(y_true, "B Cout H W D")
        sc.check(x, "B Cin H W D")

        # since we train with small batches, iterating over batch is ok
        losses = []
        for i in range(len(x)):
            # record losses
            losses.append(self.apply_loss(
                lossfun[i], x[[i]], y_pred[[i]], y_true[[i]]).view(1))

        # concat losses
        losses = torch.cat(losses, dim=0)
        sc.check(losses, "B")

        # log loss
        loss = losses.mean()
        # record loss per task
        return loss

    def apply_loss(self, lossfun, x, y_pred, y_true):
        """Applies the loss function

        Args:
            lossfun (str): The loss function to apply
            x (torch.Tensor): Tensor 1xCxHxWxD
            y_pred (torch.Tensor): Tensor 1xCxHxWxD
            y_true (torch.Tensor): Tensor 1xCxHxWxD

        Returns:
            torch.Tensor:: skalar loss item
        """
        if lossfun == 'none':
            return torch.tensor(0., dtype=x.dtype, device=x.device)
        elif lossfun == 'MSE':
            return self.MSE(y_pred, y_true)
        elif lossfun == 'SSIM':
            return self.SSIM(y_pred, y_true)
        elif lossfun == 'neg_SSIM':
            return self.neg_SSIM(y_pred, y_true)
        elif lossfun == 'PSNR':
            return self.PSNR(y_pred, y_true)
        elif lossfun == 'SoftDiceLoss':
            return self.SoftDiceLoss(y_pred, y_true)
        elif lossfun == 'DiceCoefficient':
            return self.DiceCoefficient(y_pred, y_true)
        elif lossfun == 'NCC':
            return self.NCC(y_pred, y_true)

    def MSE(self, y_pred, y_true):
        # estimated variance of the image
        var = 0.05
        return 1 / (2*var) * self.mse(y_pred, y_true)

    def NCC(self, y_pred, y_true):
        return self.ncc(y_true, y_pred)

    def neg_SSIM(self, y_pred, y_true):
        return -self.SSIM(y_pred, y_true)

    def SSIM(self, y_pred, y_true):
        # SSIM is high when images are similar, low when they are dissimilar
        assert y_pred.shape[-1] == 1, "SSIM only supports 2d"

        # sigmoid activation
        loss = self.ssim(y_true.squeeze(-1), y_pred.squeeze(-1))
        # mean, as this is not inbuild into kornia functions
        return loss.mean()

    def PSNR(self, y_pred, y_true):
        # PSNR is high when images are similar, low when they are dissimilar

        loss = kornia.metrics.psnr(y_true, y_pred, max_val=1)
        # mean, as this is not inbuild into kornia functions
        return loss.mean()

    def BCE(self, y_pred, y_true):
        # sigmoid activation is build-in
        return self.bce(y_pred, y_true)

    def SoftDiceLoss(self, y_pred, y_true):
        """ dice loss as proposed by the paper "V-Net: Fully Convolutional Neural 
        Networks for Volumetric Medical Image Segmentation"

        Applies a sigmoid activation function and re-scales the loss for minimization over a 0..1 interval
        """

        # per-batch member computation
        orig_shape = y_pred.shape
        b_size = orig_shape[0]
        smooth = 1.
        y_pred = y_pred.view(b_size, -1)
        y_true = y_true.view(b_size, -1)
        intersection = (y_pred * y_true).sum(dim=1)
        union = (y_pred**2).sum(dim=1) + (y_true**2).sum(dim=1)
        dice_loss = 1 - ((2. * intersection + smooth) / (union + smooth))

        return dice_loss

    def DiceCoefficient(self, y_pred, y_true):
        # threshold
        y_pred = torch.where(y_pred > 0.5, 1., 0.)

        # per-batch member computation
        orig_shape = y_pred.shape
        b_size = orig_shape[0]
        y_pred = y_pred.view(b_size, -1)
        y_true = y_true.view(b_size, -1)
        intersection = (y_pred * y_true).sum(dim=1)
        union = y_pred.sum(dim=1) + y_true.sum(dim=1)
        if union.item() == 0:
            dice_coeff = torch.ones((), device=union.device, dtype=union.dtype)
        else:
            dice_coeff = 2. * intersection / union

        return dice_coeff


class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss.

    We follow the NCC definition from the paper "VoxelMorph: A Learning Framework for Deformable Medical Image Registration",
    which implements it via the coefficient of determination (R2 score). 
    This is strictly the squared normalized cross-correlation, or squared cosine similarity.

    NCC over two image pacthes I, J of size N is calculated as
    NCC(I, J) = 1/N * [sum_n=1^N (I_n - mean(I)) * (J_n - mean(J))]^2 / [var(I) * var(J)]

    The output is rescaled for minimization within the 0..1 interval.

    """

    def __init__(self, window=5, reduction='mean'):
        super().__init__()
        self.win = window
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        def debug_check_for_nagtive_var(var, y_true, y_pred):
            if (var < 0).any():
                # save inputs to disk
                torch.save(y_true, "y_true.pt")
                torch.save(y_pred, "y_true.pt")
                raise Exception(
                    'Negative variance detected! saved input tensors to disk.')

        def compute_local_sums(I, J):
            # calculate squared images
            I2 = I * I
            J2 = J * J
            IJ = I * J

            # take sums
            I_sum = conv_fn(I, filt, stride=stride, padding=padding)
            J_sum = conv_fn(J, filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

            # take means
            win_size = np.prod(filt.shape)
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            # calculate cross corr components
            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
            debug_check_for_nagtive_var(I_var, y_true, y_pred)
            debug_check_for_nagtive_var(J_var, y_true, y_pred)

            return I_var, J_var, cross

        # get dimension of volume
        b, c, h, w, d = y_true.shape

        # set filter
        filt = torch.ones(c, c, self.win, self.win,
                          self.win if d > 1 else 1).type_as(y_true)

        # get convolution function
        conv_fn = nn.functional.conv3d
        stride = 1
        padding = self.win // 2

        # calculate cc
        var0, var1, cross = compute_local_sums(y_true, y_pred)
        cc = cross * cross / (var0 * var1 + 1e-5)

        # invert for minimization, rescale to 0..2 interval
        cc = -cc + 1

        # apply reduction
        if self.reduction == 'mean':
            return cc.mean()
        elif self.reduction == 'sum':
            return cc.sum()
        else:
            return cc


if __name__ == '__main__':
    import neuralizer.util.utils as utils
    from argparse import Namespace
    hparams = Namespace()
    hparams.batch_size = 64
    hparams.data_slice_only = True
    hparams.modalities = 'all'
    hparams.tasks = ['seg']
    hparams.nb_examples = 1
    datamodule = utils.load_datamodule_from_hparams(hparams)
    batch = next(iter(datamodule.train_dataloader()))
    x, y_true, support_x, support_y, task, modality, lossfun = batch

    loss_module = TaskSpecificLossModule()
    print(lossfun)
    loss = loss_module.forward(x, support_y[:, 0], y_true, lossfun)
    print(loss)
