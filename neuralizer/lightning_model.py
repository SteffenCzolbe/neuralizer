import pytorch_lightning as pl
from .models.pairwise_conv_avg_model import PairwiseConvAvgModel
from .util.shapecheck import ShapeChecker


class LightningModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        super().__init__()
        # save hparams / load hparams
        self.save_hyperparameters(hparams)

        # build model
        self.net = PairwiseConvAvgModel(dim=2 if self.hparams.data_slice_only else 3,
                                        stages=self.hparams.nb_levels,
                                        in_channels=3,
                                        out_channels=3,
                                        inner_channels=self.hparams.nb_inner_channels,
                                        conv_layers_per_stage=self.hparams.nb_conv_layers_per_stage)

    def forward(self, target_in, context_in, context_out):
        sc = ShapeChecker()
        sc.check(target_in, "B C H W", C=3, H=192, W=192)
        sc.check(context_in, "B L C H W")
        sc.check(context_out, "B L C H W")

        # run network
        y_pred = self.net(context_in, context_out, target_in)
        sc.check(y_pred, "B C H W")

        return y_pred
