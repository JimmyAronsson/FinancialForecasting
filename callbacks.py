import pytorch_lightning as pl


class FFCallback(pl.Callback):
    def __init__(self):
        super(FFCallback, self).__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # print("Train loss: ",  outputs["loss"].item())
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # print("Validation loss: ",  outputs["loss"].item())
        pass
