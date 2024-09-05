import torch.nn as nn
import torch

from convolution import CNNBackbone
from lstm import LSTMBackbone



def load_backbone_from_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')).state_dict(), strict = False)
    return model


class Classifier(nn.Module):
    def __init__(self, backbone, num_classes, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        num_classes (int): The number of classes
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Classifier, self).__init__()

        self.backbone = backbone # An LSTMBackbone or CNNBackbone

        # Load parameters from a checkpoint.
        # We do not freeze these parameters.
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )

        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()  # Loss function for classification

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        logits = self.output_layer(feats)
        loss = self.criterion(logits, targets)
        return loss, logits


class Regressor(nn.Module):
    def __init__(self, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Regressor, self).__init__()

        self.backbone = backbone  # An LSTMBackbone or CNNBackbone

        # Load parameters from a checkpoint
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )

            # We decided to not freeze the parameters because
            # we got better performance this way.
            # The code below is not needed but ok.
            for w in self.backbone.parameters():
                w.requires_grad = True

        print(self.backbone)

        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, 1)
        self.criterion = nn.MSELoss()  # Loss function for regression

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        out = self.output_layer(feats)
        out = out.squeeze()
        loss = self.criterion(out.float(), targets.float())
        return loss, out

def my_loss_function(targets, logits):
        """
        takes as input logits and targets and calculates the sum of the loss (MSE) per task 
        (valence, energy and danceability)
        """
        total_loss = 0

        # i = 1 (valence), 2 (energy), 3 (danceability)
        for i in range(3): # for each task
            total_loss += nn.MSELoss()(targets[:, i], logits[:, i+1])

        return total_loss
    
class MultitaskRegressor(nn.Module):
    def __init__(self, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(MultitaskRegressor, self).__init__()

        self.backbone = backbone # An LSTMBackbone or CNNBackbone

        # Load parameters from a checkpoint.
        # We do not freeze these parameters.
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )

        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, 3)
        self.criterion = my_loss_function  # Loss function for classification


    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        logits = self.output_layer(feats)
        loss = self.criterion(logits.float(), targets.float())
        return loss, logits
