import torch.nn as nn
import torch.nn.functional as F


class TweetPerceptronClassifier(nn.Module):
    """
    A Simple Perceptron based Tweet Classifier.
    """
    def __init__(self, num_features):
        """
        :param num_features: the size of the input features vector.
        """
        super(TweetPerceptronClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        """
        The forward pass of the classifier.

        :param x_in: an input data tensor x_in.shape should be
            (batch, num_features).
        :param apply_sigmoid: a flag for the sigmoid activation.
            It should be false if used with the cross-entropy losses.
        :return: The resulting tensor. tensor.shape should be (batch,).
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)
        return y_out
