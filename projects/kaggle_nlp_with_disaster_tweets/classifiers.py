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


class TweetMLPClassifier(nn.Module):
    """
    A 2-layer Multilayer Perceptron classifying Tweets.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        :param input_dim: the size of the input vectors.
        :param hidden_dim: the output size of the first Linear Layer.
        :param output_dim: the output size of the second Linear Layer.
        """
        super(TweetMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_sigmoid=False):
        """
        The forward pass of the classifier.

        :param x_in: an input data tensor.
            X_in.shape should be (batch, input_dim).
        :param apply_sigmoid: a flag for the sigmoid activation.
            It should be false if used with the cross-entropy losses.
        :return: the resulting tensor. Shape should be (batch, output_dim).
        """
        intermediate_vector = F.relu(self.fc1(x_in))
        prediction_vector = self.fc2(intermediate_vector).squeeze()
        if apply_sigmoid:
            prediction_vector = F.sigmoid(prediction_vector)
        return prediction_vector


class TweetMLPClassifier1(nn.Module):
    """
    A 3-layer Multilayer Perceptron classifying Tweets.
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        """
        :param input_dim: the size of the input vectors.
        :param hidden_dim1: the output size of the first Linear Layer.
        :param hidden_dim2: the output size of the second Linear Layer.
        :param output_dim: the output size of the second Linear Layer.
        """
        super(TweetMLPClassifier1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x_in, apply_sigmoid=False):
        """
        The forward pass of the classifier.

        :param x_in: an input data tensor.
            X_in.shape should be (batch, input_dim).
        :param apply_sigmoid: a flag for the sigmoid activation.
            It should be false if used with the cross-entropy losses.
        :return: the resulting tensor. Shape should be (batch, output_dim).
        """
        intermediate_vector1 = F.relu(self.fc1(x_in))
        intermediate_vector2 = F.relu(self.fc2(intermediate_vector1))
        prediction_vector = self.fc3(intermediate_vector2).squeeze()
        if apply_sigmoid:
            prediction_vector = F.sigmoid(prediction_vector)
        return prediction_vector
