import torch
import torch.nn as nn
import torch.nn.functional as F


def column_gather(y_out, x_lengths):
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(y_out[batch_index, column_index])
    return torch.stack(out)


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


class TweetCNNClassifier(nn.Module):
    """
    CNN based Tweet Clasfifier.
    """

    def __init__(self, initial_num_channels, output_dim, num_channels):
        """
        :param initial_num_channels: Size of the input feature vector.
        :param output_dim: size of the output prediction vector.
        :param num_channels: constant channel size to use throughout the network.
        """
        super(TweetCNNClassifier, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(
                in_channels=initial_num_channels,
                out_channels=num_channels,
                kernel_size=3,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels, out_channels=num_channels, kernel_size=3
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels, out_channels=num_channels, kernel_size=2
            ),
            nn.ELU(),
        )
        self.fc = nn.Linear(num_channels, output_dim)

    def forward(self, x_in, apply_sigmoid=False):
        """
        The forward pass of the classifier.

        :param x_in: an input data tensor.
            X_in.shape should be (batch, initial_num_channels).
        :param apply_sigmoid: a flag for the sigmoid activation.
            It should be false if used with the cross-entropy losses.
        :return: the resulting tensor. Shape should be (batch, num_classes).
        """
        features = self.convnet(x_in).squeeze(dim=2)
        prediction_vector = self.fc(features).squeeze()
        if apply_sigmoid:
            prediction_vector = F.sigmoid(prediction_vector)
        return prediction_vector


class TweetEmbeddingClassifier(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_embeddings,
        num_channels,
        hidden_dim,
        output_dim,
        dropout_p,
        pretrained_embeddings=None,
        padding_idx=0,
    ):
        super(TweetEmbeddingClassifier, self).__init__()
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(
                embedding_dim=embedding_size,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
            )
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(
                embedding_dim=embedding_size,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
                _weight=pretrained_embeddings,
            )
        self.convnet = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_size, out_channels=num_channels, kernel_size=3
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels, out_channels=num_channels, kernel_size=3
            ),
            nn.ELU(),
        )
        self._dropout_p = dropout_p
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_sigmoid=False):
        x_embedded = self.emb(x_in).permute(0, 2, 1)
        features = self.convnet(x_embedded)

        remaining_size = features.size(dim=2)
        features = F.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self._dropout_p)

        intermediate_vector = F.relu(F.dropout(self.fc1(features), p=self._dropout_p))
        prediction_vector = self.fc2(intermediate_vector).squeeze()
        if apply_sigmoid:
            prediction_vector = F.sigmoid(prediction_vector)
        return prediction_vector


class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(ElmanRNN, self).__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.batch_first = batch_first
        self.hidden_size = hidden_size

    def _initial_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))

    def forward(self, x_in, initial_hidden=None):
        if self.batch_first:
            batch_size, seq_size, feat_size = x_in.size()
            x_in = x_in.permute(1, 0, 2)
        else:
            seq_size, batch_size, feat_size = x_in.size()
        hiddens = []
        if initial_hidden is None:
            initial_hidden = self._initial_hidden(batch_size)
            initial_hidden = initial_hidden.to(x_in.device)
        hidden_t = initial_hidden
        for t in range(seq_size):
            hidden_t = self.rnn_cell(x_in[t], hidden_t)
            hiddens.append(hidden_t)
        hiddens = torch.stack(hiddens)
        if self.batch_first:
            hiddens = hiddens.permute(1, 0, 2)
        return hiddens


class TweetSimpleRNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_embeddings,
        output_dim,
        rnn_hidden_size,
        batch_first=True,
        padding_idx=0,
        pretrained_embeddings=None,
    ):
        super(TweetSimpleRNNClassifier, self).__init__()
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(
                embedding_dim=embedding_size,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
            )
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(
                embedding_dim=embedding_size,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
                _weight=pretrained_embeddings,
            )
        self.rnn = ElmanRNN(
            input_size=embedding_size,
            hidden_size=rnn_hidden_size,
            batch_first=batch_first,
        )
        self.fc1 = nn.Linear(in_features=rnn_hidden_size, out_features=rnn_hidden_size)
        self.fc2 = nn.Linear(in_features=rnn_hidden_size, out_features=output_dim)

    def forward(self, x_in, x_lengths=None, apply_sigmoid=False):
        x_embedded = self.emb(x_in)
        y_out = self.rnn(x_embedded)
        if x_lengths is not None:
            y_out = column_gather(y_out, x_lengths)
        else:
            y_out = y_out[:, -1, :]
        y_out = F.relu(self.fc1(F.dropout(y_out, 0.5)))
        prediction_vector = self.fc2(F.dropout(y_out, 0.5)).squeeze()
        if apply_sigmoid:
            prediction_vector = F.sigmoid(prediction_vector)
        return prediction_vector
