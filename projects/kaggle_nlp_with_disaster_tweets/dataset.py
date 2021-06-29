from torch.utils.data import Dataset
import json
import pandas as pd

from vectorizer import TweetVectorizer


class TweetDataset(Dataset):
    def __init__(
        self, tweet_df, vectorizer, token_length_cutoff=1, token_count_cutoff=4, use_full_dataset=False
    ):
        """
        :param tweet_df: Tweets Dataframe.
        :param vectorizer: Vectorizer instantiated from dataset.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :param token_count_cutoff: Cutoff to drop tokens with count less than
            the given value. default to 4.
        :param use_full_dataset: Boolean param to control whether to use the full dataset
            training the model or not. defaults to False.
        """
        self.tweet_df = tweet_df
        self._vectorizer = vectorizer
        self.token_length_cutoff = token_length_cutoff
        self.token_count_cutoff = token_count_cutoff

        if use_full_dataset:
            self.train_df = self.tweet_df[self.tweet_df.split == "train"]
            self.train_size = len(self.train_df)

            self.val_df = self.tweet_df[self.tweet_df.split == "random"]
            self.val_size = 0

            self.test_df = self.tweet_df[self.tweet_df.split == "random"]
            self.test_size = 0
        else:
            self.train_df = self.tweet_df[self.tweet_df.split == "train"]
            self.train_size = len(self.train_df)

            self.val_df = self.tweet_df[self.tweet_df.split == "val"]
            self.val_size = len(self.val_df)

            self.test_df = self.tweet_df[self.tweet_df.split == "test"]
            self.test_size = len(self.test_df)

        self._look_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size),
        }
        self.set_split("train")

    @classmethod
    def load_dataset_and_make_vectorizer(
        cls, tweets_csv, token_length_cutoff=1, token_count_cutoff=4, use_full_dataset=False
    ):
        """
        Load dataset and make a new vectorizer.

        :param tweets_csv: location of the dataset.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :param token_count_cutoff: Cutoff to drop tokens with count less than
            the given value. default to 4.
        :param use_full_dataset: Boolean param to control whether to use the full dataset
            training the model or not. defaults to False.
        :return: an instance of TweetDataset.
        """
        tweet_df = pd.read_csv(tweets_csv)
        train_tweet_df = tweet_df[tweet_df.split == "train"]
        return cls(
            tweet_df=tweet_df,
            vectorizer=TweetVectorizer.from_dataframe(
                train_tweet_df, token_length_cutoff=2, token_count_cutoff=4
            ),
            token_length_cutoff=token_length_cutoff,
            token_count_cutoff=token_count_cutoff,
            use_full_dataset=use_full_dataset
        )

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """
        A static method for loading the vectorizer from file.

        :param vectorizer_filepath: the location of serialized vectorizer.
        :return: An instance of TweetVectorizer.
        """
        with open(vectorizer_filepath) as file:
            return TweetVectorizer.from_serializable(json.load(file))

    def save_vectorizer(self, vectorizer_filepath):
        """
        Saves the vectorizer to disk using JSON.

        :param vectorizer_filepath: the location to save the vectorizer.
        """
        with open(vectorizer_filepath, "w") as file:
            json.dump(self._vectorizer.to_serializable(), file)

    def get_vectorizer(self):
        """
        Returns the vectorizer.
        :return: an instance of TweetVectorizer.
        """
        return self._vectorizer

    def set_split(self, split="train"):
        """
        Selects the splits in the dataset using a column in the dataset.

        :param split: one of train, val or test.
        """
        assert split in [
            "train",
            "val",
            "test",
        ], "Specify from train, val, test options."
        self._train_split = split
        self._target_df, self._target_size = self._look_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """
        The primary entry point method for PyTorch Datasets.

        :param index: The index to the data point.
        :return: A dict of the data point's feature and label.
        """
        row = self._target_df.iloc[index]
        tweet_vector = self._vectorizer.vectorize(row.text)
        target_index = self._vectorizer.target_vocab.lookup_token(row.target)
        return {"x_data": tweet_vector, "y_target": target_index}

    def get_num_batches(self, batch_size):
        """
        Given a batch size, return the number of batches in the dataset.

        :param batch_size:
        :return: Number of batches in the dataset.
        """
        return len(self) // batch_size
