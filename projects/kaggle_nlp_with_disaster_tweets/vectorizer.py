import numpy as np
import preprocessor as p
from gensim.parsing.preprocessing import remove_stopwords
import re
from vocabulary import Vocabulary, SequenceVocabulary
from collections import Counter


class TweetVectorizer(object):
    """
    The Tweet Vectorizer class which wraps the Vocabularies.
    """

    def __init__(
        self, tweet_vocab, target_vocab, token_length_cutoff=1, token_count_cutoff=4
    ):
        """
        :param tweet_vocab: Maps tweet tokens to integers.
        :param target_vocab: Maps target labels to integers.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :param token_count_cutoff: Cutoff to drop tokens with count less than
            the given value. default to 4.
        """
        self.tweet_vocab = tweet_vocab
        self.target_vocab = target_vocab
        self.token_length_cutoff = token_length_cutoff
        self.token_count_cutoff = token_count_cutoff

    @staticmethod
    def tokenizer(tweet, token_length_cutoff=1):
        """
        Tokenizes the given tweet.

        :param tweet: Tweet for tokenization.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :return: List of tokens generated from given tweet.
        """
        clean_tweet = p.clean(
            tweet.lower().replace("#", " ").replace("&amp;", " ").replace("'", "")
        )
        tweet_without_stop_words = remove_stopwords(clean_tweet)
        clean_tweet2 = re.sub(f"[^a-z0-9]", " ", tweet_without_stop_words)
        return list(
            filter(
                lambda token: len(token) > token_length_cutoff,
                map(lambda token: token.strip(), clean_tweet2.split(" ")),
            )
        )

    def vectorize(self, tweet):
        """
        Create a collapsed one-hot vector for the tweet.

        :param tweet: Tweet for vectorization.
        :return: one hot encoded vector for given tweet.
        """
        one_hot = np.zeros(len(self.tweet_vocab), dtype=np.float32)
        tokens = TweetVectorizer.tokenizer(tweet, self.token_length_cutoff)
        for token in tokens:
            token_index = self.tweet_vocab.lookup_token(token)
            one_hot[token_index] = 1
        return one_hot

    @classmethod
    def from_dataframe(cls, tweet_df, token_length_cutoff, token_count_cutoff):
        """
        Initialises the vectorizer from the dataset dataframe.

        :param tweet_df: The tweets dataset dataframe.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :param token_count_cutoff: Cutoff to drop tokens with count less than
            the given value. default to 4.
        :return: Returns the initialised vectorizer.
        """
        tweet_vocab = Vocabulary(add_unk=True)
        target_vocab = Vocabulary(add_unk=False)

        for target_class in sorted(tweet_df.target):
            target_vocab.add_token(target_class)

        token_counts = Counter()
        for tweet in tweet_df.text:
            for token in TweetVectorizer.tokenizer(tweet, token_length_cutoff):
                token_counts[token] += 1
        for token, count in token_counts.items():
            if count > token_length_cutoff:
                tweet_vocab.add_token(token=token)
        return cls(tweet_vocab, target_vocab, token_length_cutoff, token_count_cutoff)

    @classmethod
    def from_serializable(cls, contents):
        """
        Initializes a TweetVectorizer from a serializable dictionary.

        :param contents: the serializable dictionary.
        :return: Instance of TweetVectorizer class.
        """
        tweet_vocab = Vocabulary.from_serializable(contents["tweet_vocab"])
        target_vocab = Vocabulary.from_serializable(contents["target_vocab"])
        return cls(
            tweet_vocab=tweet_vocab,
            target_vocab=target_vocab,
            token_length_cutoff=contents["token_length_cutoff"],
            token_count_cutoff=contents["token_count_cutoff"],
        )

    def to_serializable(self):
        """
        Create the serializable dictionary for caching.

        :return: The serializable dictionary.
        """
        return {
            "tweet_vocab": self.tweet_vocab.to_serializable(),
            "target_vocab": self.target_vocab.to_serializable(),
            "token_length_cutoff": self.token_length_cutoff,
            "token_count_cutoff": self.token_count_cutoff,
        }


class CNNTweetVectorizer(object):
    """
    The Tweet Vectorizer class which wraps the Vocabularies. This
    is specific to CNN classifier.
    """

    def __init__(
        self,
        tweet_vocab,
        target_vocab,
        maximum_tweet_length,
        token_length_cutoff=1,
        token_count_cutoff=4,
    ):
        """
        :param tweet_vocab: Maps tweet tokens to integers.
        :param target_vocab: Maps target labels to integers.
        :param maximum_tweet_length: The length of the longest tweet.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :param token_count_cutoff: Cutoff to drop tokens with count less than
            the given value. default to 4.
        """
        self.tweet_vocab = tweet_vocab
        self.target_vocab = target_vocab
        self.token_length_cutoff = token_length_cutoff
        self.token_count_cutoff = token_count_cutoff
        self.maximum_tweet_length = maximum_tweet_length

    @staticmethod
    def tokenizer(tweet, token_length_cutoff=1):
        """
        Tokenizes the given tweet.

        :param tweet: Tweet for tokenization.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :return: List of tokens generated from given tweet.
        """
        clean_tweet = p.clean(
            tweet.lower().replace("#", " ").replace("&amp;", " ").replace("'", "")
        )
        tweet_without_stop_words = remove_stopwords(clean_tweet)
        clean_tweet2 = re.sub(f"[^a-z0-9]", " ", tweet_without_stop_words)
        return list(
            filter(
                lambda token: len(token) > token_length_cutoff,
                map(lambda token: token.strip(), clean_tweet2.split(" ")),
            )
        )

    def vectorize(self, tweet):
        """
        Create a collapsed one-hot vector for the tweet.

        :param tweet: Tweet for vectorization.
        :return: one hot encoded vector for given tweet.
        """
        one_hot_matrix_size = (len(self.tweet_vocab), self.maximum_tweet_length)
        one_hot = np.zeros(shape=one_hot_matrix_size, dtype=np.float32)
        tokens = TweetVectorizer.tokenizer(tweet, self.token_length_cutoff)
        for position_index, token in enumerate(tokens):
            token_index = self.tweet_vocab.lookup_token(token)
            one_hot[token_index][position_index] = 1
        return one_hot

    @classmethod
    def from_dataframe(cls, tweet_df, token_length_cutoff, token_count_cutoff):
        """
        Initialises the vectorizer from the dataset dataframe.

        :param tweet_df: The tweets dataset dataframe.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :param token_count_cutoff: Cutoff to drop tokens with count less than
            the given value. default to 4.
        :return: Returns the initialised vectorizer.
        """
        tweet_vocab = Vocabulary(add_unk=True)
        target_vocab = Vocabulary(add_unk=False)

        for target_class in sorted(tweet_df.target):
            target_vocab.add_token(target_class)

        token_counts = Counter()
        max_tweet_length = 0
        for tweet in tweet_df.text:
            tokenized_tweet = TweetVectorizer.tokenizer(tweet, token_length_cutoff)
            max_tweet_length = max(max_tweet_length, len(tokenized_tweet))
            for token in tokenized_tweet:
                token_counts[token] += 1
        for token, count in token_counts.items():
            if count > token_length_cutoff:
                tweet_vocab.add_token(token=token)
        return cls(
            tweet_vocab=tweet_vocab,
            target_vocab=target_vocab,
            maximum_tweet_length=max_tweet_length,
            token_length_cutoff=token_length_cutoff,
            token_count_cutoff=token_count_cutoff,
        )

    @classmethod
    def from_serializable(cls, contents):
        """
        Initializes a TweetVectorizer from a serializable dictionary.

        :param contents: the serializable dictionary.
        :return: Instance of TweetVectorizer class.
        """
        tweet_vocab = Vocabulary.from_serializable(contents["tweet_vocab"])
        target_vocab = Vocabulary.from_serializable(contents["target_vocab"])
        return cls(
            tweet_vocab=tweet_vocab,
            target_vocab=target_vocab,
            maximum_tweet_length=contents["maximum_tweet_length"],
            token_length_cutoff=contents["token_length_cutoff"],
            token_count_cutoff=contents["token_count_cutoff"],
        )

    def to_serializable(self):
        """
        Create the serializable dictionary for caching.

        :return: The serializable dictionary.
        """
        return {
            "tweet_vocab": self.tweet_vocab.to_serializable(),
            "target_vocab": self.target_vocab.to_serializable(),
            "token_length_cutoff": self.token_length_cutoff,
            "token_count_cutoff": self.token_count_cutoff,
            "maximum_tweet_length": self.maximum_tweet_length,
        }


class SequenceVectorizer(object):
    def __init__(
        self, tweet_vocab, target_vocab, token_length_cutoff=1, token_count_cutoff=4
    ):
        """
        :param tweet_vocab: Maps tweet tokens to integers.
        :param target_vocab: Maps target labels to integers.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :param token_count_cutoff: Cutoff to drop tokens with count less than
            the given value. default to 4.
        """
        self.tweet_vocab = tweet_vocab
        self.target_vocab = target_vocab
        self.token_length_cutoff = token_length_cutoff
        self.token_count_cutoff = token_count_cutoff

    @staticmethod
    def tokenizer(tweet, token_length_cutoff=1):
        """
        Tokenizes the given tweet.

        :param tweet: Tweet for tokenization.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :return: List of tokens generated from given tweet.
        """
        clean_tweet = p.clean(
            tweet.lower().replace("#", " ").replace("&amp;", " ").replace("'", "")
        )
        tweet_without_stop_words = remove_stopwords(clean_tweet)
        clean_tweet2 = re.sub(f"[^a-z0-9]", " ", tweet_without_stop_words)
        return list(
            filter(
                lambda token: len(token) > token_length_cutoff,
                map(lambda token: token.strip(), clean_tweet2.split(" ")),
            )
        )

    def vectorize(self, tweet, vector_length=-1):
        tokens = TweetVectorizer.tokenizer(tweet, self.token_length_cutoff)
        indices = [self.tweet_vocab.begin_seq_index]
        indices.extend(self.tweet_vocab.lookup_token(token) for token in tokens)
        indices.append(self.tweet_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[: len(indices)] = indices
        out_vector[len(indices) :] = self.tweet_vocab.mask_index
        return out_vector

    @classmethod
    def from_dataframe(cls, tweet_df, token_length_cutoff, token_count_cutoff):
        """
        Initialises the vectorizer from the dataset dataframe.

        :param tweet_df: The tweets dataset dataframe.
        :param token_length_cutoff: Cutoff to drop token less than the given value.
            defaults to 1.
        :param token_count_cutoff: Cutoff to drop tokens with count less than
            the given value. default to 4.
        :return: Returns the initialised vectorizer.
        """
        target_vocab = Vocabulary(add_unk=False)

        for target_class in sorted(tweet_df.target):
            target_vocab.add_token(target_class)

        tweet_vocab = SequenceVocabulary()

        token_counts = Counter()
        for tweet in tweet_df.text:
            for token in TweetVectorizer.tokenizer(tweet, token_length_cutoff):
                token_counts[token] += 1
        for token, count in token_counts.items():
            if count > token_length_cutoff:
                tweet_vocab.add_token(token=token)
        return cls(tweet_vocab, target_vocab, token_length_cutoff, token_count_cutoff)

    @classmethod
    def from_serializable(cls, contents):
        """
        Initializes a TweetVectorizer from a serializable dictionary.

        :param contents: the serializable dictionary.
        :return: Instance of TweetVectorizer class.
        """
        tweet_vocab = SequenceVocabulary.from_serializable(contents["tweet_vocab"])
        target_vocab = Vocabulary.from_serializable(contents["target_vocab"])
        return cls(
            tweet_vocab=tweet_vocab,
            target_vocab=target_vocab,
            token_length_cutoff=contents["token_length_cutoff"],
            token_count_cutoff=contents["token_count_cutoff"],
        )

    def to_serializable(self):
        """
        Create the serializable dictionary for caching.

        :return: The serializable dictionary.
        """
        return {
            "tweet_vocab": self.tweet_vocab.to_serializable(),
            "target_vocab": self.target_vocab.to_serializable(),
            "token_length_cutoff": self.token_length_cutoff,
            "token_count_cutoff": self.token_count_cutoff,
        }
