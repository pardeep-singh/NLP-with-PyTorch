class Vocabulary(object):
    """
    Class to process tweets and extract Vocabulary.
    """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """
        Returns a dictionary that can be serialized:
        :return: Dictionary containing token_to_idk, add_unk and
        unk_token keys.
        """
        return {
            "token_to_idx": self._token_to_idx,
            "add_unk": self._add_unk,
            "unk_token": self._unk_token,
        }

    @classmethod
    def from_serializable(cls, contents):
        """
        Instantiates the Vocabulary from a serialized dictionary.

        :param contents: contents previously created vocabulary containing
        token_to_idk, add_unk and unk_token keys.
        :return: Returns the Vocabulary object initialized with given contents.
        """
        return cls(**contents)

    def add_token(self, token):
        """
        Add token to the vocab.

        :param token: Token to be added to the vocab.
        :return: Returns the index of added token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        """
        Perform lookup using given token. Use UNK token if given
        token is not found.

        :param token: Token to perform lookup. Uses UNK token for unknown
        tokens.
        :return: Returns the index of given token if found in the vocab.
        Otherwise return index of UNK token.
        """
        if self._add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """
        Performs index lookup using given index.

        :param index: Index to perform lookup.
        :return: Returns the token for given index.
        :raises: KeyError if the index is not found in the vocabulary.
        """
        if index not in self._idx_to_token:
            raise KeyError(f"The index {index} is not found in Vocabulary.")
        return self._idx_to_token[index]

    def __str__(self):
        return f"<Vocabulary(size={len(self)}>"

    def __len__(self):
        return len(self._token_to_idx)
