from torch.utils.data import DataLoader
import torch
import numpy as np
import os


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    """
    Function to wrap PyTorch DataLoader. Ensures that each tensor
    is on the write device location.

    :param dataset: Dataset Object.
    :param batch_size: value to split the dataset into smaller batches.
    :param shuffle: Boolean param to control the shuffle behaviour while
        splitting the data. Defaults to True
    :param drop_last: Weather to drop last item or not. Defaults to True
    :param device: Device to load the tensor to. default to CPU
    :return: Iterator object
    """
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


# Helpers
def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def make_train_state(args):
    return {
        "epoch_index": 0,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": -1,
        "test_acc": -1,
        "model_filename": args.model_state_file,
        "early_stopping_best_val": 1e8,
        "stop_early": False,
        "early_stopping_step": 0,
        "learning_rate": args.learning_rate,
    }


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """
    # Save one model at least
    if train_state["epoch_index"] == 0:
        torch.save(model.state_dict(), train_state["model_filename"])
        train_state["stop_early"] = False

    # Save model if performance improved
    elif train_state["epoch_index"] >= 1:
        loss_tm1, loss_t = train_state["val_loss"][-2:]

        # If loss worsened
        if loss_t >= train_state["early_stopping_best_val"]:
            # Update step
            train_state["early_stopping_step"] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state["early_stopping_best_val"]:
                torch.save(model.state_dict(), train_state["model_filename"])

            # Reset early stopping step
            train_state["early_stopping_step"] = 0

        # Stop early ?
        train_state["stop_early"] = (
            train_state["early_stopping_step"] >= args.early_stopping_criteria
        )

    return train_state


if __name__ == "__main__":
    from dataset import TweetDataset
    from vectorizer import TweetVectorizer

    tweet_dataset = TweetDataset.load_dataset_and_make_vectorizer(
        "data/train_with_splits.csv"
    )
    print(
        f"Tweet Vocab Len: {len(tweet_dataset._vectorizer.tweet_vocab)}",
        f"Target Vocab Len: {len(tweet_dataset._vectorizer.target_vocab)}",
        f"Number of batches: {tweet_dataset.get_num_batches(10)}",
    )
    print(tweet_dataset._vectorizer.target_vocab._token_to_idx)
    print("Index for 0 target:", tweet_dataset._vectorizer.target_vocab.lookup_token(0))
    print("Index for 1 target:", tweet_dataset._vectorizer.target_vocab.lookup_token(1))
    batches = generate_batches(tweet_dataset, 10)
    for batch in batches:
        print(batch["x_data"].shape, batch["x_data"])
        print(batch["y_target"].shape, batch["y_target"])
        break
    tweet = "The Campaign: Will Ferrell and Zach Galifianakis commit comic mayhem in this hilarious political farce. 4* http://t.co/tQ3j2qGtZQ"
    one_hot = tweet_dataset._vectorizer.vectorize(tweet)
    tokens = TweetVectorizer.tokenizer(tweet)
    for token in tokens:
        try:
            token_index = tweet_dataset._vectorizer.tweet_vocab.lookup_token(token)
            print(
                f"Token:{token}, "
                f"Index:{token_index}, "
                f"One Hot Value:{one_hot[token_index]}"
            )
        except KeyError:
            print(f"{token} is not found in vocab.")
