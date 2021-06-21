from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from tqdm import notebook
import torch.nn.functional as F


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


def train_model(classifier, loss_func, optimizer, scheduler, dataset, args):
    classifier = classifier.to(args.device)
    train_state = make_train_state(args)
    epoch_bar = notebook.tqdm(
        desc="Training Routine", total=args.num_epochs, position=0
    )
    dataset.set_split("train")
    train_bar = notebook.tqdm(
        desc="split=train",
        total=dataset.get_num_batches(args.batch_size),
        position=1,
        leave=True,
    )
    dataset.set_split("val")
    val_bar = notebook.tqdm(
        desc="split=val",
        total=dataset.get_num_batches(args.batch_size),
        position=1,
        leave=True,
    )

    for epoch_index in range(args.num_epochs):
        train_state["epoch_index"] = epoch_index
        # Iterate Over Training Dataset
        # Setup: Batch Generator, set loss & acc to 0, set train mode on
        dataset.set_split("train")
        batch_generator = generate_batches(
            dataset, batch_size=args.batch_size, device=args.device
        )
        training_running_loss, training_running_acc = 0.0, 0.0
        classifier.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            # 5 Step Training Routine

            # Step 1. Zero the Gradients
            optimizer.zero_grad()

            # Step 2. Compute the gradients
            y_pred = classifier(x_in=batch_dict["x_data"].float())

            # Step 3. Compute the Output
            loss = loss_func(y_pred, batch_dict["y_target"].float())
            loss_batch = loss.item()
            training_running_loss += (loss_batch - training_running_loss) / (
                batch_index + 1
            )

            # Step 4. Use loss to produce gradients
            loss.backward()

            # Step 5. Use Optimizer to take gradient step
            optimizer.step()

            # Compute the accuracy
            acc_batch = compute_accuracy(y_pred, batch_dict["y_target"])
            training_running_acc += (acc_batch - training_running_acc) / (
                batch_index + 1
            )

            # Update the bar
            train_bar.set_postfix(
                loss=training_running_loss, acc=training_running_acc, epoch=epoch_index
            )
            train_bar.update()
        train_state["train_loss"].append(training_running_loss)
        train_state["train_acc"].append(training_running_acc)

        # Iterate Over Val Dataset
        # Setup: Batch Generator, set loss and acc to 0, set eval mode on
        dataset.set_split("val")
        batch_generator = generate_batches(
            dataset, batch_size=args.batch_size, device=args.device
        )
        val_running_loss, val_running_acc = 0.0, 0.0
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # Step 1. Compute the Output
            y_pred = classifier(x_in=batch_dict["x_data"].float())

            # Step 2. Compute the loss
            loss = loss_func(y_pred, batch_dict["y_target"].float())
            loss_batch = loss.item()
            val_running_loss += (loss_batch - val_running_loss) / (batch_index + 1)

            # Step 3. Compute the accuracy
            acc_batch = compute_accuracy(y_pred, batch_dict["y_target"])
            val_running_acc += (acc_batch - val_running_acc) / (batch_index + 1)
            val_bar.set_postfix(
                loss=val_running_loss, acc=val_running_acc, epoch=epoch_index
            )
            val_bar.update()
        train_state["val_loss"].append(val_running_loss)
        train_state["val_acc"].append(val_running_acc)
        train_state = update_train_state(
            args=args, model=classifier, train_state=train_state
        )
        scheduler.step(train_state["val_loss"][-1])

        train_bar.n, val_bar.n = 0, 0
        epoch_bar.update()

        if train_state["stop_early"]:
            print("Stopping early....")
            break

        if epoch_index % 10 == 0:
            print(
                f"{epoch_index} Epoch Stats: "
                f"Training Loss={training_running_loss}, "
                f"Training Accuracy={training_running_acc}, "
                f"Validation Loss={val_running_loss}, "
                f"Validation Accuracy={val_running_acc}."
            )
    return train_state


def evaluate_test_split(classifier, dataset, loss_func, train_state, args):
    classifier = classifier.to(args.device)
    dataset.set_split("test")
    batch_generator = generate_batches(
        dataset, batch_size=args.batch_size, device=args.device
    )
    running_loss, running_acc = 0.0, 0.0
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # Step 1. Compute the Output
        y_pred = classifier(x_in=batch_dict["x_data"].float())

        # Step 2. Compute the loss
        loss = loss_func(y_pred, batch_dict["y_target"].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

    train_state["test_loss"] = running_loss
    train_state["test_acc"] = running_acc
    print(f"Test Accuracy={running_acc}, Test Loss={running_loss}.")
    train_state = update_train_state(
        args=args, model=classifier, train_state=train_state
    )
    return train_state


def predict_class(classifier, vectorizer, tweet, decision_threshold=0.5):
    tokenized_tweet = TweetVectorizer.tokenizer(tweet)
    vectorized_tweet = torch.tensor(vectorizer.vectorizer(tokenized_tweet))
    result = classifier(vectorized_tweet.view(1, -1))
    probability_value = F.sigmoid(result).item()
    predicted_index = 1 if probability_value >= decision_threshold else 0
    return vectorizer.target_vocab.lookup_index(predicted_index)


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
