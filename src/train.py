import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from evaluate import i2t, t2i

def train_epoch(model, iterator, optimizer, criterion, device):
    """Runs the training loop for the model.
    Args:
        model (nn.Module): Pytorch model object of our defined class.
        iterator (torch.utils.Data.DataLoader): The iterator for the data.
        optimizer (torch.optim): The optimization algorithm used to train the model.
        criterion (torch.nn): The loss function.
    Returns:
        epoch_loss (float): The average loss for one epoch on the given dataloader.
        epoch_acc (float): The average accuracy one the epoch on the given dataloader.
    """
    epoch_loss = 0
    losses = []
    # Put the model in the training mode.
    model.train()

    # for each batch in the dataloader
    for batch_idx, batch in enumerate(iterator):

        # Clear out the gradients from the previous batch
        optimizer.zero_grad()

        # move the inputs and the labels to the device.
        images = batch[0].to(device)
        captions = batch[1].to(device)
        lengths = batch[2]

        image_embs, txt_embs = model(images, captions, lengths)

        # calculate the loss value using our loss function on this batch
        loss = criterion(image_embs, txt_embs)

        # Do backpropagation of the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # add the loss and the accuracy for the epoch.
        epoch_loss += loss.item()
        losses.append(loss.item())

        if batch_idx % 10 == 0:
            message = "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                batch_idx * len(batch[0]),
                len(iterator.dataset),
                100.0 * batch_idx / len(iterator),
                np.mean(losses),
            )

            print(message)
            losses = []

    return epoch_loss / len(iterator)


def validate_epoch(model, iterator, criterion, device):
    """Runs the evaluation loop for the model.
    Args:
        model (nn.Module): Pytorch model object of our defined class.
        iterator (torch.utils.Data.DataLoader): The iterator for the data.
        criterion (torch.nn): The loss function.
        device (str): The device which is available, it can be either cuda or cpu.
    Returns:
        epoch_loss (float): The average loss for the epoch on the given dataloader.
        epoch_acc (float): The average accuracy for the epoch on the given dataloader.
    """
    epoch_loss = 0
    losses = []
    # Put the model in the evaluation mode.
    model.eval()

    img_embs_full = None
    cap_embs_full = None

    # Do not calculate the gradients in the evaluaion mode.
    with torch.no_grad():

        # for each batch in the dataloader
        for batch_idx, batch in enumerate(iterator):

            # move the inputs and the labels to the device.
            images = batch[0].to(device)
            captions = batch[1].to(device)
            lengths = batch[2]
            ids = batch[3]

            image_embs, txt_embs = model(images, captions, lengths)

            if img_embs_full is None:
                img_embs_full = np.zeros((len(iterator.dataset), image_embs.size(1)))
                cap_embs_full = np.zeros((len(iterator.dataset), txt_embs.size(1)))


            ids = list(ids)
            img_embs_full[ids] = image_embs.data.cpu().numpy().copy()
            cap_embs_full[ids] = txt_embs.data.cpu().numpy().copy()

            # calculate the loss value using our loss function on this batch
            loss = criterion(image_embs, txt_embs)

            # add the loss and the accuracy for the epoch.
            epoch_loss += loss.item()
            losses.append(loss.item())

            if batch_idx % 10 == 0:
                message = "Validation: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(batch[0]),
                    len(iterator.dataset),
                    100.0 * batch_idx / len(iterator),
                    np.mean(losses),
                )

                print(message)
                losses = []
    
    (r1, r5, r10, medr, meanr) = i2t(img_embs_full, cap_embs_full)
    (r1i, r5i, r10i, medri, meanri) = t2i(img_embs_full, cap_embs_full)
    print("I2T", r1, r5, r10)
    print("T2I", r1i, r5i, r10i)
    return epoch_loss / len(iterator)


def train_and_evaluate(
    num_epochs, model, train_loader, val_loader, test_loader, optimizer, criterion, device
):
    """Call the train and evaluate function for each of the epoch, print the loss and accuracies.
    Args:
        num_epochs (int): The number of epochs for which to train the model.
        model (nn.Module): Pytorch model object of our defined class.
        train_loader (torch.utils.Data.DataLoader): The iterator for the training data.
        val_loader (torch.utils.Data.DataLoader): The iterator for the validation data.
        test_loader (torch.utils.Data.DataLoader):  The iterator for the test data.
        optimizer (torch.optim): The optimization algorithm used to train the model.
        criterion (torch.nn): The loss function.
        device (str): The device which is available, it can be either cuda or cpu.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler
    Returns:
        train_set_loss (List[float]): The loss for the training set
        train_set_acc (List[float]): The accuracy for the training set
        val_set_loss (List[float]): The loss for the validation set
        val_set_acc (List[float]): The accuracy for the validation set
        test_set_loss (List[float]): The loss for the testing set
        test_set_loss (List[float]): The accuracy for the testing set
    """

    train_set_loss = []
    # train_set_acc = []
    val_set_loss = []
    # val_set_acc = []
    test_set_loss = []
    # test_set_acc = []

    for epoch in range(config.num_epochs):

        # Call the training function with the training data loader and save loss and accuracy
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_set_loss.append(train_loss)
        # train_set_acc.append(train_acc)

        # scheduler.step()

        # Call the evaluation function with the vaidation data laoder and save loss and accuracy
        valid_loss = validate_epoch(model, val_loader, criterion, device)
        val_set_loss.append(valid_loss)
        # val_set_acc.append(valid_acc)

        # Call the evaluation function with the test data loader and save loss and accuracy.
        test_loss = validate_epoch(model, test_loader, criterion, device)
        test_set_loss.append(test_loss)
        # test_set_acc.append(test_acc)

        # print(f"======================EPOCH {epoch}=========================")
        # print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        # print(f'Val. Loss: {valid_loss:.3f}  |  Val. Acc: {valid_acc*100:.2f}%')
        # print(f'Test. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')

        print(f"======================EPOCH {epoch}=========================")
        print(f"Train Loss: {train_loss:.3f}")
        print(f"Val. Loss: {valid_loss:.3f}")
        print(f"Test. Loss: {test_loss:.3f}")
