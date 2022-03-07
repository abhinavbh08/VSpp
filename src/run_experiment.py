import numpy as np
from types import ClassMethodDescriptorType
from datasets import get_dataloaders
import config
import pickle
from vocabulary import Vocab
import torch

# from models import Combined
from losses import ContrastiveLoss

# from train import train_and_evaluate
from models import VSE
from evaluate import i2t, t2i

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_vocab_size():
    vocab = None
    with open(config.vocab_path, "rb") as file:
        vocab = pickle.load(file)
    return vocab, len(vocab)


def train(train_loader, model, epoch, val_loader):

    # switch to train mode
    model.train_start()
    loss_batches = 0.0

    for i, train_data in enumerate(train_loader):
        # Always reset to train mode, this is not the default behavior
        model.train_start()
        # Update the model
        loss_batches += model.train_emb(*train_data)
        if i % 100 == 99:
            print("Epoch: ", epoch, i, len(train_loader), loss_batches / 100)
            loss_batches = 0.0


def encode_data(model, data_loader):
    """Encode all images and captions loadable by `data_loader`"""
    # switch to evaluate mode
    model.val_start()
    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    with torch.no_grad():
        for i, (images, captions, lengths, ids) in enumerate(data_loader):
            # make sure val logger is used

            # compute the embeddings
            img_emb, cap_emb = model.forward_emb(images, captions, lengths)

            # initialize the numpy arrays given the size of th  e embeddings
            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

            # preserve the embeddings by copying from gpu and converting to numpy
            for i, id in enumerate(ids):
                img_embs[id] = img_emb.data.cpu().numpy().copy()[i]
                cap_embs[id] = cap_emb.data.cpu().numpy().copy()[i]

            # measure accuracy and record loss
            # model.forward_loss(img_emb, cap_emb)

            del images, captions

    return img_embs, cap_embs


def validate(val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(model, val_loader)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs)
    print(r1, r5, r10)
    # image retrieval
    (r1i, r5i, r10i, medri, meanri) = t2i(img_embs, cap_embs)
    print(r1i, r5i, r10i)
    # sum of recalls to be used for early stopping
    curr_score = r1 + r5 + r10 + r1i + r5i + r10i
    return curr_score


def main():
    train_loader = get_dataloaders(split="train")
    val_loader = get_dataloaders(split="val")
    test_loader = get_dataloaders(split="test")
    vocab, vocab_size = get_vocab_size()
    # model = Combined(vocab_size, device).to(device)

    # params = list(model.text_enc.parameters())
    # params += list(model.image_enc.fc.parameters())
    # if config.finetune:
    #     params += list(model.image_enc.model.parameters())
    # optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    # criterion = ContrastiveLoss(margin=config.margin)

    # train_and_evaluate(
    #     config.num_epochs,
    #     model,
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     optimizer,
    #     criterion,
    #     device,
    #     params
    # )
    best_sum = 0.0
    model = VSE(vocab_size, device)
    start_epoch = 0
    if config.resume_training:
        # checkpoint = torch.load("checkpoint.pth.tar", map_location=device)
        # filename = "/kaggle/input/checkpoint/checkpoint.pth.tar"
        checkpoint = torch.load(filename, map_location=device)
        start_epoch = checkpoint["epoch"]
        score = checkpoint["best_sum"]
        model.load_state_dict(checkpoint["model"])
        print("Loading checkpoint")
        print(f"Epoch: {start_epoch} , Score: {score}")
        best_sum = validate(val_loader, model)
        print("Current score after loading validated model", best_sum)

    for epoch in range(start_epoch, config.num_epochs):
        if epoch == 14:
            lr = config.learning_rate * 0.1
            for param_group in model.optimizer.param_groups:
                param_group["lr"] = lr

        train(train_loader, model, epoch, val_loader)
        curr_score = validate(val_loader, model)
        if curr_score > best_sum:
            best_sum = curr_score
            save_model_and_metadata(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "best_sum": best_sum,
                }
            )


def save_model_and_metadata(dct_values):
    print("Saving", dct_values["epoch"], dct_values["best_sum"])
    torch.save(dct_values, "checkpoint.pth.tar")


if __name__ == "__main__":
    main()
