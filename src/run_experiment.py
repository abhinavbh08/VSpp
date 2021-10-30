from types import ClassMethodDescriptorType
from datasets import get_dataloaders
import config
import pickle
from vocabulary import Vocab
import torch
from models import Combined
from losses import ContrastiveLoss
from train import train_and_evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_vocab_size():
    vocab = None
    with open(config.vocab_path, "rb") as file:
        vocab = pickle.load(file)
    return vocab, len(vocab)


def main():
    train_loader = get_dataloaders(split="train")
    val_loader = get_dataloaders(split="val")
    test_loader = get_dataloaders(split="test")
    vocab, vocab_size = get_vocab_size()
    model = Combined(vocab_size, device).to(device)

    params = list(model.text_enc.parameters())
    params += list(model.image_enc.fc.parameters())
    if config.finetune:
        params += list(model.image_enc.model.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    criterion = ContrastiveLoss(margin=config.margin)

    train_and_evaluate(
        config.num_epochs,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion,
        device
    )


if __name__ == "__main__":
    main()
