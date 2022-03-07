import numpy as np
import torch
import config
import pickle
from torchvision import transforms
from PIL import Image
import nltk
from vocabulary import Vocab
from models import VSE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_vocab():
    vocab = None
    with open(config.vocab_path, "rb") as file:
        vocab = pickle.load(file)
    return vocab, len(vocab)


def get_transforms():
    # List of the transforms to be applied on each image.
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    tfms = transforms.Compose(t_list + t_end)
    return tfms


paths = [
    "data/flickr30k_images/flickr30k_images/667626.jpg",
    "data/flickr30k_images/flickr30k_images/65567.jpg",
    "data/flickr30k_images/flickr30k_images/3494059.jpg",
    "data/flickr30k_images/flickr30k_images/2285664.jpg",
    "data/flickr30k_images/flickr30k_images/3025093.jpg",
]

tfms = get_transforms()
vocab, vocab_size = get_vocab()
model = VSE(vocab_size, device)
filename = "checkpoint.pth.tar"
checkpoint = torch.load(filename, map_location=device)
start_epoch = checkpoint["epoch"]
score = checkpoint["best_sum"]
model.load_state_dict(checkpoint["model"])
print("Loading checkpoint")
print(f"Epoch: {start_epoch} , Score: {score}")


images_transformed = []
for image_path in paths:
    image = Image.open(image_path).convert("RGB")
    image = tfms(image)
    images_transformed.append(image)

images_batch = torch.stack(images_transformed, dim=0)

text = "A doctor and nurse are doing something."
caption = nltk.tokenize.word_tokenize(str(text).lower())

indexed_caption = [vocab(tok) for tok in caption]
indexed_caption = torch.Tensor(indexed_caption).long()
indexed_caption = indexed_caption.unsqueeze(0)
lengths = [len(caption)]

model.val_start()
with torch.no_grad():
    img_emb, cap_emb = model.forward_emb(images_batch, indexed_caption, lengths)

dp = np.dot(img_emb, cap_emb.T).flatten()
sorted_indices = np.argsort(dp)[::-1]
print(sorted_indices)
