import numpy as np
import torch
import config
import pickle
from torchvision import transforms
from PIL import Image
import nltk
from vocabulary import Vocab
from models import VSE
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import io
from fastapi import UploadFile, File, Form

app = FastAPI()


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


def load_model():
    model = VSE(vocab_size, device)
    filename = "checkpoint.pth.tar"
    checkpoint = torch.load(filename, map_location=device)
    start_epoch = checkpoint["epoch"]
    score = checkpoint["best_sum"]
    model.load_state_dict(checkpoint["model"])
    print("Loading checkpoint")
    print(f"Epoch: {start_epoch} , Score: {score}")
    return model


def get_sorted_indices(text, myfile, model, vocab, tfms):
    print("VOCABSIZE", vocab_size)
    images_transformed = []
    for mf in myfile:
        content = mf.file
        image = Image.open(content).convert("RGB")
        print(image.size)
        image = tfms(image)
        images_transformed.append(image)

    # Convert the images into a batch.
    images_batch = torch.stack(images_transformed, dim=0)

    caption = nltk.tokenize.word_tokenize(str(text).lower())
    indexed_caption = [vocab(tok) for tok in caption]
    indexed_caption = torch.Tensor(indexed_caption).long()
    indexed_caption = indexed_caption.unsqueeze(0)
    lengths = [len(caption)]

    model.val_start()
    # Pass the batch through the model to get the embeddings.
    with torch.no_grad():
        img_emb, cap_emb = model.forward_emb(images_batch, indexed_caption, lengths)

    dp = np.dot(img_emb, cap_emb.T).flatten()

    # Sorted the indices on the order of similarity, and then reverse it.
    sorted_indices = np.argsort(dp)[::-1]
    return sorted_indices


# Loading the required things.
vocab, vocab_size = get_vocab()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tfms = get_transforms()
model = load_model()


# Path on which the request will be sent by streamlit frontend.
@app.post("/uploadfile/")
async def upload_files(myfile: List[UploadFile] = File(...), desc: str = Form(...)):
    print("Model", model)
    sorted_indices = get_sorted_indices(desc, myfile, model, vocab, tfms)
    return {"indices": sorted_indices.tolist()}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8081)
