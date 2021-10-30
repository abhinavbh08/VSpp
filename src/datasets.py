from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import os
import pickle
import torch
from torchvision.transforms.transforms import Resize
import config
import json
from vocabulary import Vocab
import nltk


class Flickr30K(Dataset):
    """Loads and gives the dataset to the dataloaders"""

    def __init__(
        self,
        root_path: str,
        images_folder: str,
        json_name: str,
        split: str,
        vocab_path: str = None,
        transformations: Optional[
            List[
                Union[
                    torchvision.transforms.ToTensor,
                    torchvision.transforms.Resize,
                    torchvision.transforms.ToTensor,
                ]
            ]
        ] = None,
    ) -> None:
        """
        Initialize the flickr30k dataset
        and read the json files creating the splits as give by Karpathy

        Args:
            root_path: Path to the directory containing the images folder and json file describing the images.
            images_folder: Name of the folder containing the images.
            json_name: Name of the json file containing the image ids and captions.
            split: Whether it is train, val or test split.
            vocab_path: the path to the saved vocab file if using an rnn based model.
            transforms: Optional list of transforms to be applied on each data point.
        """
        self.root_path = root_path
        self.images_path = os.path.join(self.root_path, images_folder)
        with open(os.path.join(root_path, json_name), "r") as file:
            self.data = json.load(file)

        # Loading the vocab is using an rnn based model.
        self.vocab = None
        if vocab_path:
            with open(vocab_path, "rb") as file:
                self.vocab = pickle.load(file)

        self.data = self.data["images"]
        self.items = []
        for index, image_info in enumerate(self.data):
            if image_info["split"] == split:
                self.items += [
                    (index, pos) for pos in range(len(image_info["sentences"]))
                ]
        self.transformations = transformations

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        tup_index = self.items[index]
        image_id = tup_index[0]
        image_name = self.data[image_id]["filename"]
        image_path = os.path.join(self.images_path, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transformations:
            image = self.transformations(image)

        caption_id = tup_index[1]
        caption = self.data[image_id]["sentences"][caption_id]["raw"]

        caption = nltk.tokenize.word_tokenize(caption.lower())

        indexed_caption = [self.vocab(tok) for tok in caption]
        indexed_caption = torch.Tensor(indexed_caption)
        return image, indexed_caption, index, image_id


def collate_fn(
    data: List[Tuple[torch.Tensor, torch.Tensor, int, int]]
) -> Tuple[torch.Tensor, torch.Tensor, List[int], Tuple[int]]:
    """
    Create mini batch of data from the list of tuples sent by the dataset class.

    Args:
        data: list of (image, caption, data_index, img_id) tuples

    Returns:
        images: Mini-batch of images.
        targets: Mini-batch of the padded and indexed captions.
        lengths: The lengths of each of the sentence in the dataset before padding.
        ids: The image ids.
    """
    images, captions, ids, img_ids = zip(*data)

    images = torch.stack(images, 0)
    lengths = [len(sent) for sent in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_dataloaders(split: str) -> DataLoader:
    """Returns a dataloader according to the split/

    Args:
        split: Whether train, test or validation dataloader is required.
    """
    # List of the transforms to be applied on each image.
    tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(256),
            transforms.CenterCrop(config.size),
        ]
    )

    dataset = Flickr30K(
        root_path=config.root_path,
        images_folder=config.images_folder,
        json_name=config.json_name,
        split=split,
        vocab_path=config.vocab_path,
        transformations=tfms,
    )

    shuffle = True if split == "train" else False

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    return loader


# dl = get_dataloaders("train")
# print("abc")
