import torch
import torch.nn as nn
import numpy as np


class ContrastiveLoss(nn.Module):

    def __init__(self, margin: float, hard_negative: bool = True):
        super().__init__()
        self.margin = margin
        self.hard_negative = hard_negative

    def forward(self, img_embeddings: torch.Tensor, caption_embeddings: torch.Tensor):
        similarity_matrix = img_embeddings.mm(caption_embeddings.t())
        
        # Diagonal contains the scores for the same image/caption pair.
        diagonal = similarity_matrix.diag()
        diagonal = diagonal.view(img_embeddings.size(0), 1)

        d1 = diagonal.expand_as(similarity_matrix)
        d2 = diagonal.t().expand_as(similarity_matrix)

        # find diff between same image and different captions
        cost_captions = (self.margin + similarity_matrix - d1).clamp(min=0)

        # find diff between same caption and different images
        cost_images = (self.margin + similarity_matrix - d2).clamp(min=0)

        # Loss is taken over different casption and images., so makin the diagonals as 0.
        mask = torch.eye(similarity_matrix.size(0)) > .5
        if torch.cuda.is_available():
            mask = mask.cuda()
        cost_captions = cost_captions.masked_fill_(mask, 0)
        cost_images = cost_images.masked_fill_(mask, 0)

        # Take the maximum if only want to consider the hard negatives.
        if self.hard_negative:
            cost_captions = cost_captions.max(dim=1)[0]
            cost_images = cost_images.max(dim=0)[0]

        return cost_captions.sum() + cost_images.sum()