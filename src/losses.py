import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float, hard_negative: bool = True):
        super().__init__()
        self.margin = margin
        self.hard_negative = hard_negative

    def forward(self, img_embeddings: torch.Tensor, caption_embeddings: torch.Tensor):
        scores = img_embeddings.mm(caption_embeddings.t())

        diagonal = scores.diag().view(img_embeddings.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.hard_negative:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
