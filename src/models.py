import torch
from torch.functional import norm
from torchvision import models
import torch.nn as nn
from losses import ContrastiveLoss
from torch.nn.utils.clip_grad import clip_grad_norm_

import config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np

def normalise(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Converts the embeddings to a unit hypersphere. Basically A/||A||

    Args:
        embeddings: The input embeddings from the image or the text encoder.

    Returns:
        embeddings: The normalised embeddings
    """
    norm = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).sqrt()
    embeddings = torch.div(embeddings, norm)
    return embeddings


class ImageEncoder(nn.Module):
    """Image Encoder module. Any encoder can be used as long as it returns an embedding for an image."""

    def __init__(
        self,
        model_name: str,
        normalise: bool,
        embedding_dim: int,
        pretrained: bool = True,
        finetune_full: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalise = normalise

        if model_name == "vgg":
            self.model = models.vgg19(pretrained=pretrained)
            if finetune_full == False:
                self.set_grads_false()
            in_feat = self.model.classifier._modules['6'].in_features
            
            # In the original implementation, the original model is broken into twoi parts, features
            # which cn then be normalised and then the fc layer.
            # SO, we will do it their way
            self.fc = nn.Linear(in_feat, embedding_dim)
            # Removing the last layer from the model
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier.children())[:-1]
            )
        elif model_name == "resnet":
            self.model = models.resnet18(pretrained=pretrained)
            if finetune_full == False:
                self.set_grads_false()
            in_feat = self.model.fc.in_features
            self.fc = nn.Linear(in_feat, embedding_dim)
            self.model.fc = nn.Sequential()

        self.init_weights()

    def forward(self, images):
        features = self.model(images)
        features = normalise(features)
        features = self.fc(features)
        if self.normalise:
            features = normalise(features)
        return features

    def set_grads_false(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)



class TextEncoder(nn.Module):
    """The text encoder which is rnn based. This can be replaced by transformers based encoders such as bert."""

    def __init__(
        self,
        normalise: bool,
        vocab_size: int,
        embedding_dim: int,
        word_embedding_dim: int,
        device
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.normalise = normalise
        self.hidden_dim = embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.word_embedding_dim)
        self.gru = nn.GRU(self.word_embedding_dim, self.embedding_dim, batch_first=True)
        self.device = device
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, inputs, lengths):
        bs = inputs.size(0)
        hidden = self.init_hidden(bs)
        embeds = self.embedding(inputs)
        embeds = pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.gru(embeds, hidden)
        outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
        # Getting the last hidden state for each of the input
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(inputs.size(0), 1, self.embedding_dim) - 1)
        if torch.cuda.is_available():
            I = I.cuda()
        features = torch.gather(outputs, 1, I).squeeze(1)
        if self.normalise:
            features = normalise(features)

        return features

    def init_hidden(self, bs):
        return torch.zeros((1, bs, self.hidden_dim), requires_grad=True).to(self.device)
        


class VSE:

    def __init__(self, vocab_size, device):
        self.image_enc = ImageEncoder(
            model_name=config.model_name,
            normalise=config.normalise,
            embedding_dim=config.embedding_dim,
            pretrained=config.pretrained,
            finetune_full=config.finetune
        )
        self.text_enc = TextEncoder(
            normalise=config.normalise,
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            word_embedding_dim=config.word_embedding_dim,
            device = device
        )
        self.grad_clip = 2.0
        if torch.cuda.is_available():
            self.image_enc.cuda()
            self.text_enc.cuda()

        self.criterion = ContrastiveLoss(margin=config.margin)
        params = list(self.text_enc.parameters())
        params += list(self.image_enc.fc.parameters())
        if config.finetune:
            params += list(self.image_enc.model.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    def train_start(self):
        self.image_enc.train()
        self.text_enc.train()

    def val_start(self):
        self.image_enc.eval()
        self.text_enc.eval()

    def forward_emb(self, images, captions, lengths):
        images = Variable(images)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.image_enc(images)
        cap_emb = self.text_enc(captions, lengths)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb):
        loss = self.criterion(img_emb, cap_emb)
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        return loss.item()
# class Combined(nn.Module):
#     def __init__(self, vocab_size: int, device):
#         super().__init__()
#         self.image_enc = ImageEncoder(
#             model_name=config.model_name,
#             normalise=config.normalise,
#             embedding_dim=config.embedding_dim,
#             pretrained=config.pretrained,
#             finetune_full=config.finetune
#         )
#         self.text_enc = TextEncoder(
#             normalise=config.normalise,
#             vocab_size=vocab_size,
#             embedding_dim=config.embedding_dim,
#             word_embedding_dim=config.word_embedding_dim,
#             device = device
#         )

#     def forward(self, images, captions, lengths):
#         images = self.image_enc(images)
#         captions = self.text_enc(captions, lengths)
#         return images, captions

# enc = ImageEncoder(model_name=config.model_name, normalise=config.normalise, embedding_dim=config.embedding_dim, pretrained=False)
# # print(enc.model)
# img = torch.rand(3, 224, 224).unsqueeze(0)
# # print(enc(img).shape)


# text_enc = TextEncoder(normalise=True, vocab_size=512, embedding_dim=512, word_embedding_dim=300, device="cpu")
# txt = torch.arange(10)
# txt = txt.unsqueeze(0)
# text_enc(txt, torch.Tensor([10]))
# vse = VSEPP(512)
# ret = vse(img, txt, torch.Tensor([10]))
# print("abc")