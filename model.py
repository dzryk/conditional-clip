import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import copy
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable

from CLIP import clip

TXTDIM = 512
VISDIM = 768


def offgrad(model, ft):
    for pname, p in model.named_parameters():
        pname = pname.lower()
        if ft in pname or 'class_embedding' in pname:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


class XELoss(nn.Module):
    def __init__(self, hparams):
        super(XELoss, self).__init__()
        self.hparams = hparams

    def loss(self, x, y, device):
        """XELoss from CLIP."""
        scores = x.mm(y.t()) / self.hparams.temp
        ground_truth = torch.arange(len(scores)).type_as(scores).long()
        cost_x = F.cross_entropy(scores, ground_truth)
        cost_y = F.cross_entropy(scores.t(), ground_truth)
        return 0.5 * (cost_x + cost_y)


class Model(pl.LightningModule):
    """Conditional CLIP Model."""
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.save_hyperparameters(args)

        clip_model = clip.load(self.hparams.clip_model, jit=False)[0].float()
        clip_model = offgrad(clip_model, self.hparams.ft)
        self.ctx_transformer = copy.deepcopy(clip_model.transformer)
        self.tgt_transformer = copy.deepcopy(clip_model.transformer)

        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.visual_positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.visual_transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding

        self.ctx_ln_final = copy.deepcopy(clip_model.ln_final)
        self.tgt_ln_final = copy.deepcopy(clip_model.ln_final)
        self.text_projection = clip_model.text_projection
        self.W = nn.Linear(TXTDIM, VISDIM)
        self.loss = XELoss(self.hparams)

    def encode_context(self, text):
        text = clip.tokenize(text, truncate=True).to(self.device)
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.ctx_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ctx_ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    def encode_image(self, image, text):
        # Context
        ctx = self.W(self.encode_context(text))

        x = self.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + ctx.unsqueeze(1), x], dim=1)
        x = x + self.visual_positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :] + ctx)
        if self.proj is not None:
            x = x @ self.proj
        return x

    def encode_target(self, text):
        text = clip.tokenize(text, truncate=True).to(self.device)
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.tgt_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.tgt_ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def compute_loss(self, image, context, target):
        source = self.encode_image(image, context)
        target = self.encode_target(target)
        source = source / source.norm(dim=-1, keepdim=True)
        target = target / target.norm(dim=-1, keepdim=True)
        return self.loss.loss(source, target, device=self.device)

    def training_step(self, batch, batch_idx):
        x, ctx, y = batch
        loss = self.compute_loss(x, ctx, y)
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, ctx, y = batch
        loss = self.compute_loss(x, ctx, y)
        self.log('vloss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lrate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.tmax)
        return [optimizer], [scheduler]