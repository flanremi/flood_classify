import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

class MarginLoss(nn.Module):
    def __init__(self, margin=0.2, hardest=True, squared=False):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)
        if self.hardest:
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            loss = anc_pos_dist - anc_neg_dist + self.margin
            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask

            triplet_loss = F.relu(triplet_loss)

            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss

def _pairwise_distance(x, squared=False, eps=1e-16):
    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances

def _get_anchor_positive_triplet_mask(labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte()
    indices_not_equal = torch.ones(labels.shape[0]).to(device).byte() - indices_not_equal
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = indices_not_equal * labels_equal
    return mask

def _get_anchor_negative_triplet_mask(labels):
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ True # 如何改成0，1形式？
    return mask

def _get_triplet_mask(labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indices_not_same = torch.eye(labels.shape[0]).to(device).byte()
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)

    mask = distinct_indices * valid_labels

    return mask