from collections import defaultdict

import numpy as np
import torch
import random

from scipy.special import comb


def calc_adv(val, k):
    c = len(np.where(val==1)[0])
    n = len(val)
    rho = 1 - comb(n-c, k) / comb(n, k)
    sigma = np.sqrt(rho * (1 - rho))
    adv_p = (1 - rho) / (sigma + 1e-6)
    adv_n = (1 - rho - comb(n-c-1, k-1)/comb(n-1,k-1)) / (sigma + 1e-6)
    new_val = np.where(val==1, adv_p, val)
    new_val = np.where(new_val==0, adv_n, new_val)
    return new_val

def compute_advantage(token_level_rewards, response_mask, index, K):
    scores = token_level_rewards.sum(dim=-1)
    
    id2score = defaultdict(list)
    uid2sid = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i].detach().item())
            uid2sid[index[i]].append(i)
        for uid in id2score.keys():
            reward = np.array(id2score[uid])
            adv = calc_adv(reward, K)
            print(uid2sid[uid])
            for i in range(len(uid2sid[uid])):
                scores[uid2sid[uid][i]] = adv[i]

    scores = scores.unsqueeze(-1) * response_mask
    
    return scores, scores