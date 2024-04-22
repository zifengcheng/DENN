import torch


def precision_at_top_k(pred, gt, k):
    _, top_k_ids = torch.topk(pred, k, dim=-1)
    p_k = torch.tensor([gt[idx, t].mean() for idx, t in enumerate(top_k_ids)]).mean()
    return float(p_k)


def normalized_discounted_cumulated_gains(pred, gt, k):
    _, top_k_ids = torch.topk(torch.tensor(pred), k, dim=-1)
    dcg = torch.sum(torch.stack([gt[idx, t] for idx, t in enumerate(top_k_ids)]) / torch.log(torch.arange(k) + 2.)[None, :], dim=-1)
    factor = torch.zeros(len(gt))
    for sample_idx, sample_gt in enumerate(gt):
        factor[sample_idx] = torch.sum(1 / torch.log(torch.arange(min(sample_gt.sum(), k)) + 2.))
    ndcg = torch.mean(dcg / factor)
    return ndcg
