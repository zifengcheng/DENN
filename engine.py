import time

import torch

import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from torch.utils.data import DataLoader

from metrics import precision_at_top_k, normalized_discounted_cumulated_gains

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.5, dataset=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.dataset = dataset

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # featuure: bsz * 2 * 768
        # labels: (bsz * 2)
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        contrast_feature = contrast_feature / contrast_feature.norm(dim=1).unsqueeze(1)
        #print(features.shape, contrast_feature.shape)

        anchor_feature = contrast_feature
        anchor_count = contrast_count


        # compute logits
        #print(anchor_feature.shape,contrast_feature.shape)
        anchor_dot_contrast = torch.matmul(anchor_feature,contrast_feature.T) / self.temperature
        #anchor_dot_contrast = torch.cdist(anchor_feature,contrast_feature) / self.temperature * -1 
        #sim = torch.cdist(feature, feature, p=2)/args.temp * -1
        
        #print(anchor_dot_contrast.shape,anchor_feature.shape,contrast_feature.shape)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if self.dataset == 'SemEval':
            labels_temp = labels.sum(1).unsqueeze(1)
            labels_temp = 1 - labels_temp.bool().long()
            labels = torch.cat((labels,labels_temp),1)

        occ = torch.matmul(labels,labels.T)
        
        ground_truth_ = labels.sum(1)      
        ground_truth_1 = ground_truth_.unsqueeze(0).expand(ground_truth_.shape[0], -1)
        ground_truth_2 = ground_truth_.unsqueeze(1).expand(-1, ground_truth_.shape[0])
        count_ = torch.where(ground_truth_1>ground_truth_2, ground_truth_1, ground_truth_2)
        
        
        weight = 2 - occ /count_
        weight = weight - torch.eye(weight.shape[0]).cuda()

        # tile mask
        #mask = mask.repeat(anchor_count, contrast_count)
        mask = torch.eye(int(batch_size)).repeat(2,2) - torch.eye(batch_size * 2)
        mask = mask.cuda()
        
        # compute log_prob
        exp_logits = torch.exp(logits) * weight
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        #print(mask.shape,log_prob.shape)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        #print('mean_log_prob_pos',mean_log_prob_pos)

        # loss
        #loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def get_label(model, criterion, optimizer, batch_data, return_prob=False):
    model.eval()


    data_ids, input_ids, attention_mask, labels = batch_data

    data_ids = data_ids.cuda(non_blocking=True)
    input_ids = input_ids.cuda(non_blocking=True)
    attention_mask = attention_mask.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    #optimizer.zero_grad()


    logits,_ = model(input_ids, attention_mask)
    loss = criterion(logits, labels, data_ids)

    #scaler.scale(loss).backward()
    #scaler.step(optimizer)
    #scaler.update()

    if not return_prob:
        return loss.detach().cpu()
    else:
        return loss.detach().cpu(), logits.detach().sigmoid().cpu()
    
@torch.no_grad()    
def get_feature(model, batch_data):
    model.eval()

    data_ids, input_ids, attention_mask, labels = batch_data

    data_ids = data_ids.cuda(non_blocking=True)
    input_ids = input_ids.cuda(non_blocking=True)
    attention_mask = attention_mask.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    _, feature = model(input_ids, attention_mask)
    return feature




def train_iter(args,model, criterion, optimizer, batch_data, return_prob=False):
    model.train()

    data_ids, input_ids, attention_mask, labels = batch_data

    data_ids = data_ids.cuda(non_blocking=True)
    input_ids = input_ids.cuda(non_blocking=True)
    attention_mask = attention_mask.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    
    #input_ids = torch.cat((input_ids,input_ids))
    #attention_mask = torch.cat((attention_mask,attention_mask))
    #labels = torch.cat((labels,labels))

    optimizer.zero_grad()

    logits_1, feature_1 = model(input_ids, attention_mask)
    logits_2, feature_2 = model(input_ids, attention_mask)
    loss = criterion(torch.cat((logits_1,logits_2)), torch.cat((labels,labels)))
    
    #label = torch.cat((torch.arange(bsz/2,bsz).long() , torch.arange(0,bsz/2).long())).cuda()
    criterion1 = SupConLoss(temperature=args.temp, dataset=args.dataset_name)
    #contrastive_loss = criterion1(torch.cat((feature_1.unsqueeze(1),feature_2.unsqueeze(1)),1), labels)
    contrastive_loss = criterion1(torch.cat((feature_1.unsqueeze(1),feature_2.unsqueeze(1)),1), torch.cat((labels,labels)))
    loss += contrastive_loss * args.alpha
    

    loss.backward()
    optimizer.step()

    if not return_prob:
        return loss.detach().cpu(), contrastive_loss
    else:
        return loss.detach().cpu(), logits.detach().sigmoid().cpu()


@torch.no_grad()
def eval(model, data_loader):
    eval_st = time.time()

    model.eval()
    pred_list, gt_list = [], []
    for _, input_ids, attention_mask, labels in data_loader:
        input_ids = input_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        logits,_ = model(input_ids, attention_mask)
        prob = func.sigmoid(logits)

        pred_list.append(prob)
        gt_list.append(labels)

    pred_list = torch.cat(pred_list, dim=0).cpu().float()
    gt_list = torch.cat(gt_list, dim=0).cpu()

    p_1 = precision_at_top_k(pred_list, gt_list, 1)
    p_3 = precision_at_top_k(pred_list, gt_list, 3)
    p_5 = precision_at_top_k(pred_list, gt_list, 5)

    n_3 = normalized_discounted_cumulated_gains(pred_list, gt_list, 3)
    n_5 = normalized_discounted_cumulated_gains(pred_list, gt_list, 5)

    pred = (pred_list > 0.5).long()
    p = precision_score(gt_list, pred, average='micro')
    r = recall_score(gt_list, pred, average='micro')
    f1 = f1_score(gt_list, pred, average='micro')
    hl = hamming_loss(gt_list, pred)
    
    p_ = precision_score(gt_list, pred, average='weighted')
    r_ = recall_score(gt_list, pred, average='weighted')
    f1_ = f1_score(gt_list, pred, average='weighted')
    
    print('weighted f1 is',p_,r_,f1_)
    
    return p_1, p_3, p_5, n_3, n_5, p, r, f1, hl, time.time() - eval_st


@torch.no_grad()
def eval_new(args, model, data_loader, feature_train, label_train, metric = 'euc'):
    eval_st = time.time()

    model.eval()
    pred_list, gt_list = [], []
    pred_list_1, gt_list_1 = [], []
    
    pred_list_2, gt_list_2 = [], []
    pred_list_3, gt_list_3 = [], []
    
    pred_list_m, gt_list_m = [], []
    feature_train = feature_train / feature_train.norm(dim=1).unsqueeze(1)
    
    for _, input_ids, attention_mask, labels in data_loader:
        input_ids = input_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        logits, feature = model(input_ids, attention_mask)
        prob = func.sigmoid(logits)
        pred_list_1.append(prob)
        gt_list_1.append(labels)
        distances = torch.empty(0,).cuda()
        
        feature = feature / feature.norm(dim=1).unsqueeze(1)
        distances = torch.matmul(feature, feature_train.T)
        
        '''for fea in feature:
            if metric == 'euc':
                distance = func.pairwise_distance(fea, feature_train)
                distances = torch.cat((distances,distance.unsqueeze(0)))
            elif metric == 'cos':
                distance = func.cosine_similarity(fea, feature_train)
                distances = torch.cat((distances,distance.unsqueeze(0)))'''
        '''distances = torch.cdist(feature, feature_train, p=2)
        if metric == 'euc':
            distances *= -1'''

        k_distance, index = torch.topk(distances,k=args.k,dim=1).values , torch.topk(distances,k=args.k,dim=1).indices
        #print(distance,index)
        k_distance_ , knn_pred = func.softmax(k_distance,-1) ,label_train[index]

        knn_pred = k_distance_.unsqueeze(2) * knn_pred
        knn_pred = knn_pred.sum(1)
        #print(prob.shape,knn_pred.shape)
        prob1 = 0.5 * prob + 0.5 * knn_pred
        #print(prob.shape,labels.shape)
        pred_list.append(prob1)
        gt_list.append(labels)
        
        k_distance_ , knn_pred = func.softmax(k_distance/2,-1) ,label_train[index]
        knn_pred = k_distance_.unsqueeze(2) * knn_pred
        knn_pred = knn_pred.sum(1)
        #print(prob.shape,knn_pred.shape)
        prob1 = 0.5 * prob + 0.5 * knn_pred
        #print(prob.shape,labels.shape)
        pred_list_2.append(prob1)
        gt_list_2.append(labels)
        
        k_distance_ , knn_pred = func.softmax(k_distance/3,-1) ,label_train[index]
        knn_pred = k_distance_.unsqueeze(2) * knn_pred
        knn_pred = knn_pred.sum(1)
        #print(prob.shape,knn_pred.shape)
        prob1 = 0.5 * prob + 0.5 * knn_pred
        #print(prob.shape,labels.shape)
        pred_list_3.append(prob1)
        gt_list_3.append(labels)
        
        
        k_distance_ , knn_pred = func.softmax(k_distance/0.5,-1) ,label_train[index]
        knn_pred = k_distance_.unsqueeze(2) * knn_pred
        knn_pred = knn_pred.sum(1)
        #print(prob.shape,knn_pred.shape)
        prob1 = 0.5 * prob + 0.5 * knn_pred
        #print(prob.shape,labels.shape)
        pred_list_m.append(prob1)
        gt_list_m.append(labels)
        

    pred_list = torch.cat(pred_list, dim=0).cpu().float()
    gt_list = torch.cat(gt_list, dim=0).cpu()

    p_1 = precision_at_top_k(pred_list, gt_list, 1)
    p_3 = precision_at_top_k(pred_list, gt_list, 3)
    p_5 = precision_at_top_k(pred_list, gt_list, 5)

    n_3 = normalized_discounted_cumulated_gains(pred_list, gt_list, 3)
    n_5 = normalized_discounted_cumulated_gains(pred_list, gt_list, 5)

    pred = (pred_list > 0.5).long()
    p = precision_score(gt_list, pred, average='micro')
    r = recall_score(gt_list, pred, average='micro')
    f1 = f1_score(gt_list, pred, average='micro')
    hl = hamming_loss(gt_list, pred)
    
    
    # hyper
    
    pred_list = torch.cat(pred_list_2, dim=0).cpu().float()
    gt_list = torch.cat(gt_list_2, dim=0).cpu()
    
    p_1_ = precision_at_top_k(pred_list, gt_list, 1)
    p_3_ = precision_at_top_k(pred_list, gt_list, 3)
    p_5_ = precision_at_top_k(pred_list, gt_list, 5)

    n_3_ = normalized_discounted_cumulated_gains(pred_list, gt_list, 3)
    n_5_ = normalized_discounted_cumulated_gains(pred_list, gt_list, 5)

    pred = (pred_list > 0.5).long()
    p_ = precision_score(gt_list, pred, average='micro')
    r_ = recall_score(gt_list, pred, average='micro')
    f1_ = f1_score(gt_list, pred, average='micro')
    hl_ = hamming_loss(gt_list, pred)
    
    print('/ 2',p_1_, p_3_, p_5_, n_3_, n_5_, p_, r_, f1_, hl_)
    
    pred_list = torch.cat(pred_list_3, dim=0).cpu().float()
    gt_list = torch.cat(gt_list_3, dim=0).cpu()
    
    p_1_ = precision_at_top_k(pred_list, gt_list, 1)
    p_3_ = precision_at_top_k(pred_list, gt_list, 3)
    p_5_ = precision_at_top_k(pred_list, gt_list, 5)

    n_3_ = normalized_discounted_cumulated_gains(pred_list, gt_list, 3)
    n_5_ = normalized_discounted_cumulated_gains(pred_list, gt_list, 5)

    pred = (pred_list > 0.5).long()
    p_ = precision_score(gt_list, pred, average='micro')
    r_ = recall_score(gt_list, pred, average='micro')
    f1_ = f1_score(gt_list, pred, average='micro')
    hl_ = hamming_loss(gt_list, pred)
    
    print('/ 3',p_1_, p_3_, p_5_, n_3_, n_5_, p_, r_, f1_, hl_)
    
    pred_list = torch.cat(pred_list_m, dim=0).cpu().float()
    gt_list = torch.cat(gt_list_m, dim=0).cpu()
    
    p_1_ = precision_at_top_k(pred_list, gt_list, 1)
    p_3_ = precision_at_top_k(pred_list, gt_list, 3)
    p_5_ = precision_at_top_k(pred_list, gt_list, 5)

    n_3_ = normalized_discounted_cumulated_gains(pred_list, gt_list, 3)
    n_5_ = normalized_discounted_cumulated_gains(pred_list, gt_list, 5)

    pred = (pred_list > 0.5).long()
    p_ = precision_score(gt_list, pred, average='micro')
    r_ = recall_score(gt_list, pred, average='micro')
    f1_ = f1_score(gt_list, pred, average='micro')
    hl_ = hamming_loss(gt_list, pred)
    
    print('/ 0.5',p_1_, p_3_, p_5_, n_3_, n_5_, p_, r_, f1_, hl_)
    
    
    # origin   
    pred_list = torch.cat(pred_list_1, dim=0).cpu().float()
    gt_list = torch.cat(gt_list_1, dim=0).cpu()
    
    p_1_ = precision_at_top_k(pred_list, gt_list, 1)
    p_3_ = precision_at_top_k(pred_list, gt_list, 3)
    p_5_ = precision_at_top_k(pred_list, gt_list, 5)

    n_3_ = normalized_discounted_cumulated_gains(pred_list, gt_list, 3)
    n_5_ = normalized_discounted_cumulated_gains(pred_list, gt_list, 5)

    pred = (pred_list > 0.5).long()
    p_ = precision_score(gt_list, pred, average='micro')
    r_ = recall_score(gt_list, pred, average='micro')
    f1_ = f1_score(gt_list, pred, average='micro')
    hl_ = hamming_loss(gt_list, pred)
    

    return p_1, p_3, p_5, n_3, n_5, p, r, f1, hl, time.time() - eval_st, p_1_, p_3_, p_5_, n_3_, n_5_, p_, r_, f1_, hl_
