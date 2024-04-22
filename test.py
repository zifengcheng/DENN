import argparse
import random
import time
import warnings

from torch.optim import Adam
from torch.utils.data import DataLoader

from criterions import *
from datasets import aapd_topics, rcv1v2_topics, load_dataset, Amazon_topics, load_dataset_no_val, EUR_topics
from engine import train_iter, eval, get_feature, eval_new
from models import Model
from sklearn.metrics import f1_score
from metrics import precision_at_top_k, normalized_discounted_cumulated_gains
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score
import torch.nn.functional as func
import os

@torch.no_grad()    
def get_feature1(model, batch_data):
    model.eval()

    data_ids, input_ids, attention_mask, labels = batch_data

    data_ids = data_ids.cuda(non_blocking=True)
    input_ids = input_ids.cuda(non_blocking=True)
    attention_mask = attention_mask.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    logits, feature = model(input_ids, attention_mask)
    return torch.sigmoid(logits), feature

def get_args_parser():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', default=1, type=int)

    parser.add_argument('--root', default='../../DataSet/AAPD', type=str)
    parser.add_argument('--dataset_name', default='AAPD', type=str, help='AAPD or RCV1-V2')
    parser.add_argument('--setting', default='F', type=str, help='{F, P, PO, OP, OPON}')
    parser.add_argument('--percentage', default=1.0, type=float)
    parser.add_argument('--gpu', default='0,1', type=str)

    parser.add_argument('--loss', default='BCE', type=str, help='{BCE, AN, WAN, AN-LS, EPR, ROLE}')
    parser.add_argument('--an_ls_epsilon', default=0.1, type=float)
    parser.add_argument('--epr_k_type', default='GEN', type=str, help='{SET, GEN}')
    parser.add_argument('--epr_k', default=1., type=float)
    parser.add_argument('--epr_lam', default=1., type=float)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_epochs', default=50, type=int)
    #parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--val_iter', default=200, type=int)
    parser.add_argument('--max_fail_times', default=6, type=int)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--temp', default=0.5, type=float)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--thresh', default=0.7, type=float)
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_topics(dataset_name):
    num_topics = 0
    if dataset_name == 'AAPD':
        num_topics = len(aapd_topics)
    if dataset_name == 'RCV1-V2':
        num_topics = len(rcv1v2_topics)
    if dataset_name == 'Amazon':
        num_topics = len(Amazon_topics)
    if dataset_name == 'EUR':
        num_topics = len(EUR_topics)
    print('num_topics is', num_topics)
    return num_topics


def get_criterion(setting, loss, args):
    #criterion = None
    criterion = nn.BCEWithLogitsLoss()
    return criterion

@torch.no_grad()
def eval_222(args, model, data_loader, feature_train, label_train, item, threshold, class_index1=None, class_index2=None, class_index3=None, class_index4=None):
    eval_st = time.time()

    model.eval()
    pred_list, gt_list = [], []
    pred_list_knn, pred_list_model, pred_list_sta = [], [], []
    
    
    pred_list_mean = []
    
    pred_list_two = []
    pred_list_mk = []
    feature_train = feature_train / feature_train.norm(dim=1).unsqueeze(1)
    
    pred_list_2, pred_list_3, pred_list_5 = [], [] ,[]
    count = 0
    
    for _, input_ids, attention_mask, labels in data_loader:
        a = time.time()
        input_ids = input_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        logits, feature = model(input_ids, attention_mask)
        feature = feature / feature.norm(dim=1).unsqueeze(1)
        prob = func.sigmoid(logits)
        b = time.time()
        #print('model time', b - a)
        # Model
        pred_list_model.append(prob)
        gt_list.append(labels)

        distances = torch.matmul(feature, feature_train.T)
        #distances = torch.cdist(feature, feature_train, p=2)
            
        #  KNN
        k_distance, index = torch.topk(distances,k=args.k,dim=1).values , torch.topk(distances,k=args.k,dim=1).indices
        l1 = 0.5
        k_distance_ , knn_pred = func.softmax(k_distance * 20,-1) ,label_train[index]
        
        knn_pred = k_distance_.unsqueeze(2) * knn_pred
        knn_pred_ = knn_pred.sum(1)


        
        k_distance, index = torch.topk(distances,k=args.k,dim=1).values , torch.topk(distances,k=args.k,dim=1).indices
        k_distance_ , knn_pred = func.softmax(k_distance * 20,-1) ,label_train[index]

        knn_pred = k_distance_.unsqueeze(2) * knn_pred
        knn_pred_ = knn_pred.sum(1)
        pred_list_knn.append(knn_pred_)
        
        prob_ = prob >= args.thresh
        knn_pred_1 = torch.where(prob_, knn_pred_, torch.Tensor([1]).cuda())
        l1 = knn_pred_1.min(1).values.unsqueeze(1)
        #print('l1 is', l1, args.thresh)
        
        pred_list_5.append( (1-l1) * prob + l1 * knn_pred_)
    
    
    
    
    pred_list_5 = torch.cat(pred_list_5, dim=0).cpu().float()
    gt_list = torch.cat(gt_list, dim=0).cpu().float()
    
    p_1 = precision_at_top_k(pred_list_5, gt_list, 1)
    p_3 = precision_at_top_k(pred_list_5, gt_list, 3)
    p_5 = precision_at_top_k(pred_list_5, gt_list, 5)

    n_3 = normalized_discounted_cumulated_gains(pred_list_5, gt_list, 3)
    n_5 = normalized_discounted_cumulated_gains(pred_list_5, gt_list, 5)
    
    pred = (pred_list_5 > threshold).long()
    p1 = precision_score(gt_list, pred, average='micro')
    r1 = recall_score(gt_list, pred, average='micro')
    f11 = f1_score(gt_list, pred, average='micro')
    hl1 = hamming_loss(gt_list, pred)
    #print('micro result of * 20 is ', p1, r1, f11, hl1)
    print(f'micro result of * 20 is p {p1:.3f}, r {r1:.3f} f1 {f11:.3f}')
    
    p = precision_score(gt_list, pred, average='macro')
    r = recall_score(gt_list, pred, average='macro')
    f1 = f1_score(gt_list, pred, average='macro')
    print(f'macro result of * 20 is p {p:.3f}, r {r:.3f} f1 {f1:.3f}')
    
    f1_ = f1_score(gt_list, pred, average=None)
    if class_index1 != None:
        class_sum = 0
        for i, a in enumerate(class_index1):
            if a == True:
                class_sum += f1_[i]
        print(class_sum / class_index1.long().sum())
    if class_index2 != None:
        class_sum = 0
        for i, a in enumerate(class_index2):
            if a == True:
                class_sum += f1_[i]
        print(class_sum / class_index2.long().sum())
    if class_index3 != None:
        class_sum = 0
        for i, a in enumerate(class_index3):
            if a == True:
                class_sum += f1_[i]
        print(class_sum / class_index3.long().sum())
    if class_index4 != None:
        class_sum = 0
        for i, a in enumerate(class_index4):
            if a == True:
                class_sum += f1_[i]
        print(class_sum / class_index4.long().sum())

    return p_1, p_3, p_5, n_3, n_5, p1, r1, f11, hl1, time.time() - eval_st


def test(args,model, criterion, optimizer, train_dataloader,val_dataloader, test_dataloader, max_epochs, val_iter, max_fail_times):
    
    k = args.k
    start_time = time.time()
            
    features = torch.empty(0,).cuda()
    labels = torch.empty(0,).cuda()
    item = torch.empty(0,).cuda()
    
    for batch_data in train_dataloader:
        _, _, _, iter_label = batch_data
        iter_label = iter_label.cuda()
        prob, iter_feature = get_feature1(model,batch_data)
        features = torch.cat((features, iter_feature))
        labels = torch.cat((labels, iter_label))
        
        count = torch.eq(iter_label, (prob>0.5).long())
        count = count.long().min(1).values
        
        item = torch.cat((item, count))
        
    #print('time is', time.time() - start_time)
    #print('shape pf training is', features.shape, labels.shape, item.shape)
    #print(item.mean())
    
    return features, labels, item


def __main__():
    args = get_args_parser()
    set_random_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(f'Dataset Name: {args.dataset_name}; Setting: {args.setting}; Loss: {args.loss}; Percentage: {args.percentage:.1f}; seed: {args.random_seed}')
    threshold = 0.5
    if args.dataset_name == 'AAPD':
        threshold = 0.4
    elif args.dataset_name == 'RCV1-V2':
        threshold = 0.45

    if args.dataset_name == 'Amazon':
        train_dataset, val_dataset = load_dataset_no_val(args.dataset_name, 'train', args.root, args.setting, args.percentage)
        test_dataset = load_dataset(args.dataset_name, 'test', args.root)
    else:
        train_dataset = load_dataset(args.dataset_name, 'train', args.root, args.setting, args.percentage)
        val_dataset = load_dataset(args.dataset_name, 'val', args.root)
        test_dataset = load_dataset(args.dataset_name, 'test', args.root)

    num_topics = get_num_topics(args.dataset_name)
    criterion = get_criterion(args.setting, args.loss, args)

    #train_dataloader = DataLoader(train_dataset, args.batch_size * 30, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset, args.batch_size * 30, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, 64, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, 64, num_workers=4, pin_memory=True)

    model = Model(num_topics)
    model.load_state_dict(torch.load(f'{args.dataset_name}_{args.random_seed}_{args.alpha}_{args.temp}_{args.beta}.pkl'))
    model = nn.DataParallel(model).cuda()

    optimizer = Adam(model.parameters(), lr=args.lr)
    
    print('Hyper-Parameter settings are:')
    print(f'bsz={args.batch_size};lr={args.lr};alpha={args.alpha};temp={args.temp};beta={args.beta};k={args.k}')

    print('Start test...')
    features, labels, item = test(args,model, criterion, optimizer, train_dataloader, val_dataloader, test_dataloader, args.max_epochs, args.val_iter, args.max_fail_times)
    
    '''for i in [0.2, 0.4, 0.6, 0.8, 1]:
        print('i is',i)
        index = torch.randperm(features.shape[0])
        index = index[: int(i * features.shape[0]) ]
        feature, label = features[index], labels[index]
        tp1, tp3, tp5, tn3, tn5, tp, tr, tf1, thl, tt = eval_222(args, model, test_dataloader, feature, label, item, threshold)'''
    count = labels.sum(0)
    index_1 = count > 4500
    index_2 = (count > 1700) & (count <= 4500)
    index_3 = (count > 870) & (count <= 1700)
    index_4 = (count <= 870)
    #print(index_1, index_2, index_3, index_4)
#     occ = labels @ labels.T
#     for i in range(occ.shape[0]):
#         occ[i,i] = 0
#     print('the results is',occ.max(0).values.min())
    
    #tp1, tp3, tp5, tn3, tn5, tp, tr, tf1, thl, tt = eval_222(args, model, test_dataloader, features, labels, item, threshold, index_1, index_2, index_3, index_4)
    tp1, tp3, tp5, tn3, tn5, tp, tr, tf1, thl, tt = eval_222(args, model, test_dataloader, features, labels, item, threshold)
    print(f'Final: (P@1={tp1:.4f};P@3={tp3:.4f};P@5={tp5:.4f};n@3={tn3:.4f};n@5={tn5:.4f};P={tp:.3f};R={tr:.3f};F1={tf1:.3f}).')


if __name__ == '__main__':
    __main__()