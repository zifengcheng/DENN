import argparse
import random
import time
import warnings

from torch.optim import Adam
from torch.utils.data import DataLoader

from criterions import *
from datasets import aapd_topics, rcv1v2_topics, load_dataset, Amazon_topics, load_dataset_no_val, EUR_topics
from engine import train_iter, eval, get_feature,eval_new
from models import Model
import os


def get_args_parser():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', default=1, type=int)

    parser.add_argument('--root', default='../../DataSet/AAPD', type=str)
    parser.add_argument('--dataset_name', default='AAPD', type=str, help='AAPD or RCV1-V2')
    parser.add_argument('--setting', default='F', type=str, help='{F, P, PO, OP, OPON}, not used in this paper')
    parser.add_argument('--percentage', default=1.0, type=float, help='not used in this paper')
    
    parser.add_argument('--gpu', default='0,1', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epochs', default=50, type=int)
    #parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--val_iter', default=200, type=int)
    parser.add_argument('--max_fail_times', default=10, type=int)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--beta', default=0.0, type=float, help='not used in this paper')
    parser.add_argument('--k', default=5, type=int)
    
    parser.add_argument('--loss', default='BCE', type=str, help='{BCE, AN, WAN, AN-LS, EPR, ROLE}, not used in this paper')

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


def training(args,model, criterion, optimizer, train_dataloader, train_dataloader1,val_dataloader, test_dataloader, max_epochs, val_iter, max_fail_times):
    max_val_res = val_counter = fail_counter = 0
    tp1 = tp3 = tp5 = tn3 = tn5 = tp = tr = tf1 = thl = 0
    train_timer = time.time()
    for ep_idx in range(max_epochs):
        losses = []
        idx = 0
        for batch_data in train_dataloader:
            idx += 1
            loss , contrastive_loss = train_iter(args, model, criterion, optimizer, batch_data)
            if idx == 1:
                print(loss, contrastive_loss)
            losses.append(loss)

            if val_counter == val_iter:
                print(f'Finish one iter training; AvgLoss={sum(losses) / len(losses):.6f}; Time: {time.time() - train_timer:.2f}s; Evaluating on the val dataset...')
                
                vp1, vp3, vp5, vn3, vn5, vp, vr, vf1, vhl, vt = eval(model, val_dataloader)
                print(f'Val:  (P@1={vp1:.4f};P@3={vp3:.4f};P@5={vp5:.4f};n@3={vn3:.4f};n@5={vn5:.4f};P={vp:.4f};R={vr:.4f};F1={vf1:.4f};HL={vhl:.4f});Time={vt:.2f}s;')
                    
                if vf1 >= max_val_res:
                    torch.save(model.module.state_dict(), f'{args.dataset_name}_{args.random_seed}_{args.alpha}_{args.temp}_{args.beta}.pkl')
                    print(':-) Got best model! Evaluating on the test dataset...')
                    tp1, tp3, tp5, tn3, tn5, tp, tr, tf1, thl, tt = eval(model, test_dataloader)
                    print(f'Test: (P@1={tp1:.4f};P@3={tp3:.4f};P@5={tp5:.4f};n@3={tn3:.4f};n@5={tn5:.4f};P={tp:.4f};R={tr:.4f};F1={tf1:.4f};HL={thl:.4f});Time={tt:.2f}s.\n')

                    max_val_res = vf1
                    fail_counter = 0

                else:
                    fail_counter += 1
                    print(f':-( Not best model! Fail time: {fail_counter} / {max_fail_times}.\n')

                if fail_counter == max_fail_times:
                    print(f':-( Max Fail Times; Break the loop of training.')
                    
                    return tp1, tp3, tp5, tn3, tn5, tp, tr, tf1, thl

                val_counter = 0
                train_timer = time.time()
                losses = []
            else:
                val_counter += 1
        
    return tp1, tp3, tp5, tn3, tn5, tp, tr, tf1, thl


def __main__():
    args = get_args_parser()
    set_random_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(f'Dataset Name: {args.dataset_name}; Setting: {args.setting}; Loss: {args.loss}; Percentage: {args.percentage:.1f};seed={args.random_seed}')

    if args.dataset_name == 'Amazon':
        train_dataset, val_dataset = load_dataset_no_val(args.dataset_name, 'train', args.root, args.setting, args.percentage)
        test_dataset = load_dataset(args.dataset_name, 'test', args.root)
    else:
        train_dataset = load_dataset(args.dataset_name, 'train', args.root, args.setting, args.percentage)
        val_dataset = load_dataset(args.dataset_name, 'val', args.root)
        test_dataset = load_dataset(args.dataset_name, 'test', args.root)

    num_topics = get_num_topics(args.dataset_name)
    criterion = get_criterion(args.setting, args.loss, args)

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    train_dataloader1 = DataLoader(train_dataset, args.batch_size*25, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, 32, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, 32, num_workers=4, pin_memory=True)

    model = Model(num_topics)
    model = nn.DataParallel(model).cuda()

    optimizer = Adam(model.parameters(), lr=args.lr)
    
    print('Hyper-Parameter settings are:')
    print(f'bsz={args.batch_size};lr={args.lr};alpha={args.alpha};temp={args.temp};beta={args.beta};k={args.k};seed={args.random_seed}')

    print('Start training...')
    res = training(args,model, criterion, optimizer, train_dataloader, train_dataloader1, val_dataloader, test_dataloader, args.max_epochs, args.val_iter, args.max_fail_times)
    tp1, tp3, tp5, tn3, tn5, tp, tr, tf1, thl = res
    print(f'Final: (P@1={tp1:.4f};P@3={tp3:.4f};P@5={tp5:.4f};n@3={tn3:.4f};n@5={tn5:.4f};P={tp:.4f};R={tr:.4f};F1={tf1:.4f};HL={thl:.4f}).')


if __name__ == '__main__':
    __main__()
