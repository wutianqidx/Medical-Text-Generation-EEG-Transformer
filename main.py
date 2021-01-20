import warnings
warnings.filterwarnings("ignore")
import os
import copy
import argparse
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data import EEGDataset, collate_wrapper
from model import EEGtoReport, loss_func

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=3, help="Batch size (default=32)")
    parser.add_argument('--run', default='', help='Continue training on runX. Eg. --run=run1')
    args = parser.parse_args()
    args.base_output = "checkpoint/"
    args.checkpoint = "eeg_text_checkpoint"
    args.eval_every = 10
    args.learning_rate = 0.001

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = 1337
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.base_output, exist_ok=True)
    if len(args.run) == 0:
        run_count = len(os.listdir(args.base_output))
        args.run = 'run{}'.format(run_count)
    args.base_output = os.path.join(args.base_output, args.run)
    os.makedirs(args.base_output, exist_ok=True)
    return args

def run_training(args, dataset, train_loader):
    checkpoint_path = os.path.join(args.base_output, args.checkpoint)
    dataset_dict = copy.deepcopy(dataset.__dict__)
    del dataset_dict['data']
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = EEGtoReport(eeg_epoch_max=checkpoint['dataset']['max_len'],
                            report_epoch_max=checkpoint['dataset']['max_len_t'],
                            vocab_size=checkpoint['dataset']['vocab_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(
            model.parameters(),
            lr=checkpoint['args']['learning_rate']
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('epochs={},batch_size={},run={},lr={} loss:{}'.format(
        checkpoint['epoch'],checkpoint['args']['batch_size'],checkpoint['args']['run'],
        checkpoint['args']['learning_rate'],checkpoint['loss']))
        epoch_start = checkpoint['epoch']
    else:
        model = EEGtoReport(eeg_epoch_max=dataset.max_len,
                            report_epoch_max=dataset.max_len_t,
                            vocab_size=dataset.vocab_size)
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate
        )
        epoch_start = 0

    model = model.to(args.device)
    model.train()
    metrics = []
    for epoch in range(args.epochs):
        epoch_metrics = []
        for batch_ndx, (input, target_i, target, length, length_t) in enumerate(train_loader):
            output = model(input, target_i, length, length_t)
            loss = loss_func(output, target, length_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_metrics.append(loss.item())
        metrics.append(np.mean(epoch_metrics))

        if epoch % args.eval_every == args.eval_every-1 or epoch == args.epochs-1:
            torch.save({
            'epoch': epoch_start+epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics[epoch],
            'dataset': dataset_dict,
            'args': vars(args)
            }, checkpoint_path)
            print('Epoch {}:'.format(args.epochs))
            print('epochs={},batch_size={},run={},lr={} loss:{}'.format(
                epoch_start+epoch+1,args.batch_size,args.run,args.learning_rate, metrics[epoch]))

def main(args):
    tic = time.time()
    dataset = EEGDataset("dataset/eeg_text_train.pkl")
    print("dataset len:", len(dataset))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_wrapper)
    run_training(args, dataset, train_loader)
    print('[{:.2f}] Finish training'.format(time.time() - tic))

if __name__ == '__main__':
    """TO DO: train and overfit on one sample at first"""
    args = get_args()
    main(args)
