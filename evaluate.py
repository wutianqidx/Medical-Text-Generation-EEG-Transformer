import warnings
import nltk
warnings.filterwarnings("ignore")
import os
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import sentence_bleu
from data import EEGDataset, CollateWrapper
from model import EEGtoReport

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='evaluate on train dataset')
    group.add_argument('--val', action='store_true', help='evaluate on val dataset')
    group.add_argument('--test', action='store_true', help='evaluate on test dataset')
    parser.add_argument('--run', default='', help='Continue training on runX. Eg. --run=run1')
    args = parser.parse_args()
    if args.train:
        args.pkl = "dataset/eeg_text_train.pkl"
    elif args.val:
        args.pkl = "dataset/eeg_text_val.pkl"
    elif args.test:
        args.pkl = "dataset/eeg_text_test.pkl"
    args.base_output = "checkpoint/"
    args.checkpoint = "eeg_text_checkpoint"

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = 1337
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.run) == 0:
        run_count = len(os.listdir(args.base_output))
        args.run = 'run{}'.format(run_count-1)
    args.base_output = os.path.join(args.base_output, args.run)
    return args

def run_evaluation(args, checkpoint, dataset, train_loader):
    model = EEGtoReport(eeg_epoch_max=checkpoint['dataset']['max_len'],
                        report_epoch_max=checkpoint['dataset']['max_len_t'],
                        vocab_size=checkpoint['dataset']['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(args.device)
    model.eval()
    with torch.no_grad():
        metrics = defaultdict(list)
        for idx, (input, target_i, target, length, length_t) in enumerate(train_loader):
            output = model(input, target_i, length, length_t, train=False)
            output = pack_padded_sequence(output, length_t, enforce_sorted=False).data.argmax(dim=1) # temporary
            target = pack_padded_sequence(target, length_t, enforce_sorted=False).data.view(-1)

            output = " ".join([ dataset.ixtoword[int(i)] for i in output])
            output.replace('<sep>','|')
            output.replace('<punc>', ',')
            output.replace('<end>','')
            output.replace('<unk>','unknown')#output idx to string

            target = " ".join([ dataset.ixtoword[int(i)] for i in target])
            target.replace('<sep>','|')
            target.replace('<punc>', ',')
            target.replace('<end>','')
            target.replace('<unk>','unknown')#target idx to string


            meteor = meteor_func(output, target)
            #cider = cider_func(output, target)
            bleu_1 = bleu_1_func(output, target)
            bleu_2 = bleu_2_func(output, target)
            bleu_3 = bleu_3_func(output, target)
            bleu_4 = bleu_4_func(output, target)

            metrics['meteor'].append(meteor)
            #metrics['cider'].append(cider)
            metrics['bleu_1'].append(bleu_1)
            metrics['bleu_2'].append(bleu_2)
            metrics['bleu_3'].append(bleu_3)
            metrics['bleu_4'].append(bleu_4)

        meteor_impression, meteor_description = list(zip(*metrics['meteor']))
        #cider_impression, cider_description = list(zip(*metrics['cider']))
        bleu_1_impression, bleu_1_description = list(zip(*metrics['bleu_1']))
        bleu_2_impression, bleu_2_description = list(zip(*metrics['bleu_2']))
        bleu_3_impression, bleu_3_description = list(zip(*metrics['bleu_3']))
        bleu_4_impression, bleu_4_description = list(zip(*metrics['bleu_4']))

        print("meteor impression: ", np.mean(meteor_impression))
        #print("cider impression: ", np.mean(cider_impression))
        print("bleu_1 impression: ", np.mean(bleu_1_impression))
        print("bleu_2 impression: ", np.mean(bleu_2_impression))
        print("bleu_3 impression: ", np.mean(bleu_3_impression))
        print("bleu_4 impression: ", np.mean(bleu_4_impression))

        print("meteor description: ", np.mean(meteor_description))
        #print("cider description: ", np.mean(cider_description))
        print("bleu_1 description: ", np.mean(bleu_1_description))
        print("bleu_2 description: ", np.mean(bleu_2_description))
        print("bleu_3 description: ", np.mean(bleu_3_description))
        print("bleu_4 description: ", np.mean(bleu_4_description))

def meteor_func(output, target):
    """TO DO: calculate metric between output and target for Impression Section & Detail Description respectively
    Output and target will have different length. So do their two parts.
    Args:
    output: model prediction(Impression Section & Detail Description seperated by 3:'<sep>')
    target: ground truth(Impression Section & Detail Description seperated by 3:'<sep>')
    Return:
    metric_impression, metric_description
    """
    output_impression, output_description = output.split('|')
    target_impression, target_description = target.split('|')
    return nltk.translate.meteor_score.meteor_score([target_impression],output_impression), \
           nltk.translate.meteor_score.meteor_score([target_description],output_description)

def cider_func(output, target):
    """TO DO:"""
    return 0, 0

def bleu_1_func(output, target):
    """TO DO:"""
    output_impression, output_description = output.split('|')
    target_impression, target_description = target.split('|')
    weights = (1.,)
    return sentence_bleu([target_impression],output_impression,weights), \
           sentence_bleu([target_description],output_description,weights)

def bleu_2_func(output, target):
    """TO DO:"""
    output_impression, output_description = output.split('|')
    target_impression, target_description = target.split('|')
    weights = (1./2.,1./2.)
    return sentence_bleu([target_impression], output_impression, weights), \
           sentence_bleu([target_description], output_description, weights)

def bleu_3_func(output, target):
    """TO DO:"""
    output_impression, output_description = output.split('|')
    target_impression, target_description = target.split('|')
    weights = (1./3.,1./3.,1./3.)
    return sentence_bleu([target_impression], output_impression, weights), \
           sentence_bleu([target_description], output_description, weights)

def bleu_4_func(output, target):
    """TO DO:"""
    output_impression, output_description = output.split('|')
    target_impression, target_description = target.split('|')
    weights = (1. / 4., 1. / 4., 1. / 4., 1. / 4.)
    return sentence_bleu([target_impression], output_impression, weights), \
           sentence_bleu([target_description], output_description, weights)

def main(args):
    # works on nltk version 3.4.5, python 3.5
    nltk.download('wordnet')
    tic = time.time()
    dataset = EEGDataset(args.pkl)
    print("dataset len:", len(dataset))
    checkpoint_path = os.path.join(args.base_output, args.checkpoint)
    checkpoint = torch.load(checkpoint_path)
    collate_wrapper = CollateWrapper(max_len=checkpoint['dataset']['max_len'], max_len_t=checkpoint['dataset']['max_len_t'])
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_wrapper)
    run_evaluation(args, checkpoint, dataset, train_loader)
    print('[{:.2f}] Finish evaluation'.format(time.time() - tic))

if __name__ == '__main__':
    args = get_args()
    main(args)
