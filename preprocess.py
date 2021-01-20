import os
import numpy as np
import pandas as pd
import pickle
import argparse
from glob import glob
import pyedflib
import string
from collections import defaultdict
import random

import nltk
from nltk.tokenize import word_tokenize
import re

def preprocess(file_path):
    """TO DO: Download dataset as instruction in README.md at first
    Pick correct file_path and traverse through it

    Preprocess files in file_path
    Arg:
    file_path: files path
    Return:
    [(eeg, report), ...], word_bag, unified frequency
    """
    data = []
    word_bag = defaultdict(int)
    freq = 250
    for dir in os.listdir(file_path):
        dir = os.path.join(file_path, dir)

        txt_f = glob(dir+"/*.txt")[0]
        report = parse_txt(txt_f, word_bag)
        print('report', report)

        edf_f_list = glob(dir+"/*.edf")
        eeg_raw, freq_raw = read_edf(edf_f_list)

        eeg = resize_eeg(eeg_raw, freq_raw, freq)

        data.append((eeg, report))
    return data, word_bag, freq

def parse_txt(txt_f, word_bag):
    """BY PEIYAO: parse report IMPRESSION & DESCRIPTION OF THE RECORD as string list, update word bag
    Some keywords are Summary of Findings (Interpretation) & Description, not IMPRESSION & DESCRIPTION OF THE RECORD. We should consider that later.
    "In addition, we also process the reports by tokenizing and converting to lower-cases."  --EEGtoText: Learning to Write Medical Reports from EEG Recordings
    Arg:
    txt_f: report.txt path
    word_bag: {word: frequency} which should be updated
    Return:
    [IMPRESSION, DESCRIPTION OF THE RECORD]
    """
    f = open(txt_f, "r")
    lines = f.readlines()

    ### IMPRESSION has several lines
    ### DESCRIPTION OF THE RECORD is one lines
    impression_prefix = 'IMPRESSION:'
    impression = ''
    impression_flag = False

    description_prefix = 'DESCRIPTION OF THE RECORD:'
    description = ''

    clinical_prefix = 'CLINICAL CORRELATION:'

    for line in lines:
    	line = line.lstrip()
    	line = line.rstrip()
    	if line.startswith(description_prefix):
    		description = line
    	elif line.startswith(impression_prefix):
    		impression_flag = True
    	elif line.startswith(clinical_prefix):
    		impression_flag = False

    	if impression_flag:
    		impression += line + ' '

    ### Interpretation and Description are several lines
    if description == '':
    	impression_prefix = 'Interpretation:'
    	impression_flag = False

    	description_prefix = 'Description:'
    	description_flag = False

    	summary_prefix = 'Summary of Findings:'

    	for line in lines:
	    	line = line.lstrip()
	    	line = line.rstrip()
	    	if line.startswith(description_prefix):
	    		description_flag = True
	    	elif line.startswith(impression_prefix):
	    		impression_flag = True
	    	elif line.startswith(summary_prefix):
	    		description_flag = False

	    	if impression_flag:
	    		impression += line + ' '

	    	if description_flag:
	    		description += line + ' '

    description = description[len(description_prefix):]
    impression = impression[len(impression_prefix):]

    description = description.lower()
    impression = impression.lower()

    description = re.sub(r'[^\w\s\-]', ' <punc> ', description)
    impression = re.sub(r'[^\w\s\-]', ' <punc> ', impression)

    description_token = description.split()
    impression_token = impression.split()

    if description_token[-1] == '<punc>':
    	description_token = description_token[:-1]

    if impression_token[-1] == '<punc>':
    	impression_token = impression_token[:-1]

    description_token.append('<end>')
    impression_token.append('<sep>')

    for token in description_token:
    	word_bag[token] += 1

    for token in impression_token:
    	word_bag[token] += 1

    impression_token.extend(description_token)
    return impression_token

def read_edf(edf_f_list):
    """BY BEICHEN: read EEG recording and calculate value for each channel
    Arg:
    edf_f_list: eeg.edf file list
    Return:
    np.array(18, SampleLength), sample frequency
    16+2 channels(5 chains)
    channels: ('EEG FP1-REF' - 'EEG F7-REF'), ('EEG F7-REF' - 'EEG T3-REF'), ...
    chains: 'EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF'
            'EEG FP1-REF','EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG O1-REF'
            'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
            'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF'
            'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF', 'EEG O2-REF'
    """
    eeg_raw = []
    freq_raw = 0
    for edf_f in edf_f_list:
        # flag for valid 18 channels existance
        valid_flag = True
        f = pyedflib.EdfReader(edf_f)
        # # print("birthdate patientname technician", len(f.birthdate), f.birthdate, f.patientname, f.technician)
        # #print("getFileDuration(self)", f.getFileDuration())
        # #print("samplefrequency", f.samplefrequency(0), f.samplefrequency(28))
        # #print("samples_in_datarecord", f.samples_in_datarecord(0), f.samples_in_datarecord(28))
        print("datarecords_in_file: {}s*{}  {}".format(f.datarecord_duration, f.datarecords_in_file, f.file_duration))
        print("startdate: {}/{}/{} {}:{}:{}".format(f.startdate_month, f.startdate_day, f.startdate_year,
                                                    f.starttime_hour, f.starttime_minute, f.starttime_second))
        print("signals_in_file", f.signals_in_file)
        print("getSampleFrequencies", len(f.getSampleFrequencies()), f.getSampleFrequencies())
        print("samples_in_file", f.samples_in_file(0), f.samples_in_file(1), f.samples_in_file(27), f.samples_in_file(28), f.samples_in_file(29))
        signal_labels = f.getSignalLabels()
        label_used = ['EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF',
                                    'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                    'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF',
                                'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF']
        channel_tup = [('FP1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),
                       ('FP1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),
                       ('FZ','CZ'),('CZ','PZ'),('FP2','F4'),('F4','C4'),
                       ('C4','P4'),('P4','O2'),('FP2','F8'),('F8','T4'),
                       ('T4','T6'),('T6','O2')]
        sigbufs = {}
        print("signal_labels",signal_labels)
        for i,label in enumerate(signal_labels):
            if label in label_used:
                sigbufs[label] = f.readSignal(i)

        for key in sigbufs:
            print(len(sigbufs[key]))
            break
        chnbufs = np.zeros((18, f.getNSamples()[0]))
        for idx,(first,second) in enumerate(channel_tup):
            try:
                chnbufs[idx] = sigbufs['EEG '+ first +'-REF'] - sigbufs['EEG '+ second +'-REF']
            except:
                print("in "+ edf_f +" pair EEG "+ first +'-REF, EEG '+ second +'-REF not found')
                valid_flag = False
                break
        if not valid_flag:
            f.close()
            continue
        if len(eeg_raw) == 0:
            eeg_raw = chnbufs
        else:
            eeg_raw = np.concatenate((eeg_raw,chnbufs),axis=1)
        if freq_raw != 0 and freq_raw != f.samplefrequency(0):
            raise Exception("inconsistent frequency between "+ freq_raw + "and " +f.samplefrequency(0) )
        freq_raw = f.samplefrequency(0)
        f.close()
    return eeg_raw, freq_raw

def resize_eeg(eeg_raw, freq_raw, freq):
    """BY BEICHEN: resize eeg_raw from (18, freq_raw) per second to (18, freq) per second
    Need to find library to do linear interpolation
    Args:
    eeg_raw: np.array(18, SampleLength)
    freq_raw: frequency of sample
    freq: frequency we want to uniform to
    Return:
    np.array(18, SampleLengthUniformed)
    """
    #total second of eeg_raw, need all file have same frequency
    seconds = eeg_raw.shape[1]/freq_raw
    time_stamp_prev = np.arange(0,seconds,1./freq_raw)
    time_stamp_new = np.arange(0,seconds,1./freq)
    eeg = []
    for channel in eeg_raw:
        new_channel = []
        for time_item in time_stamp_new:
            new_channel.append(np.interp(time_item,time_stamp_prev,channel))
        eeg.append(new_channel)
    return np.array(eeg)

if __name__ == '__main__':
    file_path = "dataset/"
    data, word_bag, freq = preprocess(file_path)
    random.shuffle(data)
    train_len = int(len(data)*0.6)
    val_len = int(len(data)*0.2)
    pickle.dump((data[:train_len], word_bag, freq), open("dataset/eeg_text_train.pkl", "wb"))
    pickle.dump((data[train_len:train_len+val_len], word_bag, freq), open("dataset/eeg_text_val.pkl", "wb"))
    pickle.dump((data[train_len+val_len:], word_bag, freq), open("dataset/eeg_text_test.pkl", "wb"))
