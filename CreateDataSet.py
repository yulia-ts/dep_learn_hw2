import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image


""" The purpose of this file:
 class preparing preparing VQA dataset"""
### Hyper Parameters

num_epochs = 2
batch_size = 50
learning_rate = 0.1
n_answers = 1000
max_questions_len = 26 #30
num_classes = 10


data_path = "./cache"
save_path = "./processed"



class VQADataset(Dataset):
    def __init__(self, input_f_type, max_q_len = max_questions_len , transform = None):
        print("Creating data set")
        # Set variables
        self.transform = transforms.Compose([transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # mean=0; std=1
        ])
        self.vqa = np.load('cache'+'/'+input_f_type+'.npy', allow_pickle=True)
        if input_f_type == 'validation':
            self.question_vocab_path = 'val_questions.txt'
        else:
            self.question_vocab_path = 'train_questions.txt'
        self.que_voc = QVocabCreate('cache'+'/'+self.question_vocab_path)
        self.ans2label_path = os.path.join('cache', 'trainval_ans2label.pkl')
        self.label2ans_path = os.path.join('cache', 'trainval_label2ans.pkl')
        if input_f_type == 'validation':
            self.target = os.path.join('cache','val_target.pkl')
        else:
            self.target = os.path.join('cache', 'train_target.pkl')
        self.ans2label = pickle.load(open(self.ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(self.label2ans_path, 'rb'))
        self.max_questions_len = max_q_len
        self.max_num_answers = 10 ##TODO to check this
        self.transform = transform
    def __getitem__(self, index):
        vqa = self.vqa
        target_f = self.target
        qst_vocab = self.que_voc
        max_q_length = self.max_questions_len
        #max_num_ans = self.max_num_answers
        transform = self.transform
        image = vqa[index]['img_path']
        question_id = vqa[index]['question_id']
        qst2idc = np.array([qst_vocab.word2lbl('<pad>')] * max_q_length)  # padded with '<pad>' in 'ans_vocab'
        qst2idc[:len(vqa[index]['question_labels'])] = [qst_vocab.word2lbl(w) for w in vqa[index]['question_labels']]
        entry = {'image': image, 'question': qst2idc}
        #load answer_idxes per question + scores
        with open(target_f, 'rb') as pickle_file:
            target = pickle.load(pickle_file)
        target_dict = {}
        for q in target:
            target_dict[q['question_id']] = {
            'labels': q['labels'],
            'scores': q['scores']
        }

        entry['answer_label'] = target_dict[question_id]['labels']
        entry['answer_scores'] = target_dict[question_id]['scores']

        if transform:
            entry['image'] = transform(entry['image'])
        print(entry)
        return entry

    def __len__(self):
        return len(self.vqa)


class QVocabCreate:
##need thiss class for setting questions vocabulary
    def __init__(self, voc_f):

        self.voc = self.load_vocab(voc_f)
        self.voc_size = len(self.voc)
        self.voc2lbl = {voc: idx for idx, voc in enumerate(self.voc)}


    def load_vocab(self, vocab_file):

        with open(vocab_file) as f:
            vocab = [v.strip() for v in f]

        return vocab

    def word2lbl(self, vocab):

        if vocab in self.voc2lbl:
            return self.voc2lbl[vocab]
        else:
            return self.voc2lbl['<unk>']

    def lbl2word(self, idx):

        return self.voc[idx]

def data_load( input_vqa_train_npy,input_vqa_valid_npy, max_qst_length, max_num_ans, batch_size=batch_size):
    transform = {
        phase: transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))])
        for phase in ['train', 'validation']}

    vqa_dset = {
        'train': VQADataset(
            input_f_type='train',
            transform=transform['train']),
        'validation': VQADataset(
            input_f_type='validation',
            transform=transform['validation'])}

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dset[phase],
            batch_size=batch_size,
            shuffle=True)
        for phase in ['train', 'validation']}

    return data_loader
