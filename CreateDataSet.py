import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image


""" The purpose of this file:
 class preparing for  VQA dataset"""
### Hyper Parameters
batch_size = 64
n_answers = 1022 #1021+1 for unk
max_questions_len = 26



data_path = "./cache1"
save_path = "./processed1"

def get_answers_matrix(ans_voc_size,labels, scores):
    ## updates weights according to scores
    ans_hot_vec = torch.zeros(ans_voc_size, dtype=torch.float)
    for i,v in enumerate(labels):
        score = scores[i]
        ans_hot_vec[v] = score
    return ans_hot_vec

class VQADataset(Dataset):
    def __init__(self, input_f_type, max_q_len=max_questions_len , transform = None):
        print("Creating data set")
        # Set variables
        self.vqa = np.load('cache1'+'/'+input_f_type+'.npy', allow_pickle=True)
        if input_f_type == 'validation':
            self.question_vocab_path = 'val_questions.txt'
        else:
            self.question_vocab_path = 'train_questions.txt'
        self.que_voc = QVocabCreate('cache1'+'/'+self.question_vocab_path)
        self.ans2label_path = os.path.join('cache1', 'trainval_ans2label_final.pkl')
        self.label2ans_path = os.path.join('cache1', 'trainval_label2ans_final.pkl')
        if input_f_type == 'validation':
            self.target_f = os.path.join('cache1','val_target_final.pkl')
        else:
            self.target_f = os.path.join('cache1', 'train_target_final.pkl')
        self.ans2label = pickle.load(open(self.ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(self.label2ans_path, 'rb'))
        self.ans_voc_size = len(self.ans2label)+1  #+1 for un
        self.max_questions_len = max_q_len
        #self.max_num_answers = 10 ##
        self.transform = transform
        #load answer_idxes per question + scores
        with open(self.target_f, 'rb') as pickle_file:
            target = pickle.load(pickle_file)
        self.target_dict = {}
        for q in target:
            self.target_dict[q['question_id']] = {
                'labels': q['labels'],
                'scores': q['scores'],
                'chosen_label': q['multiple_choice_label']

            }
    def __getitem__(self, index):
        vqa = self.vqa
        qst_vocab = self.que_voc
        max_q_length = self.max_questions_len
        transform = self.transform
        image = Image.open(vqa[index]['img_path'])
        question_id = vqa[index]['question_id']
        qst2idc = np.array([qst_vocab.word2lbl('<pad>')] * max_q_length)  # padded with '<pad>' in 'ans_vocab'
        qst2idc[:len(vqa[index]['question_labels'])] = [qst_vocab.word2lbl(w) for w in vqa[index]['question_labels']]
        entry = {'image': image, 'question': qst2idc}
        answers_matrix = get_answers_matrix(self.ans_voc_size, self.target_dict[question_id]['labels'], self.target_dict[question_id]['scores'])
        entry['answer_label'] = np.array(self.target_dict[question_id]['chosen_label'])
        entry['answer_labels'] = self.target_dict[question_id]['labels']
        #entry['answer_scores'] = self.target_dict[question_id]['scores']
        #if len(self.target_dict[question_id]['labels']) < 1:
        #    entry['answer_labels'] = [0]
         #   entry['answer_scores'] = [0]
        entry['answer_mat'] = answers_matrix
        if transform:
            entry['image'] = transform(entry['image'])
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

def data_load(batch_size=batch_size):
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
            shuffle=True, )
        for phase in ['train', 'validation']}

    return data_loader
