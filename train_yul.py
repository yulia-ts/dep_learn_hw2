import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from CreateDataSet import data_load
#from models import MyNet


### Hyper Parameters
num_epochs = 2
batch_size = 50
learning_rate = 0.1
n_answers = 1000
max_questions_len = 26 #30
num_classes = 10


data_path = "./cache"
save_path = "./processed"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    data_loader = data_load(
        input_vqa_train_npy='train.npy',
        input_vqa_valid_npy='valid.npy',
        max_qst_length= max_questions_len,
        max_num_ans = 10,
        batch_size= batch_size)

    q_voc_size = data_loader['train'].dataset.que_voc.voc_size
    for epoch in range(num_epochs):

        for phase in ['train', 'valid']:
            running_loss_train = 0.0
            running_corr_exp1 = 0
            running_corr_exp2 = 0
            batch_step_size = len(data_loader[phase].dataset) / batch_size
            """if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()"""

            for batch_idx, batch_sample in enumerate(data_loader[phase]):
                image = batch_sample['image'].to(device)
                print("image")
                print(image)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].to(device)
                #multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

                #optimizer.zero_grad()


if __name__ == '__main__':
    main()