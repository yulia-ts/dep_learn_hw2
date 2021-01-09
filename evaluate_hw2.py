from __future__ import unicode_literals, print_function, division
import os
import numpy as np
import torch
from CreateDataSet import data_load
from vqa_model import VQA_NET
import time
from soft_acc import count_soft_acc


### Hyper Parameters
batch_size = 64
num_all_pred_answer = 1021 + 1  # +1 for unknown answer
max_questions_len = 26  # 30
num_layers = 2
hidden_size = 512
data_path = "./cache1"
save_path = "./processed1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""def count_soft_acc(pred_exp, label_vec):
    ##function input: predicted answer, true labels-scores matrix
    ##output: score/accuracy
    acc = 0.000
    for i, v in enumerate(pred_exp):
        acc += label_vec[i][v]
    return acc"""

# monochromatic images were resized to RGB and have  in_channels = 3


def evaluate_hw2():

    print("validating the model")
    data_loader = data_load(batch_size=batch_size)
    q_voc_size = data_loader['train'].dataset.que_voc.voc_size
    ans_voc_size = data_loader['train'].dataset.ans_voc_size
    print("questions vocabulary size")
    print(q_voc_size)
    print("answers vocabulary size")
    print(ans_voc_size)
    start_time = time.time()
    model = VQA_NET(
        embed_size=1024,
        q_voc_size=q_voc_size,
        ans_voc_size=ans_voc_size,
        word_emb_size=300,
        num_layers=num_layers,
        hidden_size=hidden_size).to(device)

    model.load_state_dict(torch.load('model2-dict_128_final.pkl'))
    model.eval()


    for phase in ['validation']:
        running_accuracy = 0
        batch_step_size = len(data_loader[phase].dataset) / batch_size
        for batch_idx, batch_sample in enumerate(data_loader[phase]):
            image = batch_sample['image'].to(device)
            question = batch_sample['question'].to(device)
            label_vec = batch_sample['answer_mat'].to(device)
            label = batch_sample['answer_label']
            output = model(image, question)  # [batch_size, ans_voc_size=1022]
            _, pred_exp = torch.max(output, 1)  # [batch_size]
            # unk asnwer is not accepted by our model
            acc_step = count_soft_acc(pred_exp, label_vec)
            running_accuracy += acc_step
            # Print the average loss in a mini-batch.
            if batch_idx % 400 == 0:
                    print('| {} SET | Step [{:04d}/{:04d}], Accuracy: {:.4f}, '
                          .format(phase.upper(), batch_idx, int(batch_step_size), acc_step))
        print("total validation time minutes")
        print((time.time() - start_time) / 60)
        total_acc = running_accuracy.double() / len(data_loader[phase].dataset)
        print('| {} SET | Total Accuracy: {:.4f} \n'
                  .format(phase.upper(),  total_acc))

    return total_acc


if __name__ == '__main__':
    print(evaluate_hw2())
