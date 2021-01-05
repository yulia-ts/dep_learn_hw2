import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from CreateDataSet import data_load
from vqa_model import VQA_NET


### Hyper Parameters
num_epochs = 2
batch_size = 50
learning_rate = 0.1
n_answers = 1000
max_questions_len = 26 #30
num_classes = 10
num_layers = 2
hidden_size = 512
data_path = "./cache"
save_path = "./processed"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Function receiving as  an input true labels, scores and number of possible answers, and returning a sparse matrix
def to_categorical(y, num_classes):
    return torch.zeros(num_classes)[y]

def count_soft_acc(pred_exp, label, answer_scores):
    if pred_exp in label:
        for index,val in label:
            if pred_exp == label[index]:
                return answer_scores[index]
    else:
        return 0

def main():
    data_loader = data_load(
        input_vqa_train_npy='train.npy',
        input_vqa_valid_npy='valid.npy',
        max_qst_length= max_questions_len,
        max_num_ans = 10,
        batch_size= batch_size)

    q_voc_size = data_loader['train'].dataset.que_voc.voc_size
    ans_voc_size = data_loader['train'].dataset.ans_voc_size
    print("questions vocabulary size")
    print(q_voc_size)
    print("answers vocabulary size")
    print(ans_voc_size)

    model = VQA_NET(
        embed_size=1024,
        q_voc_size = q_voc_size,
        ans_voc_size=ans_voc_size,
        word_emb_size=300,
        num_layers=num_layers,
        hidden_size=hidden_size).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.image_encoder.vgg_model.parameters()) + list(model.question_encoder.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters()), lr=learning_rate)
    ##step size = period of learning decay
    ##gamma - multiplicative factor of learning rate decay.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(num_epochs):
        print("epoch")
        for phase in ['train', 'validation']:
            running_loss = 0.0
            running_accuracy = 0
            batch_step_size = len(data_loader[phase].dataset) / batch_size
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            for batch_idx, batch_sample in enumerate(data_loader[phase]):
                image = batch_sample['image'].to(device)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label']
                answer_scores = batch_sample['answer_scores']  # not tensor, list.

                print(image.shape())
                print(question.shape())
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    output = model(image, question)  # [batch_size, ans_vocab_size=1000]
                    _, pred_exp = torch.max(output, 1)  # [batch_size]
                    loss = loss_function(output, label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # unk asnwer is not accepted by our model
                running_loss += loss.item()
                running_accuracy += count_soft_acc(pred_exp, label, answer_scores)
                # Print the average loss in a mini-batch.
                if batch_idx % 10 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                          .format(phase.upper(), epoch + 1, num_epochs, batch_idx, int(batch_step_size),
                                  loss.item()))
        # Print the average loss and accuracy in an epoch.
        epoch_loss = running_loss / batch_step_size
        epoch_acc = running_accuracy.double() / len(data_loader[phase].dataset)


        print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc(Exp2): {:.4f} \n'
              .format(phase.upper(), epoch + 1, num_epochs, epoch_loss, epoch_acc))

if __name__ == '__main__':
    main()