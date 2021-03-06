import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from CreateDataSet import data_load
from vqa_model import VQA_NET
import time
from matplotlib import pyplot as plt

### Hyper Parameters
num_epochs = 20
batch_size = 64
learning_rate = 0.001
num_all_pred_answer = 1021 + 1  # +1 for unknown answer
max_questions_len = 26
num_classes = 10
num_layers = 2
hidden_size = 512
data_path = "./cache1"
save_path = "./processed1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_soft_acc(pred_exp, label_vec):
    ##function input: predicted answer, true labels-scores matrix
    ##output: score/accuracy
    acc = 0.000
    for i, v in enumerate(pred_exp):
        acc += label_vec[i][v]
    return acc


def main():
    data_loader = data_load(batch_size=batch_size)
    q_voc_size = data_loader['train'].dataset.que_voc.voc_size
    ans_voc_size = data_loader['train'].dataset.ans_voc_size
    print("questions vocabulary size")
    print(q_voc_size)
    print("answers vocabulary size")
    print(ans_voc_size)
    start_time = time.time()
    # Loss list
    train_loss = []
    test_loss = []

    # Score list
    train_accuracy = []
    test_accuracy = []

    model = VQA_NET(
        embed_size=1024,
        q_voc_size=q_voc_size,
        ans_voc_size=ans_voc_size,
        word_emb_size=300,
        num_layers=num_layers,
        hidden_size=hidden_size).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
          list(model.image_encoder.fc.parameters()) + list(model.question_encoder.parameters()) + list(
           model.fc1.parameters()) + list(model.fc2.parameters()), lr=learning_rate)
    ##step size = period of learning decay
    ##gamma - multiplicative factor of learning rate decay.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(num_epochs):
        epoch_time = time.time()
        print("epoch")
        print(epoch + 1)
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
                label = batch_sample['answer_label'].to(device)
                label_vec = batch_sample['answer_mat'].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(image, question)  # [batch_size, ans_voc_size=1022]
                    loss = loss_function(output, label)
                    _, pred_exp = torch.max(output, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # unk answer is not accepted by our model
                running_loss += loss.item()
                acc_step = count_soft_acc(pred_exp, label_vec)
                running_accuracy += acc_step
                # Print the average loss in a mini-batch.
                if batch_idx % 400 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Accuracy: {:.4f}, '
                          .format(phase.upper(), epoch + 1, num_epochs, batch_idx, int(batch_step_size),
                                  loss.item(), acc_step))
            # scheduler.step()
            # Print the average loss and accuracy in an epoch.
            print("epoch time minutes")
            print((time.time() - epoch_time) / 60)
            epoch_loss = running_loss / batch_step_size
            epoch_acc = running_accuracy.double() / len(data_loader[phase].dataset)
            print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc: {:.4f} \n'
                  .format(phase.upper(), epoch + 1, num_epochs, epoch_loss, epoch_acc))
            # Save loss and accuracy for train and test
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_accuracy.append(epoch_acc)
            else:
                test_loss.append(epoch_loss)
                test_accuracy.append(epoch_acc)

    print("all epochs time minutes")
    print((time.time() - start_time) / 60)

    ### Save the model
    torch.save(model, './model_hw2_128_final.pkl')
    torch.save(model.state_dict(), './model_hw2_128_64batch_final.pkl')
    # Define the current figure, all functions/commands will apply to the current figure
    ## first figure: Loss for train and test
    plt.figure(1)
    tr_loss = []
    index = range(1, num_epochs + 1)
    tr_loss.extend(
        plt.plot(index, train_loss, color='b', linestyle='--', marker='o', markerfacecolor='b', label='train loss'))
    tr_loss.extend(plt.plot(index, test_loss, color='green', linestyle='-.', marker='D', markerfacecolor='green',
                            label='test loss'))
    plt.setp(tr_loss, linewidth=2, markersize=5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.xticks(index, index)
    # Legend Box appearance
    plt.legend(shadow=True, fancybox=True)
    # Auto layout design function
    plt.tight_layout()

    ## second figure: Accuracy for train and test
    plt.figure(2)
    error = []
    index = range(1, num_epochs + 1)
    # Each plot function is a specific line (scenario)
    error.extend(
        plt.plot(index, train_accuracy, color='b', linestyle='--', marker='o', markerfacecolor='b',
                 label='train accuracy'))
    error.extend(plt.plot(index, test_accuracy, color='green', linestyle='-.', marker='D', markerfacecolor='green',
                          label='test accuracy'))
    plt.setp(tr_loss, linewidth=2, markersize=5)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy  - Epochs')
    plt.xticks(index, index)
    # Legend Box appearance
    plt.legend(shadow=True, fancybox=True)
    # Auto layout design function
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
