"""
Main file
"""
from __future__ import print_function
import os
from PIL import Image
import numpy as np
from glob import glob
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import argparse
import sys
import json
import numpy as np
import re
import pickle
#from CreateDataSet import VQADataset

##preprocessing the answers given functions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from dataset import Dictionary
#import utils


contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']


def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1


def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer


def filter_answers(answers_dset, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence = {}
    for ans_entry in answers_dset:
        gtruth = ans_entry['multiple_choice_answer']
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
    print("number of total answers")
    print(len(occurence))
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence


def create_ans2label(occurence, name, cache_root):
    """Note that this will also create label2ans.pkl at the same time

    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1
    #os.mkdir(cache_root)
    cache_file = os.path.join(cache_root, name+'_ans2label.pkl')

    pickle.dump(ans2label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_root, name+'_label2ans.pkl')
    pickle.dump(label2ans, open(cache_file, 'wb'))
    return ans2label


def compute_target(answers_dset, ans2label, name, cache_root):
    """Augment answers_dset with soft score as label

    ***answers_dset should be preprocessed***

    Write result into a cache file
    """
    target = []
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer['answer']
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        labels = []
        scores = []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            score = get_score(answer_count[answer])
            scores.append(score)

        label_counts = {}
        for k, v in answer_count.items():
            if k in ans2label:
                label_counts[ans2label[k]] = v

        target.append({
            'question_id': ans_entry['question_id'],
            'question_type': ans_entry['question_type'],
            'image_id': ans_entry['image_id'],
            'label_counts': label_counts,
            'labels': labels,
            'scores': scores
        })

    #os.mkdir(cache_root)
    cache_file = os.path.join(cache_root, name+'_target.pkl')
    with open(cache_file, 'wb') as f:
      pickle.dump(target, f)
    return target


def get_answer(qid, answers):
    for ans in answers:
        if ans['question_id'] == qid:
            return ans


def get_question(qid, questions):
    for question in questions:
        if question['question_id'] == qid:
            return question


def load_v2():
    train_answer_file = '/datashare/v2_mscoco_train2014_annotations.json'
    with open(train_answer_file) as f:
        train_answers = json.load(f)['annotations']

    val_answer_file = '/datashare/v2_mscoco_val2014_annotations.json'
    with open(val_answer_file) as f:
        val_answers = json.load(f)['annotations']
    occurence = filter_answers(train_answers, 9)
    ans2label = create_ans2label(occurence, 'trainval', "cache")
    compute_target(train_answers, ans2label, 'train', "cache")
    compute_target(val_answers, ans2label, 'val', "cache")



if __name__ == '__main__':

    ##starting with preprocessing of data, it should be made once, for creating txt and pickle files to be used by the model
    ##data files locations +outputdir:
    image_dir_train = "/datashare/train2014"
    image_dir_val = "/datashare/val2014"
    annotation_file_train = "/datashare/v2_mscoco_train2014_annotations.json"
    question_file_train = "/datashare/v2_OpenEnded_mscoco_train2014_questions.json"
    annotation_file_val = "/datashare/v2_OpenEnded_mscoco_val2014_questions.json"
    question_file_val = "/datashare/v2_mscoco_val2014_annotations.json"
    output_dir_train = "./processed/train"
    output_dir_val = "./processed/val"
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }"""

    ##first we resize images from the input directory and saving it in the output one
    """
    image_size = [256,256]
    ##train images
    images_train = os.listdir(image_dir_train)
    num_train_images = len(images_train)
    print("number of train images")
    print(num_train_images)
    
    # with h5py.File('data.h5', 'w') as hf:
    #     for iimage, image in enumerate(images_train):
    #         file_n_images = os.path.join(image_dir_train, image)
    #         im = cv2.imread(file_n_images)
    #         im = cv2.resize(image, image_size, interpolation = cv2.INTER_AREA)
    #         cv2.imwrite(os.path.join(output_dir_train , image),im)
    #         if (iimage+1) % 1000 == 0:
    #            print("[{}/{}] Resized and saved the images '{}'.".format(iimage+1, num_train_images,output_dir_train))
    ################### RUN it only once for saving resized RGB images
    for iimage, image in enumerate(images_train):
        file_n_images = os.path.join(image_dir_train, image)
        with Image.open(file_n_images) as im:
            #if im.mode != 'RGB':
                #print(im.size)
            im = im.resize(image_size,Image.ANTIALIAS)
            if im.mode != 'RGB':
                #print(im.size)
                #print(im.mode)
                im = im.convert(mode="RGB")
                #print(im.mode)
                #print(im.size)
            #print(output_dir_train)
            #print(image)
            im.save(os.path.join(output_dir_train, image), im.format)
        if (iimage+1) % 1000 == 0:
            print("[{}/{}] Resized and saved the images '{}'.".format(iimage+1, num_train_images,output_dir_train))
    ##VAL images
    images_val = os.listdir(image_dir_val)
    num_val_images = len(images_val)
    print("number of val images")
    print(num_val_images)
    for iimage, image in enumerate(images_val):
        file_n_images = os.path.join(image_dir_val, image)
        with Image.open(file_n_images) as im:
            im = im.resize(image_size, Image.ANTIALIAS)
            if im.mode != 'RGB':
                im = im.convert(mode="RGB")
            im.save(os.path.join(output_dir_val, image), im.format)
        if (iimage + 1) % 1000 == 0:
            print("[{}/{}] VAL: Resized and saved the images '{}'.".format(iimage + 1, num_val_images, output_dir_val))"""


    ##make vocabulary

    #preprocessing answers(using given script):
    load_v2()

    #preprocessing questions:

    #preprocessing val set



    ##train

    ##validation





    # Load dataset
    train_dataset = VQADataset(path=cfg['main']['paths']['train'])
    val_dataset = VQADataset(path=cfg['main']['paths']['validation'])

    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
                             num_workers=cfg['main']['num_workers'])

