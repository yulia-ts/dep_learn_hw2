"""
Main file
"""
import os
import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#from CreateDataSet import VQADataset
#from torch.utils.data import DataLoader




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

    ##first we resize images from the input directory and saving it in the output one
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
    }


    image_size = [256,256]
    ##train images
    images_train = os.listdir(image_dir_train)
    num_train_images = len(images_train)
    print("number of train images")
    print(num_train_images)

    for iimage, image in enumerate(images_train):
        file_n_images = os.path.join(image_dir_train, image)
        with Image.open(file_n_images) as im:
            im = im.resize(image_size, Image.ANTIALIAS)
            im.save(os.path.join(output_dir_train , image), im.format)
        if (iimage+1) % 1000 == 0:
            print("[{}/{}] Resized and saved the images '{}'.".format(iimage+1, num_train_images,output_dir_train))

    ##make vocabulary

    #preprocessing answers:


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

