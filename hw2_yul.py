"""
Main file
"""

from CreateDataSet import VQADataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

    ##starting with preprocessing of data, it should be made once, for creating txt and pickle files to be used by the model
    ##data files locations +outputdir:
    image_dir_train = "/datashare/train2014"
    image_dir_val = "/datashare/val2014"
    annotation_dir_train = "/datashare"
    question_dir_train = "/datashare"
    annotation_dir_val = "/datashare"
    question_dir_val = "/datashare"
    output_dir = "./processed"
    #preprocessed data
    proces_data = {}







    # Load dataset
    train_dataset = VQADataset(path=cfg['main']['paths']['train'])
    val_dataset = VQADataset(path=cfg['main']['paths']['validation'])

    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
                             num_workers=cfg['main']['num_workers'])

    for x, y,z in sample_loader:
        print(f'Input img: {x}')
        print(f'Label batch {y}')

        break
