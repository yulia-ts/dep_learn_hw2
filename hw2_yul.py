"""
Main file
"""

from CreateDataSet import VQADataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

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
