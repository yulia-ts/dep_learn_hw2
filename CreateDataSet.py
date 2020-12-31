import torch
from torch.utils.data import Dataset, DataLoader
import os


""" The purpose of this file:
 class preparing a vocabilary and class preparing VQA dataset"""
### Hyper Parameters
#int_size = 784
#hid_size = 128
num_epochs = 2
batch_size = 50
learning_rate = 0.1
n_answers = 1000
max_questions_len = 25 #30
#num_classes = 10


data_path = "/datashare"
save_path = "./processed"

class VQA_Vocab:
    def __init__(self, vocab_file):

        self.vocab = self.load_vocab(vocab_file)
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def load_vocab(self, vocab_file):

        with open(vocab_file) as f:
            vocab = [v.strip() for v in f]

        return vocab

    def word2idx(self, vocab):

        if vocab in self.vocab2idx:
            return self.vocab2idx[vocab]
        else:
            return self.vocab2idx['<unk>']

    def idx2word(self, idx):

        return self.vocab[idx]


class VQADataset(Dataset):
    def __init__(self, data_path, input_f_type, max_q_len = max_questions_len, n_features: int = 1024, n_samples= n_answers):
        # Set variables
        #self.img_file_name='/datashare/%s/COCO_%s_%012d.jpg'%(input_f_type,input_f_type,image_id)
        #path to images
        self.img_input = torch.load(os.path.join(data_path,input_f_type))
        #max questions len
        self.max_questions_l = max_q_len
        # Load features
        self.features = self._get_features()

        # Create list of entries
        self.entries = self._get_entries()

        --
        self.n_features = n_features
        self.n_samples = n_samples
        self.entries = self._create_entries()

    def _create_entries(self):
        """
                This function create a list of all the entries. We will use it later in __getitem__
                :return: list of samples
                """
        entries = []

        for idx, item in self.features.items():
            entries.append(self._get_entry(item))



    def __getitem__(self, index):
        entry = self.entries[index]

        return entry['img'], entry['question'], entry['answer']

    def __len__(self):
        return len(self.entries)

    def _get_features(self) -> Any:
        """
        Load all features into a structure (not necessarily dictionary). Think if you need/can load all the features
        into the memory.
        :return:
        :rtype:
        """
        with open(self.path, "rb") as features_file:
            features = pickle.load(features_file)

        return features

    def _get_entry(item: Dict) -> Dict:
        """
        :item: item from the data. In this example, {'input': Tensor, 'y': int}
        """
        x = item['input']
        y = torch.Tensor([1, 0]) if item['label'] else torch.Tensor([0, 1])

        return {'x': x, 'y': y}


