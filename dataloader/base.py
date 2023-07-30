import random


class Dataloader:
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.label_to_idx = dataset['label_to_index']
        