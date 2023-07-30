from dataset import dataset_factory
from .bert import BertDataloader

def dataloader_factory(args):
    dataset = dataset_factory()
    dataloader = BertDataloader(args, dataset)
    val = dataloader.get_dataloaders()
    return val
