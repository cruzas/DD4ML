from src.base_trainer import *


class Trainer(BaseTrainer):

    def __init__(self, config, model, train_dataset):
        super().__init__(config, model, train_dataset)
