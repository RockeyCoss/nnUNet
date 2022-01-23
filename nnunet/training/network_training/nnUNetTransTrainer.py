from .nnUNetTrainerV2 import nnUNetTrainerV2

class nnUNetTransTrainer(nnUNetTrainerV2):
    def __init__(self,
                 plans_file,
                 fold,
                 output_folder=None,
                 dataset_directory=None,
                 batch_dice=True,
                 stage=None,
                 unpack_data=True,
                 deterministic=True,
                 fp16=False):
        super(nnUNetTransTrainer, self).__init__(plans_file,
                                                 fold,
                                                 output_folder,
                                                 dataset_directory,
                                                 batch_dice,
                                                 stage,
                                                 unpack_data,
                                                 deterministic,
                                                 fp16)

    def initialize_network(self):
        # TODO: initialize network
        pass
        
