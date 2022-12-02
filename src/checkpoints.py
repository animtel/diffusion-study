import torch
from pathlib import Path
import uuid


def save_state_dict(module, path):
    torch.save(module.state_dict(), path)
    
    
def load_state_dict(module, path):
    state_dict = torch.load(path, map_location='cpu')
    res = module.load_state_dict(state_dict)
    print(res, path)


class CheckpointSaver:
    def __init__(self, 
        checkpoint_dir='./checkpoints', 
        every_n_epochs=1, 
        model_prefix='model',
        random_prefix=None,
        ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every_n_epochs = every_n_epochs

        if random_prefix is None:
            random_prefix = str(uuid.uuid4())[:6]

        self.model_prefix = model_prefix + random_prefix
        
    
    def on_epoch_end(self, epoch_i, model):
        out_file = self.checkpoint_dir/f'{self.model_prefix}_epoch_i{epoch_i}.ckpt'
        if epoch_i % self.every_n_epochs == 0:
            print(f'Saving at epoch {epoch_i} to {out_file}')
            self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
            save_state_dict(model, out_file)