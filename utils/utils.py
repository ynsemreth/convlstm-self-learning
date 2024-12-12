from models.conv_lstm import ConvLSTM_Model
import torch 


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

    
def get_model(args):
    if args.model == 'convlstm':
        return ConvLSTM_Model(args)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
def load_checkpoint(model, args, path):
    checkpoint = torch.load(path, map_location=torch.device("mps"))
    
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: Missing 'model_state_dict' in {path}")

    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
    start_epoch = checkpoint.get('epoch', 0)
    lr = checkpoint.get('lr', args.lr)

    return start_epoch, lr, optimizer_state_dict


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr': optimizer.param_groups[0]['lr'],
    }, path)
    if "best" in path:
        print(f"Saved checkpoint at epoch {epoch}")