import os
import argparse
import tfp.config.config as config
import torch
from torch.utils.data import DataLoader
from tfp.utils.data_loader import PoseDataset
from tfp.models.acGRU import acModel
from tqdm import tqdm, trange


parser = argparse.ArgumentParser(description="Training Information")
# parser.add_argument("--category",
#                     help="The catergory of data for which you have to train")
parser.add_argument("--seq_len",
                    help="Sequence length for which you have to train", default=100, type=int)
parser.add_argument("--overlap",
                    help="overlap for sequence length for which you have to train", default=0, type=int)
parser.add_argument(
    "--location", help="location of data set", default="data/salsa")
parser.add_argument("--split_ratio", help="Test/Train ratio",
                    default=0.2, type=float)
parser.add_argument("--num_joints", help="number of joints",
                    default=21, type=int)
parser.add_argument("--source_length", help="source_length", default=60, type=int)
parser.add_argument('--split', help='Either "train" or "test"', default='train')
parser.add_argument('--normalize', help='Use this flag to normalize data', default=1, type=int)

args = parser.parse_args()


if __name__ == "__main__":

    # Spliting into train and testdata
    # transformed data location
    posedataset = PoseDataset(args)
    train_loader = DataLoader(posedataset, batch_size=8, num_workers=4, shuffle=True, pin_memory=True)
    print()
    data_iter = iter(train_loader)

    model = acModel(256, num_layers=3, num_joints=21, residual_velocities=True)
    model = model.cuda()
    model.load_state_dict(torch.load('saved_models/state_dict_100000_iterations.pt'))

    ITERATIONS = 100000
    condition_length = 10
    ground_truth_length = 10
    save_freq = 2000
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt.load_state_dict(torch.load('saved_models/opt_state_dict_100000_iterations.pt'))
    avg_loss = 0.0
    print('Starting Training: \n')
    t = trange(ITERATIONS, unit='batch')
    for iteration in t:
        t.set_description(f'Step {iteration+1}')

        try:
            input_seq, target_seq = next(data_iter) 
        except StopIteration: 
            data_iter = iter(train_loader)
            input_seq, target_seq = next(data_iter)

        opt.zero_grad()
        output = model(input_seq.cuda(), condition_length, ground_truth_length)
        loss = model.calculate_loss(target_seq.cuda(), output)
        loss.backward()
        opt.step()
        avg_loss = avg_loss + 0.05*(loss.item()-avg_loss)
        t.set_postfix_str(f'Loss: {avg_loss:.5f}')
        t.update()

        if (iteration+1) % save_freq == 0:
            torch.save(model.state_dict(),
                       f'saved_models/state_dict_{iteration+1}_iterations.pt')
            torch.save(opt.state_dict(),
                       f'saved_models/opt_state_dict_{iteration+1}_iterations.pt')