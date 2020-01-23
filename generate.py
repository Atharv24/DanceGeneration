import os
import argparse
import numpy as np
import tfp.config.config as config
import torch
from torch.utils.data import DataLoader
from tfp.utils.visualization import Visualizer
from tfp.utils.data_loader import PoseDataset
from tfp.models.acGRU import acModel

def add_root(rel_root_joints, root=0):
        """
        Assumed shape: [number_frames, number_joints, 3]
        """
        abs_joints_root_xyz = np.zeros(rel_root_joints.shape)
        root_locations = rel_root_joints[:, root, :].copy()
        abs_joints_root_xyz = rel_root_joints + rel_root_joints[:, root:root+1, :]
        abs_joints_root_xyz[:, root, :] = root_locations

        return abs_joints_root_xyz

if __name__ == "__main__":
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
    parser.add_argument('--split', help='Either "train" or "test"', default='test')
    parser.add_argument('--normalize', help='Use to flag to normalize data', default=1, type=int)
    parser.add_argument('--checkpoint', help='Checkpoint file to load', default=1000)

    args = parser.parse_args()
    visualizer_gen = Visualizer('Generated')
    visualizer_truth = Visualizer('Ground Truth')

    model = acModel(256, num_layers=3, num_joints=21, residual_velocities=True).cuda()
    model.eval()
    model.load_state_dict(torch.load(f'saved_models/state_dict_{args.checkpoint}_iterations.pt'))

    dataset = PoseDataset(args)
    test_data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    n_frames = 1000
    warmup_length = 99
    ground_truth = None
    for inp, tgt in test_data_loader:
        inp = inp.cuda()
        output = model.generate(inp, n_frames)
        ground_truth = tgt
        break

    output = output.detach().cpu().numpy()
    output = add_root(output[0])
    ground_truth = add_root(ground_truth[0].numpy())
    visualizer_gen.generate_and_save_avi(output, 30)
    visualizer_truth.generate_and_save_avi(ground_truth, 30)


