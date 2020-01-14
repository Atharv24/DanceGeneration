import os
import argparse
import numpy as np
import tfp.config.config as config
import torch
from torch.utils.data import DataLoader
from tfp.utils.visualization import Visualizer
from tfp.utils.data_loader import PoseDataset
from tfp.models.seq2seq import Seq2SeqModel

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
                        default=0.2, type=int)
    parser.add_argument("--num_joints", help="number of joints",
                        default=21, type=int)
    parser.add_argument("--source_length", help="source_length", default=60, type=int)
    parser.add_argument('--split', help='Either "train" or "test"', default='test')

    args = parser.parse_args()

    model = Seq2SeqModel(None, 128, num_layers=3, num_joints=21, residual_velocities=True, dropout=0.3, teacher_ratio=1.0)
    model.eval()
    model.load_state_dict(torch.load('saved_models/state_dict_60_epochs.pt'))

    dataset = PoseDataset(args)
    test_data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    n_frames = 200
    ground_truth = None
    for enc, dec, tgt in test_data_loader:
        output = model(enc, dec)
        ground_truth = tgt.numpy().reshape(-1, 39, 21, 3)
        break

    output = output.detach().numpy().reshape(-1, 39, 21, 3)
    visualizer_gen = Visualizer('Generated')
    visualizer_truth = Visualizer('Ground Truth')
    visualizer_gen.generate_and_save_avi(output[0], 60)
    visualizer_truth.generate_and_save_avi(ground_truth[0], 60)


