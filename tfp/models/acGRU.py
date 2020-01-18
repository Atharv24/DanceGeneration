import numpy as np
import torch
from torch import nn

class acModel(nn.Module):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self,
                 hidden_size,
                 num_layers=1,
                 num_joints=21,
                 residual_velocities=False):
        """Create the model.
        Args:
          hidden_size: number of units in the rnn.
          num_layers: number of rnns to stack.
          residual_velocities: whether to use a residual connection that models velocities.
        """
        super(acModel, self).__init__()

        self.input_size = num_joints*3
        self.num_joints = num_joints
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.residual_velocities = residual_velocities
        # === Create the RNN that will keep the state ===

        self.GRU = nn.GRU(self.input_size, self.hidden_size, num_layers=self.num_layers)
        self.projector1 = nn.Linear(self.hidden_size*self.num_layers, self.hidden_size*self.num_layers)
        self.projector2 = nn.Linear(self.hidden_size*self.num_layers, self.hidden_size*self.num_layers)
        self.projector3 = nn.Linear(self.hidden_size*self.num_layers, self.input_size)

    def get_condition_list(self, seq_len, condition_length, ground_truth_length):
        gt_lst=np.ones((100, ground_truth_length))
        con_lst=np.zeros((100, condition_length))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]

    def forward(self, real_seq, condition_length, ground_truth_length):
        """
        Args:
            real_seq: batch of dance pose sequences, shape=(batch_size,seq_length,num_joints*3)
        Returns:
            outputs: batch of predicted dance pose sequences, shape=(batch_size,seq_length,num_joints*3)
        """
        # First calculate the encoder hidden state
        batch_size = real_seq.size(0)
        seq_len = real_seq.size(1)
        real_seq = real_seq.view(batch_size, seq_len, self.input_size)
        condition_list = self.get_condition_list(seq_len, condition_length, ground_truth_length)

        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        output = torch.zeros(batch_size, self.input_size).cuda()
        outputs = []
        # Iterate over decoder inputs
        for i in range(seq_len):
            if condition_list[i]==1:
                inp = real_seq[:, i, :]
            else:
                inp = output
            _, hidden = self.GRU(inp.unsqueeze(0), hidden)
            # Apply residual network to help in smooth transition between subsequent poses
            if self.residual_velocities:
                linear_out = self.projector2(self.projector1(hidden.view(batch_size, -1)))
                output = inp + self.projector3(linear_out)
            else:
                linear_out = self.projector2(self.projector1(hidden.view(batch_size, -1))) 
                output = self.projector3(linear_out)
            outputs.append(output)

        outputs = torch.stack(outputs)
        return outputs.transpose(0, 1).view(batch_size, seq_len, self.num_joints, 3)

    def calculate_loss(self, ground_truth, prediction):
        loss_function = nn.MSELoss()
        loss = loss_function(ground_truth, prediction)
        return loss
    
    def generate(self, warmup_seq, n_frames):
        with torch.no_grad():
            batch_size = warmup_seq.size(0)
            warmup_seq = warmup_seq.view(batch_size, -1, self.input_size)
            _, hidden = self.GRU(warmup_seq.transpose(0, 1))
            if self.residual_velocities:
                last_warmup_input = warmup_seq[:, -1, :]
                linear_out = self.projector2(self.projector1(hidden.view(batch_size, -1)))
                output = last_warmup_input + self.projector3(linear_out)
            else:
                linear_out = self.projector2(self.projector1(hidden.view(batch_size, -1)))
                output = self.projector3(linear_out)
            outputs = []
            outputs.append(output)
            for i in range(n_frames-1):
                _, hidden = self.GRU(output.unsqueeze(0), hidden)
                if self.residual_velocities:
                    linear_out = self.projector2(self.projector1(hidden.view(batch_size, -1)))
                    output = output + self.projector3(linear_out)
                else:
                    linear_out = self.projector2(self.projector1(hidden.view(batch_size, -1)))
                    output = self.projector3(linear_out)
                outputs.append(output)
            outputs = torch.stack(outputs)
            return outputs.transpose(0, 1).view(batch_size, -1, self.num_joints, 3)