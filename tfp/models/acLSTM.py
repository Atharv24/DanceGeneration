import numpy as np
import torch
from torch import nn
import torch.functional as F
from torch.autograd import Variable

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

        self.LSTM1 = nn.LSTMCell(self.input_size, self.hidden_size)
        self.LSTM2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.LSTM3 = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.projector = nn.Linear(self.hidden_size, self.input_size)

    def get_condition_list(self, seq_len, condition_length, ground_truth_length):
        gt_lst=np.ones((100, ground_truth_length))
        con_lst=np.zeros((100, condition_length))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]

    def init_hidden(self, batch_size):
        h1 = Variable(next(self.parameters()).data.new(batch_size, self.hidden_size), requires_grad=False).zero_()
        h2 = Variable(next(self.parameters()).data.new(batch_size, self.hidden_size), requires_grad=False).zero_()
        h3 = Variable(next(self.parameters()).data.new(batch_size, self.hidden_size), requires_grad=False).zero_()
        
        c1 = Variable(next(self.parameters()).data.new(batch_size, self.hidden_size), requires_grad=False).zero_()
        c2 = Variable(next(self.parameters()).data.new(batch_size, self.hidden_size), requires_grad=False).zero_()
        c3 = Variable(next(self.parameters()).data.new(batch_size, self.hidden_size), requires_grad=False).zero_()
        return (h1, h2, h3), (c1, c2, c3)

    def _forward_lstm(self, inp, h0, c0):
        
        h1, c1 = self.LSTM1(inp, (h0[0], c0[0]))
        h2, c2 = self.LSTM2(h1, (h0[1], c0[1]))
        h3, c3 = self.LSTM3(h2, (h0[2], c0[2]))

        return h3, (h1, h2, h3), (c1, c2, c3)


        


    def forward(self, real_seq, condition_length=5, ground_truth_length=5):
        """
        Args:
            real_seq: batch of dance pose sequences, shape=(batch_size,seq_length,num_joints*3)
        Returns:
            outputs: batch of predicted dance pose sequences, shape=(batch_size,seq_length,num_joints*3)
        """
        batch_size = real_seq.size(0)
        seq_len = real_seq.size(1)
        real_seq = real_seq.view(batch_size, seq_len, self.input_size)
        condition_list = self.get_condition_list(seq_len, condition_length, ground_truth_length)

        hidden_state, cell_state = self.init_hidden(batch_size)
        outputs = []
        prevOut = None
        # Iterate over decoder inputs
        for i in range(seq_len):
            if condition_list[i]==1:
                inp = real_seq[:, i, :]
            else:
                inp = prevOut
            h3, hidden_state, cell_state = self._forward_lstm(inp, hidden_state, cell_state)
            # Apply residual network to help in smooth transition between subsequent poses
            if self.residual_velocities:
                output = inp + self.projector(h3.view(batch_size, -1))
            else:
                output = self.projector(h3.view(batch_size, -1))
            prevOut = output
            outputs.append(output)

        outputs = torch.stack(outputs)
        return outputs.transpose(0, 1).view(batch_size, seq_len, self.num_joints, 3)

    def calculate_loss(self, ground_truth, prediction):
        loss_function = nn.MSELoss()
        loss = loss_function(ground_truth, prediction)
        return loss
    
    def generate(self, warmup_seq, n_frames):
        with torch.no_grad():
            outputs = []
            batch_size = warmup_seq.size(0)
            seq_len = warmup_seq.size(1)
            hidden, cell_state = self.init_hidden(batch_size)
            warmup_seq = warmup_seq.view(batch_size, seq_len, self.input_size)
            for frame in warmup_seq.transpose(0, 1):
                h3, hidden, cell_state = self._forward_lstm(frame, hidden, cell_state)
                if self.residual_velocities:
                    last_warmup_input = warmup_seq[:, -1, :]
                    output = last_warmup_input + self.projector(h3.view(batch_size, -1))
                else:
                    output = self.projector(h3.view(batch_size, -1))
                outputs.append(output)
            output = outputs[-1]
            for i in range(n_frames-1):
                h3, hidden, cell_state = self._forward_lstm(output, hidden, cell_state)
                if self.residual_velocities:
                    output = output + self.projector(h3.view(batch_size, -1))
                else:
                    output = self.projector(h3.view(batch_size, -1))
                outputs.append(output)
            outputs = torch.stack(outputs)
            
            return outputs.transpose(0, 1).view(batch_size, -1, self.num_joints, 3)