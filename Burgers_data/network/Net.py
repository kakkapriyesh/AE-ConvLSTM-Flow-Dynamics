from torch import nn
import torch.nn.functional as F
import torch
from network.subnet import make_layers

class ED(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input,hidden_state_prev):
        if hidden_state_prev==None:
            hidden_state_prev=None
        else:
            X=(hidden_state_prev)
            
        
            hidden_state_prev=X#tuple(map(torch.stack, zip(*X)))
        state = self.encoder(input,hidden_state_prev)
        output = self.decoder(state)
        return output,state
class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)   #subnets are basically normal CNN blocks
        self.blocks = len(subnets)  #blocks given length of subnets

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params)) #set attributes to make layers, below is get attributes
            setattr(self, 'rnn' + str(index), rnn)   #making layers one by one simulatenously

    def forward_by_stage(self, inputs,hidden_states_Prev, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)   #subnets is made in subnet file
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))

        outputs_stage, state_stage = rnn(inputs, hidden_states_Prev)  #rnn returns hx(activation),cx(state) and output
        return outputs_stage, state_stage               #conv_lstm is alread used in encoder module

    def forward(self, inputs,hidden_states_Prev):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        
        hidden_states = []
        if hidden_states_Prev==None:
            for i in range(1, self.blocks + 1):  #self.blocks is length of subnets
                inputs, state_stage = self.forward_by_stage(
                    inputs,hidden_states_Prev, getattr(self, 'stage' + str(i)),
                    getattr(self, 'rnn' + str(i)))
                hidden_states.append(state_stage)
        else:
            for i in range(1, self.blocks + 1):  #self.blocks is length of subnets
                inputs, state_stage = self.forward_by_stage(
                    inputs,hidden_states_Prev[i-1], getattr(self, 'stage' + str(i)),
                    getattr(self, 'rnn' + str(i)))
                hidden_states.append(state_stage)

        return tuple(hidden_states)  #only hidden state is fed to decoder


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))  # make layers goes to utils and does the work

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=10)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):

        inputs = self.forward_by_stage(None, hidden_states[-1],  # no input (just state) for stage 3 or first decoder
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # S,B,1,64,64 to B,S,1,64,64 #line 1 of fwd by stage
        return inputs