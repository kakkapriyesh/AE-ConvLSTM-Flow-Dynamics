from collections import OrderedDict
from network.rnn_10 import CLSTM_cell


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=32),  #When called goes to ConvRNN dir
        CLSTM_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64),
        CLSTM_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=64)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [32, 16, 3, 1, 1],
            'conv4_none_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32),
    ]
]

