import torch
import torch.nn as nn


class TransducerGRU(nn.Module):
    def __init__(self, image_channels, image_features, gru_layers, hidden_size, num_classes, bidirectional=True):
        super(TransducerGRU, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = gru_layers
        self.num_classes = num_classes
        self.gru_encoder_h1 = nn.GRU(image_features,
                                     hidden_size,
                                     num_layers=self.num_layers,
                                     bidirectional=bidirectional,
                                     batch_first=True)
        self.gru_decoder_h1 = nn.GRU(2 * hidden_size,
                                     hidden_size,
                                     num_layers=self.num_layers,
                                     bidirectional=bidirectional,
                                     batch_first=True)
        self.gru_encoder_h2 = nn.GRU(image_features,
                                     hidden_size,
                                     num_layers=self.num_layers,
                                     bidirectional=bidirectional,
                                     batch_first=True)
        self.gru_decoder_h2 = nn.GRU(2 * hidden_size,
                                     hidden_size,
                                     num_layers=self.num_layers,
                                     bidirectional=bidirectional,
                                     batch_first=True)

        self.dense1_h1 = nn.Linear(self.hidden_size * 2, self.num_classes)
        self.dense1_h2 = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x_h1, x_h2, hidden_h1, hidden_h2):
        hidden_h1 = hidden_h1.transpose(0, 1).contiguous()
        hidden_h2 = hidden_h2.transpose(0, 1).contiguous()

        self.gru_encoder_h1.flatten_parameters()
        self.gru_encoder_h2.flatten_parameters()
        x_out_h1, hidden_out_h1 = self.gru_encoder_h1(x_h1, hidden_h1)
        x_out_h2, hidden_out_h2 = self.gru_encoder_h2(x_h2, hidden_h2)

        self.gru_decoder_h1.flatten_parameters()
        self.gru_decoder_h2.flatten_parameters()
        x_out_h1, hidden_final_h1 = self.gru_decoder_h1(x_out_h1, hidden_out_h1)
        x_out_h2, hidden_final_h2 = self.gru_decoder_h2(x_out_h2, hidden_out_h2)

        x_out_h1 = self.dense1_h1(x_out_h1)
        x_out_h2 = self.dense1_h2(x_out_h2)

        hidden_final_h1 = hidden_final_h1.transpose(0, 1).contiguous()
        hidden_final_h2 = hidden_final_h2.transpose(0, 1).contiguous()
        return x_out_h1, x_out_h2, hidden_final_h1, hidden_final_h2

    def init_hidden(self, batch_size, num_layers, bidirectional=True):
        num_directions = 1
        if bidirectional:
            num_directions = 2

        return torch.zeros(batch_size, num_directions * num_layers, self.hidden_size)