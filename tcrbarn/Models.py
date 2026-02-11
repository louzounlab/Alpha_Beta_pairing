import torch
import torch.nn as nn


class EncoderLstm(nn.Module):
    """
    Encoder LSTM module for sequence-to-sequence models.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_size (int): Size of the embedding vectors.
        hidden_size (int): Size of the hidden state in the LSTM.
        latent_size (int): Size of the latent vector.
        dropout_prob (float, optional): Dropout probability. Default is 0.
        layer_norm (bool, optional): Whether to apply layer normalization. Default is False.
        num_layers (int, optional): Number of LSTM layers. Default is 1.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, dropout_prob=0, layer_norm=False,
                 num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size * 2) if layer_norm else nn.Identity()
        self.bidirectional = bidirectional
        direction = 2 if self.bidirectional else 1
        self.fc_mean = nn.Linear(hidden_size * 2 * direction, latent_size)
        self.fc_logvar = nn.Linear(hidden_size * 2 * direction, latent_size)
        # self.fc_latent = nn.Linear(hidden_size * 2 * direction, latent_size) # For AE

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            outputs (Tensor): LSTM outputs of shape (batch_size, sequence_length, hidden_size).
            mean (Tensor): Mean of the latent distribution of shape (batch_size, latent_size * 2).
            log_var (Tensor): Log variance of the latent distribution of shape (batch_size, latent_size * 2).
        """
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden, cell = hidden.contiguous(), cell.contiguous()
        if self.bidirectional:
            forward_last_hidden = hidden[-2]
            backward_last_hidden = hidden[-1]
            hidden = torch.cat((forward_last_hidden, backward_last_hidden), dim=1).unsqueeze(0)
            forward_last_cell = cell[-2]
            backward_last_cell = cell[-1]
            cell = torch.cat((forward_last_cell, backward_last_cell), dim=1).unsqueeze(0)
        concatenated_states = torch.cat((hidden, cell), dim=-1)
        concatenated_states = self.layer_norm(concatenated_states)
        concatenated_states = self.dropout(concatenated_states)
        mean = self.fc_mean(concatenated_states)
        log_var = self.fc_logvar(concatenated_states)
        return outputs, mean, log_var
        # latent = self.fc_latent(concatenated_states)
        # return outputs, latent  # For AE


class DecoderLstm(nn.Module):
    """
    Decoder LSTM module for sequence-to-sequence models.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_size (int): Size of the embedding vectors.
        hidden_size (int): Size of the hidden state in the LSTM.
        latent_size (int): Size of the latent vector.
        dropout_prob (float, optional): Dropout probability. Default is 0.2.
        layer_norm (bool, optional): Whether to apply layer normalization. Default is False.
        num_layers (int, optional): Number of LSTM layers. Default is 1.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, dropout_prob=0.2, layer_norm=False,
                 num_layers=1, ALSTM=False, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.fc_hidden = nn.Linear(latent_size, hidden_size)
        self.fc_cell   = nn.Linear(latent_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
        self.bidirectional = bidirectional
        direction = 2 if self.bidirectional else 1
        self.ALSTM = ALSTM
        if ALSTM:
            self.att_gate = nn.Linear(embed_size + latent_size * direction, embed_size)


    def init_hidden(self, z):
        if z.shape[0] < self.num_layers:
            h0 = torch.tanh(self.fc_hidden(z).repeat(self.num_layers, 1, 1))
            c0 = torch.tanh(self.fc_cell(z).repeat(self.num_layers, 1, 1))
        else:
            h0 = torch.tanh(self.fc_hidden(z))
            c0 = torch.tanh(self.fc_cell(z))
        return h0, c0

    def forward(self, x, hidden_cell):
        """
        Forward pass for the decoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length).
            hidden_cell (Tensor): Tuple of hidden and cell states of shape (batch_size, hidden_size).

        Returns:
            out (Tensor): Output tensor of shape (batch_size, sequence_length, vocab_size).
            hidden_cell (Tensor): Tuple of hidden and cell states of shape (batch_size, hidden_size).
        """
        out = self.embedding(x)
        (hidden, cell) = hidden_cell
        hidden, cell = hidden.contiguous(), cell.contiguous()
        if self.ALSTM:
            # Compute attention weights based on hidden state
            # Expand hidden to match sequence dimension
            if self.bidirectional:
                h_forward = hidden[-2]
                h_backward = hidden[-1]
                last_hidden = torch.cat([h_forward, h_backward], dim=-1)
            else:
                last_hidden = hidden[-1]
            hidden_expanded = last_hidden.unsqueeze(1).expand(-1, out.size(1), -1)
            combined = torch.cat([out, hidden_expanded], dim=-1)
            gate = torch.sigmoid(self.att_gate(combined))
            out = out * gate  # weighted input
        out, (hidden, cell) = self.lstm(out, (hidden, cell))
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out).reshape(out.size(0), -1)
        return out, (hidden, cell)


class FFNN(nn.Module):
    """
    Feedforward Neural Network (FFNN) module.

    Args:
        input_dim (int): Dimension of the input features.
        dropout_prob_cl (float, optional): Dropout probability. Default is 0.
        norm_cl (bool, optional): Whether to apply batch normalization. Default is False.
    """
    def __init__(self, input_dim, dropout_prob_cl=0, norm_cl=False, h1 = 256, h2 = 64):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.dropout = nn.Dropout(dropout_prob_cl)
        self.norm1 = nn.BatchNorm1d(h1) if norm_cl else nn.Identity()
        self.norm2 = nn.BatchNorm1d(h2) if norm_cl else nn.Identity()

    def forward(self, x):
        x = x[-1]
        x = torch.relu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class LogisticRegression(nn.Module):
    """
    Logistic Regression model..

    Args:
        input_dim (int): Dimension of the input features.
    """
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x[-1]
        x = self.fc(x)
        return x