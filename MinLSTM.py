import torch.nn as nn
import torch
class MinLSTMCell(nn.Module):
    def __init__(self, units, input_shape):
        super(MinLSTMCell, self).__init__()
        self.units = units
        self.input_shape = input_shape
        
        # Initialize the linear layers for the forget gate, input gate, and hidden state transformation
        self.linear_f = nn.Linear(self.input_shape, self.units)
        self.linear_i = nn.Linear(self.input_shape, self.units)
        self.linear_h = nn.Linear(self.input_shape, self.units)

    def forward(self, pre_h, x_t):
        """
        pre_h: (batch_size, units) - previous hidden state (h_prev)
        x_t: (batch_size, input_size) - input at time step t
        """

        # Forget gate: f_t = sigmoid(W_f * x_t)
        f_t = torch.sigmoid(self.linear_f(x_t))  # (batch_size, units)

        # Input gate: i_t = sigmoid(W_i * x_t)
        i_t = torch.sigmoid(self.linear_i(x_t))  # (batch_size, units)

        # Hidden state: tilde_h_t = W_h * x_t
        tilde_h_t = self.linear_h(x_t)  # (batch_size, units)

        # Normalize the gates
        sum_f_i = f_t + i_t
        f_prime_t = f_t / sum_f_i  # (batch_size, units)
        i_prime_t = i_t / sum_f_i  # (batch_size, units)

        # New hidden state: h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t
        h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t  # (batch_size, units)

        return h_t  # (batch_size, units)
    
class MinRNN(nn.Module):
    def __init__(self, units, embedding_size, input_length):
        super(MinRNN, self).__init__()
        self.input_length = input_length
        self.units = units

        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = MinLSTMCell(units, embedding_size)
        self.classification_model = nn.Sequential(
            nn.Linear(units, 64),
            nn.Linear(64, 1), #for multi-class classification change number of classes here
            nn.Sigmoid()
        )

    def forward(self, sentence):
        """
        Args:
            sentence: (batch_size, input_length)

        output:
            (batch_size, 1)

        """
        batch_size = sentence.shape[0]

        # Initialize the hidden state, only the h needs to be initialized
        pre_h = torch.zeros(batch_size, self.units, device=sentence.device)

        # Pass the sentence through the embedding layer for the word vectors embeddings
        # embedded_sentence = self.embedding(sentence)

        sequence_length = sentence.shape[1]

        # Pass the entire sequence through the LSTM + hidden_state
        for i in range(sequence_length):
            word = sentence[:, i, :]  # (batch_size, embedding_size)
            pre_h = self.lstm(pre_h, word)  # Only update h (hidden state)

        return self.classification_model(pre_h)  # Pass the final hidden state into the classification network