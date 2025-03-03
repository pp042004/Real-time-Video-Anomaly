import torch.nn as nn
from torchdiffeq import odeint as odeint
import torch 
from Transformer import TransformerNet
from MinLSTM import MinRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ODE function and model
class ODEFunc(nn.Module):
    def __init__(self, latent_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, t, y):
        return self.net(y)

class ODEModel(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, nc,max_len, classifier_name):
        super(ODEModel, self).__init__()
        self.ode_func = ODEFunc(latent_dim)
        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)
        self.classifier_name = classifier_name
        self.Feature_Extraction = False #default value is false
        if classifier_name == 'LSTM':
            self.classifier = nn.LSTM(latent_dim, int(latent_dim/2), num_layers=1)
            self.fc2 = nn.Linear(int(latent_dim/2), nc)
        elif classifier_name == 'BERT':
            self.classifier = TransformerNet(latent_dim, latent_dim, 8, 2, max_len, nc, 0.5).to(device) # for multiclass classification
        # self.classifier = MinRNN(units=128, embedding_size=latent_dim, input_length=max_len)
        # self.fc2 = nn.Linear(int(latent_dim/2), 1) 
        
    
    def forward(self, initial_state, time_steps):
        zs = odeint(self.ode_func, initial_state, time_steps)
        initial_state = initial_state.unsqueeze(0)
        input_lstm = torch.cat((initial_state, zs)).permute(1, 0, 2)
        if self.classifier_name == 'LSTM':
            lstm_out, _ = self.classifier(input_lstm)
            if self.Feature_Extraction:
                return lstm_out[:, -1, :], lstm_out[:, -1, :]
            lstm_out = self.fc2(lstm_out[:, -1, :]) #This line of code is only needed when your using RNN
        if self.classifier_name == 'BERT':
            lstm_out= self.classifier(input_lstm) 
        # lstm_out = self.fc2(lstm_out[:, -1, :])
        hs = self.l2h(zs)
        xs = self.h2o(hs)
        return xs, lstm_out

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rnn = nn.GRU(input_dim, hidden_dim,batch_first=True)
        # self.rnn = TransformerNet(input_dim, latent_dim, 2, 2, t, hidden_dim, 0.5).to(device)
        self.hid2lat = nn.Linear(hidden_dim, 2*latent_dim)

    def forward(self, x, t):
        _, h0 = self.rnn(x[:, :t, :])
        z0 = self.hid2lat(h0[0])
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]
        return z0_mean, z0_log_var
    

class ODEVAE(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, nc, max_len, classifier_name):
        super(ODEVAE, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = RNNEncoder(output_dim, hidden_dim, latent_dim)
        self.decoder = ODEModel(output_dim, hidden_dim, latent_dim, nc, max_len, classifier_name)
    
    def forward(self, x, t, MAP=False):
        z_mean, z_log_var = self.encoder(x, t)
        if MAP:
            z = z_mean
        else:
            z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var)
        t = torch.linspace(0, 1, steps=x.shape[1]-t).to(device)
        x_p, lstm_out = self.decoder(z, t)
        return x_p, z, z_mean, z_log_var, lstm_out
