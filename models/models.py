from nextlocationprediction.models.build import MODEL_REGISTRY

import torch
import torch.nn as nn
from torch.autograd import Variable

@MODEL_REGISTRY.register()
class LSTM(nn.Module):
    #Defining variables upon initialization of the network
    #     input:
    #           self.cfg.LSTM.embedding_dim        Dimension size of the embedding dimension
    #           cfg.LSTM.hidden_dim           Dimension size of the hidden dimension
    #           self.cfg.LSTM.n_loc_rank           Amount of rank location ID's 
    #           self.cfg.n_loc_type           Amount of type location ID's
    #           self.cfg.LSTM.n_layers             Number of LSTM layers
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Defining the weights of the weighted sum of the loss between exploration and location prediction
        self.weight_loss_w2v = Variable(torch.Tensor([1]).float(),requires_grad=True)
        self.weight_expl = Variable(torch.Tensor([1]).float(),requires_grad=True)
        self.weight_dist = nn.Parameter(torch.Tensor([1]).float(),requires_grad=True)
        
        
        # Defining the embedding layers
        self.embeddings_rank    = nn.Embedding(self.cfg.LSTM.n_loc_rank, self.cfg.LSTM.embedding_dim)        
        self.embeddings_type    = nn.Embedding(self.cfg.LSTM.n_loc_type, self.cfg.LSTM.embedding_dim)        
        self.embeddings_pep     = nn.Embedding(860, self.cfg.LSTM.embedding_dim)        
        self.embeddings_day     = nn.Embedding(8, self.cfg.LSTM.embedding_dim)      
        self.embeddings_global  = nn.Embedding(34279, self.cfg.LSTM.embedding_dim)

        # Defining the network layers
        self.lstm = nn.LSTM(self.cfg.LSTM.embedding_dim+3, self.cfg.LSTM.hidden_dim, self.cfg.LSTM.n_layers)
        self.decoder = nn.Linear(cfg.LSTM.hidden_dim, self.cfg.LSTM.embedding_dim)        
        self.decoder_expl = nn.Linear(cfg.LSTM.hidden_dim,2)

        self.hidden = self.init_hidden(self.cfg.TRAIN.BATCH_SIZE)
    
    # Function which takes in the data and transforms the data into the input embeddings,
    #   and return the summation of the embeddings. 
    #   Input:
    #         x         All of the data
    #   Return:
    #         Embedding of inputs
    def embeddings_input(self, x):
        return self.embeddings_rank(x[0].long())+self.embeddings_type(x[-1].long())+\
                self.embeddings_pep(x[5].long())+self.embeddings_day(x[3].long())+ self.embeddings_global(x[8].long())
    
    # Function which takes in the data and transforms the data into the output embeddings,
    #   and return the summation of the embeddings.
    #   Input:
    #         x         All of the data
    #   Return:
    #         Embedding of output
    def embeddings_out(self, x):   
        #import pdb; pdb.set_trace()
        return self.embeddings_rank(x[0].long()) + self.embeddings_type(x[-1].long())+ \
                    self.embeddings_pep(x[1].long()) + self.embeddings_global(x[2].long())
    
    #def embeddings_out(self, x):   
    #    return self.embeddings_type(x[-1].long()) + self.embeddings_pep(x[1].long())
    

    # Forward step.
    #   Input:
    #         x         All of the data
    #         hidden    Weights of hidden
    #   Return:
    #         logits         logits of next location prediction
    #         hidden         new hidden tensor 
    #         logits_expl    logits of exploration prediction
    def forward(self, x):
        # Embedding of input
        embeds = self.embeddings_input(x)   # (time, bs) -> (time, bs, self.cfg.LSTM.embedding_dim)  
        
        # Concatination of embedding and raw data that is not embedded
        embeds = torch.cat((embeds,x[[1,2,4]].permute(1,2,0).float()),2)
        
        # Actual network
        lstm_out, self.hidden = self.lstm(embeds,self.hidden)        # (time, bs, cfg.LSTM.hidden_dim) -> (time, bs, hidden_dim)
        logits = self.decoder(lstm_out)                    # (time, bs, cfg.LSTM.hidden_dim) -> (time, bs, n_loc)
        logits_expl = self.decoder_expl(lstm_out)
        self.hidden = self.repackage_hidden(self.hidden)
        return logits, self.hidden, logits_expl

    def init_hidden(self,bs):
        weight = next(self.parameters())
        return (weight.new_zeros(self.cfg.LSTM.n_layers, bs, self.cfg.LSTM.hidden_dim),
                weight.new_zeros(self.cfg.LSTM.n_layers, bs, self.cfg.LSTM.hidden_dim))
    
    # Return and save best network weights.
    def best_model_dict(self, save = False):
        print(f"Returning best model with accuracy: {self.best_accu}")
        if save:
            today = date.today()
            d_today = today.strftime("%b-%d-%Y")
            torch.save(self.best_model,f"Network/network_epoch{self.epochs}_{self.best_accu:4.3f}_{d_today}")
        return self.best_model
    
    # Repackage hidden dimension to avoid deletion when doing backpropagation
    def repackage_hidden(self,h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)