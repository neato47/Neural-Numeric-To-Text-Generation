import itertools
from collections import OrderedDict
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn
try:
    from torchtext.data import Batch
except ImportError:
    from torchtext.legacy.data import Batch
    
from torchtext.vocab import Vocab

from reporter.util.config import Config
from reporter.util.constant import (
    GENERATION_LIMIT,
    N_LONG_TERM,
    N_SHORT_TERM,
    TIMESLOT_SIZE,
    Phase,
    SeqType,
    SpecialToken
)
from reporter.util.conversion import stringify_ric_seqtype

from proto_lib import getTruthValue

class Attention(nn.Module):
    '''This implementation is based on `Luong et al. (2015) <https://arxiv.org/abs/1508.04025>`_.
    '''

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, h_t: Tensor, h_s: Tensor) -> Tensor:
        return self.align(h_t, h_s)

    def align(self, h_t: Tensor, h_s: Tensor) -> Tensor:
        r'''
        .. math:
            a_{ij} =
                \frac{%
                    \exp\left(
                        \operatorname{score}\left(
                            \boldsymbol{h}^\text{target}_j, \boldsymbol{h}^\text{source}_i
                        \right)
                    \right)
                }{%
                    \sum_{\iota = 1}^I
                        \exp\left(
                            \operatorname{score}\left(
                                \boldsymbol{h}^\text{target}_j, \boldsymbol{h}^\text{source}_\iota
                            \right)
                        \right)
                }
        '''
        return nn.functional.softmax(self.score(h_t, h_s), dim=1)

    def score(self, h_t: Tensor, h_s: Tensor) -> Tensor:
        raise NotImplementedError


class GeneralAttention(Attention):

    def __init__(self, h_t_size: int, h_s_size: int):
        super(Attention, self).__init__()
        r'''
        Args:
            h_t_size (int): the size of target hidden state
            h_s_size (int): the size of source hidden state

        This calculates scores by
        ..math:
            \boldsymbol{h}^{target}_j
                \cdot
            \boldsymbol{W}^\text{attn} \boldsymbol{h}^\text{source}_i.
        '''
        self.w_a = nn.Linear(h_s_size, h_t_size, bias=False)

    def score(self, h_t: Tensor, h_s: Tensor) -> Tensor:
        return torch.bmm(self.w_a(h_s), h_t.transpose(1, 2))


class ConcatAttention(Attention):

    def __init__(self, h_t_size: int, h_s_size: int, v_a_size: int):
        r'''
        Args:
            h_t_size (int): Size of target hidden state
            h_s_size (int): Size of source hidden state
            v_a_size (int): Size of parameter :math:`\boldsymbol{v}^\text{attn}`

        This calculates scores by
        ..math:
            \boldsymbol{v}^{attn}
                \cdot
            \tanh\left(
                \boldsymbol{W}^\text{attn}
                    \left[
                        \boldsymbol{h}^\text{target};
                        \boldsymbol{h}^\text{source}
                    \right]
            \right)
        where :math:`[\boldsymbol{v}_1;\boldsymbol{v}_2]` denotes concatenation
        of :math:`\boldsymbol{v}_1` and :math:`\boldsymbol{v}_2`.
        '''

        super(Attention, self).__init__()
        self.v_a_transposed = nn.Linear(v_a_size, 1, bias=False)
        self.w_a_cat = nn.Linear(h_t_size + h_s_size, v_a_size, bias=False)

    def score(self, h_t: Tensor, h_s: Tensor) -> Tensor:
        return self.v_a_transposed(torch.tanh(self.w_a_cat(torch.cat((h_t, h_s), 2))))

    def align(self, h_t: Tensor, h_s: Tensor) -> Tensor:
        return nn.functional.softmax(self.score(h_t, h_s), dim=1)


def setup_attention(config: dict, seqtypes: List[SeqType]) -> Union[None, Attention]:

    enc_time_hidden_size = config['time_embed_size'] * len(seqtypes)

    if config['attn_type'] == 'general':
        # h_t·(W_a h_s)
        return GeneralAttention(config['dec_hidden_size'], enc_time_hidden_size)
    elif config['attn_type'] == 'concat':
        # v_a·tanh(W[h_t;h_s])
        return ConcatAttention(h_t_size=config['dec_hidden_size'],
                               h_s_size=config['enc_time_embed_size'] * len(seqtypes),
                               v_a_size=config['dec_hidden_size'])
    else:
        return None


class Encoder(nn.Module):
    def __init__(self, config: Config, device: torch.device, enc_type: str = "None"):

        super(Encoder, self).__init__()
        #self.used_seqtypes = [SeqType.NormMovRefLong,
                              #SeqType.NormMovRefShort,
                              #SeqType.StdLong,
                              #SeqType.StdShort] \
            #if config.use_standardization \
            #else [SeqType.NormMovRefLong,
                  #SeqType.NormMovRefShort]
        #self.used_rics = config.rics
        #self.use_extra_rics = len(self.used_rics) > 1
        #self.base_ric = config.base_ric
        #self.extra_rics = [ric for ric in self.used_rics if ric != self.base_ric]
        self.base_ric_hidden_size = config['base_ric_hidden_size']
        self.ric_hidden_size = config['ric_hidden_size']
        self.hidden_size = config['enc_hidden_size']
        self.n_layers = config['enc_n_layers']
        self.prior_encoding = 0
        self.dropout = config['use_dropout']
        self.device = device
        self.attn = config['use_attn']

        self.use_dropout = config['use_dropout']
        self.use_series = config['use_series']
        self.ric_seqtype_to_mlp = dict()
        
        self.input_size = 1
        self.seq_length = config['seq_length']
        self.series_length = config['max_series_size']
        self.output_size = config['enc_output_size']
        #self.seq_enc = CNN(self.input_size,
                #self.hidden_size,
                #self.output_size,
                #n_layers=self.n_layers).to(self.device)
        self.seq_enc = CNN(self.input_size,
                self.hidden_size,
                self.output_size,
                int((self.seq_length)/4)*self.output_size,
                n_layers=self.n_layers).to(self.device)                
        self.series_enc = CNN(self.input_size,
                self.hidden_size,
                self.output_size,
                int((self.series_length)/4)*self.output_size,
                n_layers=self.n_layers).to(self.device)
        self.enc_type = enc_type

        #if self.enc_type == "summ":
            #lengths = [config['batch_size']]
        #else:
        #lengths = [config['seq_lengths']]
        #lengths = [12]
        #total_length = sum(lengths)
        #input(config['batch_size'])
        multiple = 1
        if self.use_series:
            multiple = 2
        self.cat_hidden_size = config['max_series_size']*(multiple-1) + config['seq_length'] + config['enc_output_size']*multiple
        #print(self.cat_hidden_size)
        #self.cat_hidden_size = \
            #total_length + self.prior_encoding * self.base_ric_hidden_size + (self.prior_encoding * self.base_ric_hidden_size + \
            #(len(lengths) - self.prior_encoding) * self.ric_hidden_size)
        
        #input([self.cat_hidden_size,self.hidden_size])
        self.dense = nn.Linear(self.cat_hidden_size,self.hidden_size)
        self.attn_dense = nn.Linear(self.output_size*multiple,self.hidden_size)
        #if self.attn:
            #print(self.output_size*2,self.hidden_size)

        if self.use_dropout:
            self.drop = nn.Dropout(p=0.30)

    def forward(self,
                batch: Batch,
                mini_batch_size: int) -> Tuple[Tensor, Tensor]:

        #D = OrderedDict()  # low-level representation
        #input(batch)

        attn_vector = []
        if self.enc_type == "summ":
            rows = batch.token.numpy()
            
            if len(rows[0]) < self.cat_hidden_size:
                diff = self.cat_hidden_size - len(rows[0])
            else:
                diff = 0 
                
            seq_vals = []
            import numpy as np
            for row in rows:
                row = list(row)
                for i in range(diff):
                    row.append(-1)         
                row = np.asarray(row)
                seq_vals.append(torch.tensor(row).float())
        else:
            seq_vals = batch.passengers
        
        series_vals = batch.series # Might need to put this into else statement if self.enc_type == "summ"
        seq_size = self.seq_length
        vals = []
        for i in range(len(seq_vals)):
            seq = seq_vals[i].tolist()
            diff = [0]*(seq_size-len(seq))
            val = seq + diff
            vals.append(val)
                        
        l_short = torch.tensor(vals).to(self.device)
        h_short = self.seq_enc(l_short.view(len(seq_vals),1,1,seq_size))
        h_short = h_short.view(h_short.size(0),h_short.size(-1))
        
        #input(h_short.shape)
        
        if self.use_series:
            series_size = self.series_length
            vals = []
            for i in range(len(series_vals)):
                series = series_vals[i].tolist()
                diff = [0]*(series_size-len(series))
          
                val = series + diff
                vals.append(val)
                
            l_long = torch.tensor(vals).to(self.device)
            #input(l_long.shape)
            try:
                h_long = self.series_enc(l_long.view(len(series_vals),1,1,series_size))
            except:
                print(l_long)
                input(["l_long",l_long.shape,len(series_vals),series_size])
                h_long = self.series_enc(l_long.view(len(series_vals),1,1,series_size))
            h_long = h_long.view(h_long.size(0),h_long.size(-1))
   
        if self.use_series:
            concatenation = torch.cat((l_short,l_long,h_short,h_long),1)
            
            if self.attn:
                attn_vector = torch.cat((h_short,h_long),1)
            #input([h_short.shape,h_long.shape])
        else:
            concatenation = torch.cat((l_short,h_short),1)
            
            if self.attn:
                attn_vector = h_short
            
        #input(concatenation.shape)

        try:
            enc_hidden = self.dense(concatenation)
        except:
            input(["l_short, h_short, concatenation, cat hidden, hidden",l_short.shape,h_short.shape,concatenation.shape,self.cat_hidden_size,self.hidden_size])   
            enc_hidden = self.dense(concatenation)        
        
        if self.use_dropout:
            enc_hidden = self.drop(enc_hidden)

        if len(attn_vector) > 0:
            #input(attn_vector.shape)
            attn_vector = self.attn_dense(attn_vector)
            attn_vector = attn_vector.view(mini_batch_size, 1, -1)
            
        #input(enc_hidden.shape)

        return (enc_hidden, attn_vector)


class Decoder(nn.Module):
    def __init__(self,
                 config: Config,
                 output_vocab_size: int,
                 attn: Union[None, Attention],
                 device: torch.device,
                 dec_type: str = "None"):

        super(Decoder, self).__init__()

        self.device = device
        #input(
        self.dec_hidden_size = config['dec_hidden_size']
        self.word_embed_size = config['word_embed_size']
        self.series_embed_size = config['word_embed_size']
        #self.word_embed_size = int(output_vocab_size**(1/4))
        
        #self.time_embed_size = config.time_embed_size
        self.attn = attn
        self.dec_type = dec_type

        self.word_embed_layer = nn.Embedding(output_vocab_size, self.word_embed_size, padding_idx=0) 
        self.series_embed_layer = nn.Embedding(output_vocab_size, self.series_embed_size) 
        
        #print([output_vocab_size+1, self.word_embed_size])
        #self.time_embed_layer = nn.Embedding(TIMESLOT_SIZE, self.time_embed_size)
        self.output_layer = nn.Linear(self.dec_hidden_size, output_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.dec_hidden_size = self.dec_hidden_size
        self.input_hidden_size = self.word_embed_size
        #input([self.input_hidden_size,self.dec_hidden_size])
        self.recurrent_layer = nn.LSTMCell(self.input_hidden_size, self.dec_hidden_size)
        
        self.summarizer_loss = config['summarizer_loss']
        self.quantifier_loss = config['quantifier_loss']
        self.summarizer_training_loss = config['summarizer_training_loss']

        if isinstance(attn, Attention):
            #self.enc_hidden_size = config['enc_hidden_size']
            attn_size = config['dec_hidden_size']*2
            self.linear_attn = nn.Linear(attn_size, self.dec_hidden_size)

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        zeros = torch.zeros(batch_size, self.dec_hidden_size, device=self.device)
        #double_zeros = torch.zeros(batch_size*2, self.dec_hidden_size, device=self.device)
        self.h_n = zeros
        self.c_n = zeros
        #input(zeros.shape)
        return (self.h_n, self.c_n)

    def forward(self,
                word: Tensor,
                attn_vector: Tensor,
                batch_size: int) -> Tuple[Tensor, Tensor]:
        
        #print(word.shape)
        #input(self.h_n[0])
        
        #input(copied_hidden.shape)
        #weights = []
        #for i in range(batch_size):
            #weight = self.attn(torch.cat((self.h_n[0][0]))
        #input(weight)
        #weight = 0.0
        word_embed = self.word_embed_layer(word.type(torch.LongTensor).to(self.device)).view(batch_size,self.word_embed_size)
        self.h_n, self.c_n = self.recurrent_layer(word_embed, (self.h_n, self.c_n))
        hidden = self.h_n         
        
               
        
        #input(self.h_n.shape)
        #copied_hidden = copied_hidden.view(1,
        #print(word)
        #copied_hidden = copied_hidden[0][0].view(1,1,-1)
        #input(word_embed.unsqueeze(0))
        #input([copied_hidden[0][0].view(1,1,-1).shape,attn_vector.unsqueeze(0).shape])
        #input([copied_hidden.shape,attn_vector.shape])
        if isinstance(self.attn, Attention):
            copied_hidden = hidden.unsqueeze(1)
            weight = self.attn(copied_hidden,attn_vector)
            #print([weights.view(batch_size, -1, 1).shape,word.float().view(batch_size, 1, -1).shape])
            weighted = torch.bmm(weight.view(batch_size, -1, 1),
                                     attn_vector.float().view(batch_size, 1, -1))
            #input(weighted.shape)
            weighted = weighted.squeeze()
            #input([self.h_n.shape,weighted.unsqueeze(0).shape])
            hidden = self.h_n
            hidden = torch.tanh(self.linear_attn(torch.cat((hidden, weighted), 1)))
            self.h_n = hidden            
        else:
            weight = 0.0
        #print(weights)
        #input(weights.shape)
        #input(word_embed.shape)
        

        #if isinstance(self.attn, Attention):
            ##_, num_copy, _ = seq_ric_tensor.size()

            #if isinstance(self.attn, ConcatAttention):
                #copied_hidden = hidden.expand(num_copy, batch_size, self.hidden_size)
                #copied_hidden = torch.transpose(copied_hidden, 0, 1)
            #else:
                #copied_hidden = hidden.unsqueeze(1)
            #weight = self.attn(copied_hidden)
            #input(weight)
            #weighted_ric = torch.bmm(weight.view(batch_size, -1, num_copy),
                                     #seq_ric_tensor.view(batch_size, num_copy, -1))
            #weighted_ric = weighted_ric.squeeze()
            #hidden = torch.tanh(self.linear_attn(torch.cat((hidden, weighted_ric), 1)))
            #self.h_n = hidden
        output = self.softmax(self.output_layer(hidden))
        
        return (output, weight)
    

class CNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 mid_size: int,
                 output_size: int,
                 linear_input: int,
                 n_layers: int = 3,
                 activation_function: str = 'tanh'):
        
        super(CNN,self).__init__()
        self.n_layers = n_layers
        
        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        else:
            raise NotImplementedError
        
        #print([input_size,mid_size,output_size])
        #output_size = 30
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(input_size,mid_size,kernel_size=(1,3),stride=1,padding=1),
            nn.BatchNorm2d(mid_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(mid_size,output_size,kernel_size=(1,3),stride=1,padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        #linear_input_size = 64
        self.linear_layers = nn.ModuleList()
        if n_layers == 1:
            self.linear_layers.append(nn.Linear(linear_input, output_size))
        else:
            self.linear_layers.append(nn.Linear(linear_input, mid_size))
            for _ in range(n_layers - 2):
                self.linear_layers.append(nn.Linear(mid_size, mid_size))
            self.linear_layers.append(nn.Linear(mid_size, output_size))            
    
    def forward(self, x: Tensor) -> Tensor:
        #input(x.shape)
        x = self.cnn_layers(x)
        x = x.view(x.size(0),-1)
        for i in range(self.n_layers): 
            x = self.linear_layers[i](x)
            x = self.activation_function(x)
        return x
        
class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 mid_size: int,
                 output_size: int,
                 n_layers: int = 3,
                 activation_function: str = 'tanh'):
        '''Multi-Layer Perceptron
        '''

        super(MLP, self).__init__()
        self.n_layers = n_layers

        assert(n_layers >= 1)

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        else:
            raise NotImplementedError

        self.MLP = nn.ModuleList()
        if n_layers == 1:
            self.MLP.append(nn.Linear(input_size, output_size))
        else:
            self.MLP.append(nn.Linear(input_size, mid_size))
            for _ in range(n_layers - 2):
                self.MLP.append(nn.Linear(mid_size, mid_size))
            self.MLP.append(nn.Linear(mid_size, output_size))

    def forward(self, x: Tensor) -> Tensor:
        out = x.float()
        for i in range(self.n_layers):
            input(out.shape)
            out = self.MLP[i](out)
            out = self.activation_function(out)
        return out


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder: List,
                 decoder: List,
                 device: torch.device,
                 learn_templates: bool,
                 cross_entropy: bool,
                 truth_loss: bool):
        super(EncoderDecoder, self).__init__()

        self.device = device
        
        self.encoders = [x.to(self.device) for x in encoder]
        self.decoders = [x.to(self.device) for x in decoder]
        self.encoder = self.encoders[0]
        self.decoder = self.decoders[0]      
        self.learn_templates = learn_templates
        self.cross_entropy = cross_entropy
        self.truth_loss = truth_loss
        if learn_templates:
            self.temp_decoder = self.decoders[1]
        #self.decoders = []
        #input()
        #for d in decoder:
            #decoder = decoder.to(self.device)
            #self.decoders.append(decoder)
            
        self.weight_lambda = 10 ** 0  # for supervised attention

    def forward(self,
                batch: Batch,
                mini_batch_size: int,
                tokens: Tensor,
                template_tokens: Tensor,
                vocab: Vocab,
                criterion: nn.NLLLoss,
                phase: Phase,
                template_vocab: Vocab) -> Tuple[nn.NLLLoss, Tensor, Tensor]:

        #for i in range(len(self.encoders)):
            #encoder = self.encoders[i]
            #decoder = self.decoders[i]
            #decoder.init_hidden(mini_batch_size)
 
            #decoder.h_n, attn_vector = encoder(batch, mini_batch_size)
        
        #input(type(series_embedding))
        self.decoder.init_hidden(mini_batch_size)
        self.decoder.h_n, attn_vector = self.encoder(batch, mini_batch_size)

        if self.learn_templates:
            #self.temp_decoder = self.decoder
            self.temp_decoder.init_hidden(mini_batch_size)
        
            self.temp_decoder.h_n, attn_vector = self.encoder(batch, mini_batch_size)
            
        n_tokens,n_cols = tokens.size()
        
        decoder_input = tokens[0] 
        #input(len(decoder_input))
        
        loss = 0.0
        pred = []
        attn_weight = []
        pred.append(decoder_input.cpu().numpy())
        if phase == Phase.Train:            
            
            for i in range(1,n_tokens):

                summ_decoder_output, weight = \
                    self.decoder(decoder_input, attn_vector, mini_batch_size)

                temp_loss = 0
                
                if self.learn_templates:
                    temp_decoder_output, weight = \
                    self.temp_decoder(decoder_input, attn_vector, mini_batch_size)  
                    
                    topv_, topi_ = temp_decoder_output.data.topk(1)
                    multiplier = 1
                    special_chars = set(["A","S","Q","TW"])
                    for j in range(len(template_tokens[i])):
                        token = int(template_tokens[i][j].item())
                        pred_token = int(topi_[j].item())
                        if pred_token != token and template_vocab.itos[token] in special_chars:
                            multiplier += 1
                            
                            if template_vocab.itos[token] == 'S' and (self.decoder.summarizer_loss or self.decoder.summarizer_training_loss):
                                multiplier += 1
                                
                            if template_vocab.itos[token] == 'Q' and self.decoder.quantifier_loss:
                                multiplier += 1                          
                                
                    temp_loss = multiplier*criterion(temp_decoder_output, template_tokens[i].to(self.device))
                
                #input([summ_decoder_output.shape,tokens[i].shape])
                loss += criterion(summ_decoder_output, tokens[i]) + temp_loss
                
                topv, topi = summ_decoder_output.data.topk(1)
               
                pred.append([t[0] for t in topi.cpu().numpy()])
            
                if self.decoder.attn:
                    weight = weight.squeeze()
                    attn_weight.append(weight)

                decoder_input = tokens[i]
            
            # Add in truth value loss
            if self.truth_loss:
                tmp=loss
                gold_summaries = []
                pred_summaries = []
                for i in range(tokens.size(1)):
                    gold_summary = ""
                    for j in range(1,len(tokens[:,i])):
                        token = tokens[:,i][j]
                        if vocab.itos[token] == SpecialToken.EOS.value:
                            gold_summary = gold_summary[:-1]
                            break
                        gold_summary += vocab.itos[token] + " "
                    gold_summaries.append(gold_summary)
                
                import numpy as np
                pred_tensor = torch.tensor(np.array(pred))
                for i in range(pred_tensor.size(1)):
                    pred_summary = ""
                    for j in range(1,len(pred_tensor[:,i])):
                        p = pred_tensor[:,i][j]
                        if vocab.itos[p] == SpecialToken.EOS.value:
                            pred_summary = pred_summary[:-1]
                            break                    
                        pred_summary += vocab.itos[p] + " "  
                    pred_summaries.append(pred_summary)
                
                gold_truths = []
                pred_truths = []
                for i in range(len(gold_summaries)):
                    gold_truths.append(getTruthValue(gold_summaries[i],batch.sax[i]))
                    pred_truths.append(getTruthValue(pred_summaries[i],batch.sax[i]))
                    
                gold_truths = torch.tensor(np.array(gold_truths)).float()
                pred_truths = torch.tensor(np.array(pred_truths)).float()
               
                if self.cross_entropy:
                    n_classes = 3
                    sublist = []
                    pred_classes = []
                    for i in range(n_classes):
                        pred_classes.append([])
                    gold_classes = []
                    buckets = [float(x)/n_classes for x in range(1,n_classes+1)]
                    
                    for i in range(len(pred_truths)):
                        pred_truth = pred_truths[i]
                        
                        val = 1
                        for j in range(n_classes):
                            if pred_truth < buckets[j]:
                                pred_classes[j].append(val)
                                val = 0
                            else:
                                pred_classes[j].append(0)
                            
                    for i in range(len(gold_truths)):
                        gold_truth = gold_truths[i]
                        val = 2
                        for j in range(len(buckets)):
                            if gold_truth < buckets[j]:
                                val = j
                                break
                        gold_classes.append(val)
                        
                    truth_loss = torch.nn.CrossEntropyLoss()
                    pred_classes = torch.tensor(pred_classes).float()
                    gold_classes = torch.tensor(gold_classes)
                    
                    loss += truth_loss(pred_classes.t(),gold_classes)*100
                else:
                    truth_loss = torch.nn.MSELoss()
                    loss += truth_loss(gold_truths,pred_truths)*100
        else:
            for i in range(1, GENERATION_LIMIT):
                #input([i,decoder_input])
                #decoder_output, weight = \
                    #self.decoder(decoder_input, attn_vector, mini_batch_size)                
                outputs = []
                temp_loss = 0
                if self.learn_templates:
                    temp_decoder_output, weight = \
                    self.temp_decoder(decoder_input, attn_vector, mini_batch_size)    
                    
                    topv_, topi_ = temp_decoder_output.data.topk(1)
                    multiplier = 1
                    special_chars = set(["A","S","Q","TW"])
                    if i < n_tokens:
                        for j in range(len(template_tokens[i])):
                            token = int(template_tokens[i][j].item())
                            pred_token = int(topi_[j].item())
                            if pred_token != token and template_vocab.itos[token] in special_chars:
                                multiplier += 1
                                
                                if template_vocab.itos[token] == 'S' and (self.decoder.summarizer_loss or self.decoder.summarizer_training_loss):
                                    multiplier += 1
                                
                                if template_vocab.itos[token] == 'Q' and self.decoder.quantifier_loss:
                                    multiplier += 1
                                    
                        temp_loss = multiplier*criterion(temp_decoder_output, template_tokens[i].to(self.device))                    
                decoder_output, weight = \
                    self.decoder(decoder_input, attn_vector, mini_batch_size)
                outputs.append(decoder_output)                

                if type(n_tokens) != int:
                    n_tokens = n_tokens[0]
                if i < n_tokens:
                    loss += criterion(decoder_output, tokens[i]) + temp_loss

                topv, topi = decoder_output.detach().topk(1)
                pred.append([t[0] for t in topi.cpu().numpy()])

                if self.decoder.attn:
                    #input(weight.shape)
                    weight = weight.squeeze(2).cpu().detach().numpy()
                    attn_weight.append(weight)                    
    
                decoder_input = topi.squeeze()
                
            # Add in truth value loss
            if self.truth_loss:
                tmp=loss
                gold_summaries = []
                pred_summaries = []
                for i in range(tokens.size(1)):
                    gold_summary = ""
                    for j in range(1,len(tokens[:,i])):
                        token = tokens[:,i][j]
                        if vocab.itos[token] == SpecialToken.EOS.value:
                            gold_summary = gold_summary[:-1]
                            break
                        gold_summary += vocab.itos[token] + " "
                    gold_summaries.append(gold_summary)
                
                import numpy as np
                pred_tensor = torch.tensor(np.array(pred))
                for i in range(pred_tensor.size(1)):
                    pred_summary = ""
                    for j in range(1,len(pred_tensor[:,i])):
                        p = pred_tensor[:,i][j]
                        if vocab.itos[p] == SpecialToken.EOS.value:
                            pred_summary = pred_summary[:-1]
                            break                    
                        pred_summary += vocab.itos[p] + " "  
                    pred_summaries.append(pred_summary)
                
                gold_truths = []
                pred_truths = []
                for i in range(len(gold_summaries)):
                    gold_truths.append(getTruthValue(gold_summaries[i],batch.sax[i]))
                    pred_truths.append(getTruthValue(pred_summaries[i],batch.sax[i]))
                    
                gold_truths = torch.tensor(np.array(gold_truths)).float()
                pred_truths = torch.tensor(np.array(pred_truths)).float()
                
                if self.cross_entropy:
                    n_classes = 3
                    sublist = []
                    pred_classes = []
                    for i in range(n_classes):
                        pred_classes.append([])
                    gold_classes = []
                    buckets = [float(x)/n_classes for x in range(1,n_classes+1)]
                    
                    for i in range(len(pred_truths)):
                        pred_truth = pred_truths[i]
                        
                        val = 1
                        for j in range(n_classes):
                            if pred_truth < buckets[j]:
                                pred_classes[j].append(val)
                                val = 0
                            else:
                                pred_classes[j].append(0)
                            
                    for i in range(len(gold_truths)):
                        gold_truth = gold_truths[i]
                        val = 2
                        for j in range(len(buckets)):
                            if gold_truth < buckets[j]:
                                val = j
                                break
                        gold_classes.append(val)
                        
                    truth_loss = torch.nn.CrossEntropyLoss()
                    pred_classes = torch.tensor(pred_classes).float()
                    gold_classes = torch.tensor(gold_classes)
                    
                    loss += truth_loss(pred_classes.t(),gold_classes)*100
                else:
                    truth_loss = torch.nn.MSELoss()
                    loss += truth_loss(gold_truths,pred_truths)*100                
                    
                #print([gold_truths,pred_truths])
                #tmp=loss
                
                #alpha = 0.3
                #loss = alpha*truth_loss(gold_truths,pred_truths) + (1-alpha)*loss
                loss += truth_loss(gold_truths,pred_truths)*100  
            
        return (loss, pred, attn_weight)
