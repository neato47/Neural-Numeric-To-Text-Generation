import torch
import torch.nn as nn

from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE


class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Switch between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    pe_period:
        If using the ``'regular'` pe, then we can define the period. Default is ``24``.
    """

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_temp_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 use_decoder: bool = False,
                 pe_period: int = 24,
                 enc_hidden_size: int = 1,
                 max_series_size: int = 1,
                 seq_length: int = 7):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        #self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      #q,
                                                      #v,
                                                      #h,
                                                      #attention_size=attention_size,
                                                      #dropout=dropout,
                                                      #chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([nn.TransformerDecoderLayer(d_model,
                                                                         h,
                                                                         dropout=dropout) for _ in range(N)])
        self.temp_layers_decoding = nn.ModuleList([nn.TransformerDecoderLayer(d_model,
                                                                         h,
                                                                         dropout=dropout) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        self.cat_hidden_size = (d_model+d_input)*(max_series_size+seq_length)
        self.dense = nn.Linear(self.cat_hidden_size,enc_hidden_size)
        #print(d_model,max_series_size,seq_length)
        #print(d_model*(max_series_size+seq_length),enc_hidden_size)
        self.attn_dense = nn.Linear(d_model*(max_series_size+seq_length),enc_hidden_size)

        if d_temp_output != None:
            self._temp_linear = nn.Linear(d_model, d_temp_output)
        self.input_len = seq_length + max_series_size
        self.use_decoder = use_decoder

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor, device: torch.device, y: torch.Tensor = None, decoder_decoder_attn: bool = False, PAD_TOKEN: float = 1.0, EOS_TOKEN: float = 3.0) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size*K, d_output).
        """
        x = x.to(device)
        batch_size = x.shape[0]
        K = x.shape[1]

        # Embedding module
        encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            #pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        #print(encoding.shape)
        for layer in self.layers_encoding:
            encoding = layer(encoding)
        #print(encoding.shape)

        #encoding = self.encoder_embedding(encoding)

        # Decoding stack
        if y != None:
            # Shift outputs right for input
            for i in range(len(y)):
                tmp = y[i].tolist()

                # Remove EOS token
                tmp.remove(EOS_TOKEN)

                # Replace with pad token
                tmp = tmp + [PAD_TOKEN] # Continue this

                y[i] = torch.tensor(tmp)

            y = y.view(y.shape[0],y.shape[1],-1)

            decoding = self._embedding(y)
            temp_decoding = self._embedding(y)
        else:
            decoding = encoding
            temp_decoding = encoding

        # Add positional encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding)
            temp_decoding.add_(positional_encoding)

        if self.use_decoder:
            import copy
            for i in range(len(self.layers_decoding)):
                #past_decoding = copy.deepcopy(decoding)
                #past_temp_decoding = copy.deepcopy(temp_decoding)

                past_decoding = decoding.detach()
                past_temp_decoding = temp_decoding.detach()

                decoding_layer = self.layers_decoding[i]
                temp_decoding_layer = self.temp_layers_decoding[i]

                if decoder_decoder_attn:
                    decoding = decoding_layer(past_decoding,encoding,past_temp_decoding)
                    temp_decoding = temp_decoding_layer(past_temp_decoding,encoding,past_decoding)
                else:
                    decoding = decoding_layer(past_decoding,encoding)
                    temp_decoding = temp_decoding_layer(past_temp_decoding,encoding)
        else:
            #print(x.view(batch_size,-1).shape,encoding.view(batch_size,-1).shape)
            concatenation = torch.cat((x.view(batch_size,-1),encoding.view(batch_size,-1)),1)
            #input(concatenation[0])
            #print(batch_size)
            #input(encoding.view(batch_size,-1).shape)
            attn_vector = self.attn_dense(encoding.view(batch_size,-1))
            #print(attn_vector.shape)
            attn_vector = attn_vector.view(batch_size, 1, -1)
            #input(attn_vector.shape)
            encoding = self.dense(concatenation)
            concatenation.detach()
            decoding.detach()
            temp_decoding.detach()

            if self._generate_PE is not None:
                positional_encoding.detach()


            return encoding, attn_vector

        # Output module

        output = self._linear(decoding)
        temp_output = self._temp_linear(temp_decoding)
        #input([output.shape,decoding.shape])
        #output = torch.sigmoid(output)
        #softmax = nn.Softmax(dim=1)
        #output = softmax(output)
        output = nn.functional.log_softmax(output,dim=1)
        temp_output = nn.functional.log_softmax(temp_output,dim=1)

        output = output.view(output.shape[0]*output.shape[1],-1)
        temp_output = temp_output.view(temp_output.shape[0]*temp_output.shape[1],-1)
        x.detach()


        return output, temp_output
