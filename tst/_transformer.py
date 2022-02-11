import torch
import torch.nn as nn

from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE

#class PositionalEncoding(nn.Module):

    #def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        #super().__init__()
        #self.dropout = nn.Dropout(p=dropout)

        #position = torch.arange(max_len).unsqueeze(1)
        #div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        #pe = torch.zeros(max_len, 1, d_model)
        #pe[:, 0, 0::2] = torch.sin(position * div_term)
        #pe[:, 0, 1::2] = torch.cos(position * div_term)
        #self.register_buffer('pe', pe)

    #def forward(self, x: Tensor) -> Tensor:
        #"""
        #Args:
            #x: Tensor, shape [seq_len, batch_size, embedding_dim]
        #"""
        #x = x + self.pe[:x.size(0)]
        #return self.dropout(x)


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
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = 24):
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

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)
        self.seq_len = d_input

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

    def forward(self, x: torch.Tensor, output_seq_len: int, device: torch.device) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
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
        for layer in self.layers_encoding:
            encoding = layer(encoding)
        #encoding = self.encoder_embedding(encoding)
        #input(encoding.shape)


        # Decoding stack
        decoding = encoding
        decoding = decoding.view(-1,decoding.shape[2],decoding.shape[1])

        seq_linear = nn.Linear(K,output_seq_len)
        decoding = seq_linear(decoding).view(-1,output_seq_len,decoding.shape[1])

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(output_seq_len, self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding)

        #for layer in self.layers_decoding:
            #decoding = layer(decoding, encoding)

        # Output module
        #print(decoding.shape)
        output = self._linear(decoding)
        #input(output.shape)
        #output = torch.sigmoid(output)
        #softmax = nn.Softmax(dim=1)
        #output = softmax(output)
        output = nn.functional.log_softmax(output,dim=1)

        #input(output.shape)
        output = output.view(output.shape[0]*output.shape[1],-1)

        return output
