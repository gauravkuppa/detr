import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import argparse
import numpy as np
from torchsummary import summary


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, input_channels=3):
        super().__init__()
        #self.pos_encoder = PositionEmbeddingSine()# TODO: positional encoding, what is hidden dim?

        # TODO: also seems that encoder/decoder take sequential info, can you transform this into fmap input?
        self.embedding_conv = nn.Conv2d(1, d_model, kernel_size=(1, input_channels), stride=(1, input_channels))# TODO: add some sort of image embedding
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.activation = _get_activation_fn(activation)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, src):
        # reshape inputs

        # src = X
        # say input is X
        # dim(X) = [batch, channels, width, height]

        bs, c, h, w = src.shape
        src = src.permute(0, 2, 3, 1).contiguous()
        # dim(x) = [batch, height, width, channels]
        src = src.view(bs, h, w * c)
        # dim(x) = [batch, height, width * channels]

        # what is query_embed?

        print("transformer:", src.size())
        # inputs --> embeddings
        src = src.unsqueeze(1)
        print("transformer:", src.size())
        src = self.activation(self.embedding_conv(src))
        print("transformer:", src.size())
        src = src.permute([0, 2, 3, 1])  # move channels to the end
        print("transformer:", src.size())
        #src = src.view(src.shape[0], -1, src.shape[1])
        #print("transformer:", src.size())
        # TODO: figure what the fuck is going on here
        # positional encoding
        # src = self.add_timing_signal(src)


        tgt = torch.zeros(100, self.d_model)
        # call encoder
        memory = self.encoder(src)
        # call decoder
        hs = self.decoder(tgt, memory)

        # flatten NxCxHxW to HWxNxC

        #src = src.flatten(2).permute(2, 0, 1)
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        #query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        #mask = mask.flatten(1)

        #tgt = torch.zeros_like(query_embed)
        #memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
        #                  pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        output = src

        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 ffn_layer="conv", norm="spectral", activation="relu"):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout) # TODO: how to edit MultiHeadAttention?
        # Implementation of Feedforward model
        self.ffn1 = nn.Conv2d(d_model, dim_feedforward) if ffn_layer=="conv" else nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.ffn2 = nn.Conv2d(d_model, dim_feedforward) if ffn_layer=="conv" else nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.utils.spectral_norm if norm=="spectral" else nn.LayerNorm(d_model) # TODO: what does spectral normalization do? nn.SpectralNorm()
        self.norm2 = nn.utils.spectral_norm if norm=="spectral" else nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.d_model = d_model

    # positional encoding
    def add_timing_signal(self, X, min_timescale=1.0, max_timescale=1.0e4):
        num_dims = len(X.shape) - 2  # 2 corresponds to batch and hidden_size dimensions
        num_timescales = self.d_model // (num_dims * 2)
        log_timescale_increment = np.log(max_timescale / min_timescale) / (num_timescales - 1)
        inv_timescales = min_timescale * torch.exp(
            (torch.arange(num_timescales).float() * -log_timescale_increment))
        inv_timescales = inv_timescales.to(X.device)
        total_signal = torch.zeros_like(X)  # Only for debugging purposes
        for dim in range(num_dims):
            length = X.shape[dim + 1]  # add 1 to exclude batch dim
            position = torch.arange(length).float().to(X.device)
            scaled_time = position.view(-1, 1) * inv_timescales.view(1, -1)
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
            prepad = dim * 2 * num_timescales
            postpad = self.d_model - (dim + 1) * 2 * num_timescales
            signal = F.pad(signal, (prepad, postpad))
            for _ in range(1 + dim):
                signal = signal.unsqueeze(0)
            for _ in range(num_dims - 1 - dim):
                signal = signal.unsqueeze(-2)
            print("X", X.size())
            print("signal", signal.size())
            X += signal
            total_signal += signal
        return X

    def forward(self, src):
        q = k = self.add_timing_signal(src).view(src.shape[0], -1, src.shape[3])
        v = src.view(src.shape[0], -1, src.shape[3])
        src2 = self.self_attn(q, k, v)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn2(self.dropout(self.activation(self.ffn1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src




class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 ffn_layer=None, norm=None, activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.ffn1 = nn.Conv2d(d_model, dim_feedforward, 1) if ffn_layer=="conv" else nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.ffn2 = nn.Conv2d(d_model, dim_feedforward, 1) if ffn_layer=="conv" else nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.utils.spectral_norm if norm=="spectral" else nn.LayerNorm(d_model) # TODO: what does spectral normalization do? nn.SpectralNorm()
        self.norm2 = nn.utils.spectral_norm if norm=="spectral" else nn.LayerNorm(d_model)
        self.norm3 = nn.utils.spectral_norm if norm=="spectral" else nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.d_model = d_model

    # positional encoding
    def add_timing_signal(self, X, min_timescale=1.0, max_timescale=1.0e4):
        num_dims = len(X.shape) - 2  # 2 corresponds to batch and hidden_size dimensions
        num_timescales = self.d_model // (num_dims * 2)
        log_timescale_increment = np.log(max_timescale / min_timescale) / (num_timescales - 1)
        inv_timescales = min_timescale * torch.exp(
            (torch.arange(num_timescales).float() * -log_timescale_increment))
        inv_timescales = inv_timescales.to(X.device)
        total_signal = torch.zeros_like(X)  # Only for debugging purposes
        for dim in range(num_dims):
            length = X.shape[dim + 1]  # add 1 to exclude batch dim
            position = torch.arange(length).float().to(X.device)
            scaled_time = position.view(-1, 1) * inv_timescales.view(1, -1)
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
            prepad = dim * 2 * num_timescales
            postpad = self.d_model - (dim + 1) * 2 * num_timescales
            signal = F.pad(signal, (prepad, postpad))
            for _ in range(1 + dim):
                signal = signal.unsqueeze(0)
            for _ in range(num_dims - 1 - dim):
                signal = signal.unsqueeze(-2)
            X += signal
            total_signal += signal
        return X

    def forward(self, tgt, memory):
        q = k = self.add_timing_signal(tgt)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(self.add_timing_signal(tgt),
                                   self.add_timing_signal(memory),
                                   memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ffn2(self.dropout(self.activation(self.ffn1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k=None, d_v=None, weight_layer=None, norm="spectral", dropout=0.1):
        super().__init__()

        self.n_head = n_head
        d_k = d_k if d_k else d_model
        d_v = d_v if d_v else d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Conv2d(d_model, n_head * self.d_k, 1) if weight_layer == "conv" else nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Conv2d(d_model, n_head * d_k, 1) if weight_layer == "conv" else nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Conv2d(d_model, n_head * d_v, 1) if weight_layer == "conv" else nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Conv2d(n_head * d_v, d_model, 1) if weight_layer == "conv" else nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.utils.spectral_norm if norm=="spectral" else nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        print("len1:", d_k, d_v, n_head)
        print("len2:", sz_b, len_q, len_k, len_v)
        print("len3", q.size())
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        print(q.size())
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        print(q.size())
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer():
    parser = get_args_parser()
    args = parser.parse_args()
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    """parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")"""
    # * Backbone
    """parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")"""
    # TODO: do i need this
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots") # TODO: what is the default value?
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument("--ffn_layer", default="conv", type=str, choices=("conv", "linear"),
                        help="LaTransformeryers used in Feed Forward Network")


    """# * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')"""
    return parser

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    #TODO: maybe test with other activation functions?
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "elu":
        return F.elu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def main():
    device = torch.device('cpu')
    t = Transformer().to(device)
    
    print("Transformer")
    #print(t)
    summary(t, (3, 256, 256))
    
main()