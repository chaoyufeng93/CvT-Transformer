import torch
import math
import numpy as np
import torch.nn.functional as F
from einops import rearrange

class Conv_Token_Emb(torch.nn.Module):
  def __init__(self, in_channel, emb_dim, k_size, stride, padding):
    super(Conv_Token_Emb, self).__init__()
    self.conv = torch.nn.Conv2d(in_channels = in_channel, out_channels = emb_dim, kernel_size = k_size, stride = stride, padding = padding)
    self.layer_norm = torch.nn.LayerNorm(emb_dim, eps = 1e-5)

  def forward(self, x):
    #shape_flow: x (b_s, c, h, w) > conv+permute (b_s, h, w, c) > (b_s, (h*w), c)
    out = self.conv(x).permute(0,2,3,1) # (b_s, h, w, c)
    h, w = out.shape[1], out.shape[2]
    out = rearrange(out, 'b h w c -> b (h w) c')
    out = self.layer_norm(out)
    out = rearrange(out, 'b (h w) c -> b h w c',h = h, w = w).permute(0,3,1,2) # (b_s, c, h, w)
    return out

class Attention(torch.nn.Module):
  def __init__(self, emb_dim, head, dropout = 0):
    super(Attention, self).__init__()
    self.emb_dim = emb_dim
    self.head = head
    self.softmax = torch.nn.Softmax(dim = -1)
    self.dropout = torch.nn.Dropout(p = dropout)

  #sent k.T in (transpose k before sent in forward)
  def forward(self, q, k, v):
    qk = torch.matmul(q, k) / math.sqrt(self.emb_dim//self.head)
    att_w = self.dropout(self.softmax(qk))
    out = torch.matmul(att_w, v)
    return out

class Multi_Head_ATT(torch.nn.Module):
  def __init__(self, emb_dim, multi_head = 1, dropout = 0):
    super(Multi_Head_ATT,self).__init__()
    self.head = multi_head
    self.emb_dim = emb_dim
    self.attention = Attention(emb_dim, multi_head, dropout = dropout)
    self.WO = torch.nn.Linear(emb_dim, emb_dim)
    self.dropout = torch.nn.Dropout(p = dropout)

  def forward(self, q,k,v):
    seq_len = q.shape[1]
    if self.head == 1:
      out = self.attention(q, k.permute(0,2,1), v)
    else:
      # (b_s, seq_len, head, emb//head) > (b_s, head, seq_len, emb_dim//head)
      q = q.view(-1,seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3)
      k = k.view(-1,seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3).permute(0,1,3,2)
      v = v.view(-1,seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3)
      out = self.attention(q, k, v).permute(0,2,1,3).contiguous().view(-1,seq_len,self.emb_dim)
    out = self.WO(out)   
    out = self.dropout(out)
    return out

class Feed_Forward(torch.nn.Module): 
  def __init__(self, emb_dim, dim_expan = 4, dropout = 0):
    super(Feed_Forward,self).__init__()
    self.w1 = torch.nn.Linear(emb_dim, dim_expan*emb_dim)
    self.w2 = torch.nn.Linear(dim_expan*emb_dim, emb_dim)
    self.gelu = torch.nn.GELU()
    self.LN = torch.nn.LayerNorm(emb_dim, eps = 1e-5)
    self.dropout = torch.nn.Dropout(p = dropout)
  def forward(self,x):
    res = x
    x = self.LN(x)
    out = self.dropout(self.gelu(self.w1(x)))
    out = self.w2(out)
    out = self.dropout(out)
    out = out + res
    return out

# Conv_Emb > Conv_proj > MHSA > MLP
class Cov_Trans_Block(torch.nn.Module):
  def __init__(self, emb_dim, multi_head = 1, dim_expan = 4, dropout = 0, final = False):
    super(Cov_Trans_Block, self).__init__()
    self.is_final = final
    self.layer_norm = torch.nn.LayerNorm(emb_dim, eps = 1e-5)
    # groups == depth wise conv
    self.conv_proj_q = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels = emb_dim, out_channels = emb_dim, kernel_size = 3, stride = 1, padding = 1, groups = emb_dim),
        torch.nn.BatchNorm2d(emb_dim),
        torch.nn.Conv2d(in_channels = emb_dim, out_channels = emb_dim, kernel_size = 1, stride = 1)
    )
    self.conv_proj_k = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels = emb_dim, out_channels = emb_dim, kernel_size = 3, stride = 1, padding = 1, groups = emb_dim),
        torch.nn.BatchNorm2d(emb_dim),
        torch.nn.Conv2d(in_channels = emb_dim, out_channels = emb_dim, kernel_size = 1, stride = 1)
    )
    self.conv_proj_v = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels = emb_dim, out_channels = emb_dim, kernel_size = 3, stride = 1, padding = 1, groups = emb_dim),
        torch.nn.BatchNorm2d(emb_dim),
        torch.nn.Conv2d(in_channels = emb_dim, out_channels = emb_dim, kernel_size = 1, stride = 1)
    )
    self.MHSA = Multi_Head_ATT(emb_dim, multi_head = multi_head, dropout = dropout)
    self.FF = Feed_Forward(emb_dim, dim_expan = dim_expan, dropout = dropout)

  def forward(self, x, h, w):
    res = x
    if self.is_final == True:
      cls = x[:,0,:].view(x.shape[0],1,x.shape[2])
      x = x[:,1:,:]
    x = rearrange(x, 'b (h w) c -> b h w c', h = h, w = w)
    x = x.permute(0,3,1,2)
    q, k, v = self.conv_proj_q(x), self.conv_proj_k(x), self.conv_proj_v(x)
    q, k, v = q.permute(0,2,3,1), k.permute(0,2,3,1), v.permute(0,2,3,1)
    q, k, v = rearrange(q, 'b h w c -> b (h w) c'), rearrange(k, 'b h w c -> b (h w) c'), rearrange(v, 'b h w c -> b (h w) c')
    if self.is_final == True:
      q, k, v = torch.cat([cls, q], dim = 1),torch.cat([cls, k], dim = 1),torch.cat([cls, v], dim = 1)
    out = self.MHSA(q, k.permute(0,2,1), v)
    out += res
    out = self.FF(out)
    return out

class CvTransformer(torch.nn.Module):
  def __init__(
               self, class_num,
               in_channel = [3, 64, 192],
               emb_dim_list = [64, 192, 384],
               k_size = [7, 3, 3], 
               stride = [4, 2, 2], 
               padding = [2,0,0],
               multi_head = [1,3,6], 
               dim_expan = 4, 
               dropout = 0, 
               block_list = [1, 2, 10],
               ):
    
    super(CvTransformer, self).__init__()
    self.blk_list = block_list
    self.class_token = torch.nn.Parameter(torch.zeros(1, emb_dim_list[2]))

    self.conv_emb_1 = Conv_Token_Emb(in_channel[0], emb_dim_list[0], k_size[0], stride[0], padding[0])    
    self.stage_1 = torch.nn.ModuleList(
        [Cov_Trans_Block(emb_dim_list[0], multi_head = multi_head[0], dim_expan = dim_expan, dropout = dropout) for i in range(block_list[0])]
    )

    self.conv_emb_2 = Conv_Token_Emb(in_channel[1], emb_dim_list[1], k_size[1], stride[1], padding[1])    
    self.stage_2 = torch.nn.ModuleList(
        [Cov_Trans_Block(emb_dim_list[1], multi_head = multi_head[1], dim_expan = dim_expan, dropout = dropout) for i in range(block_list[1])]
    )

    self.conv_emb_3 = Conv_Token_Emb(in_channel[2], emb_dim_list[2], k_size[2], stride[2], padding[2])    
    self.stage_3 = torch.nn.ModuleList(
        [Cov_Trans_Block(emb_dim_list[2], multi_head = multi_head[2], dim_expan = dim_expan, dropout = dropout, final = True) for i in range(block_list[2])]
    )
    self.layer_norm = torch.nn.LayerNorm(emb_dim_list[2], eps = 1e-5)
    self.linear = torch.nn.Linear(emb_dim_list[2], class_num)

  def forward(self, x):
    out = self.conv_emb_1(x) #(b_s, c, h, w)
    h, w = out.shape[2], out.shape[3]
    out = out.permute(0,2,3,1) #(b_s, h, w, c)
    out = rearrange(out, 'b h w c -> b (h w) c') #(b_s, h*w, c)
    for i in range(self.blk_list[0]):
      out = self.stage_1[i](out, h, w)

    out = rearrange(out, 'b (h w) c -> b h w c', h = h, w = w).permute(0,3,1,2) # (b_s, c, h, w)
    out = self.conv_emb_2(out)
    h, w = out.shape[2], out.shape[3]
    out = out.permute(0,2,3,1)
    out = rearrange(out, 'b h w c -> b (h w ) c')
    for i in range(self.blk_list[1]):
      out = self.stage_2[i](out, h, w)

    out = rearrange(out, 'b (h w) c -> b h w c', h = h, w = w).permute(0,3,1,2)
    out = self.conv_emb_3(out)
    h, w = out.shape[2], out.shape[3]
    out = out.permute(0,2,3,1)
    out = rearrange(out, 'b h w c -> b (h w ) c')
    cls = self.class_token.repeat(x.shape[0],1,1)
    out = torch.cat([cls, out], dim = 1)
    for i in range(self.blk_list[2]):
      out = self.stage_3[i](out, h, w)
    output = self.linear(self.layer_norm(out[:,0,:]))
    return output
