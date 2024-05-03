import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class Mlp(nn.Module):
    """ Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.activation = activation()
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


def window_partition(x, window_size):
    """
    Partition the input tensor into windows for localized multi-headed self attention.
    
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, Wd*Wh*Ww, C)
    """
    # Batch, Depth, Height, Width, Channel (channel remains untouched throughout the process)
    B, D, H, W, C = x.shape
    # Divide the three spatial dimensions (D, H, W) into windows
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    # Rearrange the data to be of shape (B, #windows_D, #windows_H, #windows_W, Wd, Wh, Ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
    # Align memory contiguously
    windows = windows.contiguous()
    # Merge the spatial dimensions into one dimension, so that the final result is (B*#windows, Wd*Wh*Ww, C)
    windows = windows.view(-1, reduce(mul, window_size), C)

    return windows


def undo_window_partition(windows, window_size, B, D, H, W):
    """
    Rearrange the windowed input back to an original input tensor shape.
   
    Args:
        windows: (B*num_windows, Wd*Wh*Ww, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    # Reshape flattened windowed dimensions into (B, #windows_D, #windows_H, #windows_W, Wd, Wh, Ww, C)
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    # Rearrange the data to be of shape (B, #windows_D, Wd, #windows_H, Wh, #windows_W, Ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)
    # Align memory contiguously 
    x = x.contiguous()
    # Merge #windows_{} and window_size_{} axes into one dimension, so that the final result is (B, D, H, W, C)
    x = x.view(B, D, H, W, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """
    Get the corrected window size and shift sizes for the input tensor, ensuring 
    that the window size is not larger than the input size in any dimension.

    Args:
        x_size (tuple[int]): Input tensor size
        window_size (tuple[int]): Window size (for localized multi-headed self attention)
        shift_size (tuple[int]): Shift size (for SWIN-like shifting windows)
    Returns:
        corrected_window_size (tuple[int]): Corrected window size
        corrected_shift_size (tuple[int]): Corrected shift size
    """
    # initialize the corrected window size and shift size
    corrected_window_size = list(window_size)

    if shift_size is not None:
        corrected_shift_size = list(shift_size)

    # Make sure the window size is not larger than the input size in any dimension
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            corrected_window_size[i] = x_size[i]

            if shift_size is not None:
                corrected_shift_size[i] = 0

    if shift_size is None:
        return tuple(corrected_window_size)
    else:
        return tuple(corrected_window_size), tuple(corrected_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.num_heads = num_heads  # nH
        head_dim = dim // num_heads  # C//nH
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of biases which we can index into
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # (2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH)

        # Get pair-wise relative position index for each token inside the window

        # Three lists of coordinates for each dimension individually
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # Meshgrid returns three tensors of shape (Wd, Wh, Ww), so we stack them together into a single tensor
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # (3, Wd, Wh, Ww)
        # Flatten the coordinates into a 1D tensor for each dimension
        coords_flattened = torch.flatten(coords, 1)  # (3, Wd*Wh*Ww)
        # Encode relative position of each token w.r.t. every other token within the window by subtracting two broadcasted tensors:
        # (3, Wd*Wh*Ww, 1) - (3, 1, Wd*Wh*Ww) --{broadcasting}--> (3, Wd*Wh*Ww, Wd*Wh*Ww) - (3, Wd*Wh*Ww, Wd*Wh*Ww)
        relative_coords = coords_flattened[:, :, np.newaxis] - coords_flattened[:, np.newaxis, :]  # (3, Wd*Wh*Ww, Wd*Wh*Ww) 
        # Permute the tensor and lay memory out contiguously
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wd*Wh*Ww, Wd*Wh*Ww, 3)
        # Add the window size along each dim to the relative coordinates on the same dim to ensure that they are non-negative
        # [-(Wd-1), ..., +(Wd-1)] --{shift}--> [0, ..., 2*Wd-2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1  # shift to start from 0
        relative_coords[:, :, 2] += self.window_size[2] - 1  # shift to start from 0

        # Convert the relative coordinates to unique indices

        # Changes in depth should shift indices by a full plane (i.e. the product of the height and width of the window)
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        # Changes in height should shift indices by a row (i.e. the width of the window)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        # Changes in width should shift indices by a column (i.e. 1)
        # DO NOTHING: relative_coords[:, :, 2] *= 1

        # Sum the coordinates of the 3 different dimensions together to get the final indices
        relative_position_indices = relative_coords.sum(-1)  # (Wd*Wh*Ww, Wd*Wh*Ww)

        # Register the relative position index as a buffer so that it is saved in the model's state_dict
        # This prevents relative position indices from treating as trainable model parameters
        self.register_buffer("relative_position_indices", relative_position_indices)

        # Layer to create query, key, and value matrices given an input feature vector
        # Using a single linear layer like this is an optimization to save memory
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Layer to project the output of the multi-head self attention back to the original dimension
        self.project = nn.Linear(dim, dim)

        # Initialize the relative position bias table with a truncated normal distribution
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # Softmax function to normalize the attention scores
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C), where N is the number of elements in each window.
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        # Total number of windows, number of elements in each window, and number of channels
        B_, N, C = x.shape
        # Generate the query, key, and value matrices
        qkv = self.qkv(x) # (B_, N, 3 * C)
        # Separate the query, key, and value matrices from each other and distribute them across the attention heads
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads) # (B_, N, 3, nH, C//nH)
        # Permute the tensor to make the query, key, and value matrices individually contiguous in memory
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B_, nH, N, C//nH)
        # Extract the query, key, and value matrices
        queries, keys, values = qkv[0], qkv[1], qkv[2]  # each is (B_, nH, N, C//nH)
        # Scale the query matrix
        queries = queries * self.scale

        # Calculate the attention scores
        attn = queries @ keys.transpose(-2, -1)

        # Look up the relative position bias table for each pair of tokens
        indices = self.relative_position_indices[:N, :N].reshape(-1)
        relative_position_bias = self.relative_position_bias_table[indices].reshape(N, N, -1)  # (Wd*Wh*Ww, Wd*Wh*Ww, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, Wd*Wh*Ww, Wd*Wh*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)  # (B_, nH, N, N)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ values).transpose(1, 2).reshape(B_, N, C)
        x = self.project(x)
        return x


class SepSTSBlock(nn.Module):
    """ A basic Sep-STS Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        depth_window_size (tuple[int]): Spatial attention window size.
        shift_size (tuple[int]): Shift size for spatial attention.
        point_window_size (tuple[int]): temporal attention window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        activation (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, num_heads, depth_window_size=(1, 8, 8), shift_size=(0, 0, 0),
                 point_window_size=(4, 1, 1), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 activation=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.depth_window_size = depth_window_size
        self.point_window_size = point_window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size[0] < self.depth_window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.depth_window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.depth_window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.depth_attn = WindowAttention3D(
            dim, window_size=self.depth_window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.point_attn = WindowAttention3D(
            dim, window_size=point_window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, activation=activation)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.depth_window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = torch.nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.depth_attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = undo_window_partition(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        ######################point attn###########################
        window_size = get_window_size((D, H, W), self.point_window_size)
        x_windows = window_partition(x, window_size)
        attn_windows = self.point_attn(x_windows, mask=None)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        x = window_partition(attn_windows, window_size, B, D, H, W)

        return x

    def forward_part2(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        x = self.forward_part1(x, mask_matrix)
        x = shortcut + x
        x = x + self.forward_part2(x)

        return x


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class SepSTSBasicLayer(nn.Module):
    """ A Sep-STS layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        depth_window_size (tuple[int]): spatial attention window size. Default: (1,8,8).
        temporal_window_size (tuple[int]): temporal attention window size. Default: (4,1,1).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 depth_window_size=(1, 8, 8),
                 point_window_size=(4, 1, 1),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.depth_window_size = depth_window_size
        self.shift_size = tuple(i // 2 for i in depth_window_size)
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SepSTSBlock(
                dim=dim,
                num_heads=num_heads,
                depth_window_size=depth_window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                point_window_size=point_window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.depth_window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for _, blk in enumerate(self.blocks):
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        x = rearrange(x, 'b d h w c -> b c d h w')
        return x
