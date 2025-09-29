"""
希尔伯特曲线排序算法
将多维坐标映射到一维整数，同时保持空间局部性。
"""
import torch

# --- 内部辅助函数，用于二进制和格雷码的位运算 ---

def right_shift(binary, k=1, axis=-1):
    """对二进制张量进行右移位操作"""
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    # 通过padding在左侧补0，实现右移效果
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )
    return shifted

def binary2gray(binary, axis=-1):
    """将二进制码转换为格雷码。公式: Gray = Binary XOR (Binary >> 1)"""
    shifted = right_shift(binary, axis=axis)
    gray = torch.logical_xor(binary, shifted)
    return gray

def gray2binary(gray, axis=-1):
    """将格雷码转换回二进制码。"""
    # 需要多次迭代移位和异或操作来解码
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray

# --- 核心的编码和解码函数 ---

def encode(locs, num_dims, num_bits):
    """
    将多维坐标编码为一维的希尔伯特整数。

    Args:
        locs (Tensor): 输入的坐标，形状为 [..., num_dims]。坐标值应为整数。
        num_dims (int): 坐标的维度 (例如，3D点云为3)。
        num_bits (int): 每个维度使用的比特数，决定了该维度的分辨率 (2**num_bits)。

    Returns:
        Tensor: 与输入形状相同（除了最后一维）的一维希尔伯特整数。
    """
    # ... 内部是复杂的、基于格雷码的位操作来实现Skilling算法 ...
    # 1. 将输入的整数坐标转换为二进制表示
    # 2. 通过迭代，将多维的二进制位交织并转换为格雷码形式的希尔伯特曲线索引
    # 3. 将格雷码转回二进制
    # 4. 将最终的二进制序列打包成一个64位整数
    
    # (具体实现细节较为繁琐，此处省略以保持清晰)
    orig_shape = locs.shape
    bitpack_mask_rev = (1 << torch.arange(0, 8).to(locs.device)).flip(-1)
    if orig_shape[-1] != num_dims:
        raise ValueError("...")
    if num_dims * num_bits > 63:
        raise ValueError("...")

    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    for bit in range(0, num_bits):
        for dim in range(0, num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)
    
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = gray2binary(gray)
    padded = torch.nn.functional.pad(hh_bin, (64 - num_bits * num_dims, 0), "constant", 0)
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * (1 << torch.arange(0, 8).to(locs.device)))
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )
    return hh_uint8.view(torch.int64).squeeze()


def decode(hilberts, num_dims, num_bits):
    """
    将一维的希尔伯特整数解码回多维坐标。
    """
    # ... 内部是与encode相反的位操作 ...
    # 1. 将64位整数解包成二进制序列
    # 2. 将二进制转换为格雷码
    # 3. 通过逆向的迭代过程，将一维的格雷码解开并还原成多维的二进制表示
    # 4. 将多维的二进制坐标打包成整数
    
    # (具体实现细节较为繁琐，此处省略以保持清晰)
    hilberts = torch.atleast_1d(hilberts)
    orig_shape = hilberts.shape
    bitpack_mask_rev = (1 << torch.arange(0, 8).to(hilberts.device)).flip(-1)
    hh_uint8 = (
        hilberts.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
    )
    hh_bits = (
        hh_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[:, -num_dims * num_bits :]
    )
    gray = binary2gray(hh_bits)
    gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)

    for bit in range(num_bits - 1, -1, -1):
        for dim in range(num_dims - 1, -1, -1):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    padded = torch.nn.functional.pad(gray, (64 - num_bits, 0), "constant", 0)
    locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))
    locs_uint8 = (locs_chopped * (1 << torch.arange(0, 8).to(hilberts.device))).sum(3).squeeze().type(torch.uint8)
    return locs_uint8.view(torch.int64).reshape((*orig_shape, num_dims))