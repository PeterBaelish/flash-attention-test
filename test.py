import math
import random
import time
import os
from einops import rearrange
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

def custom_attention(q, k, v, causal=False):
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        mask = torch.triu(torch.ones(score.shape[-2], score.shape[-1]), diagonal=1)
        mask = mask.masked_fill(mask==1, torch.finfo(q.dtype).min)
        mask = mask.to(q.device, q.dtype)
        score = score + mask
    attn = F.softmax(score, dim=-1)
    o = torch.matmul(attn, v)
    return o

def pytorch_func(q, k, v, causal=False):
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

def flash_attention(q, k, v, causal=False):
    o = flash_attn_func(q, k, v, causal=causal)
    return o

def test(func_name, q, k, v, *args, **kwargs):
    if func_name in ["custom_attention", "pytorch_func"]:
        q = rearrange(q, "a b c d -> a c b d")
        k = rearrange(k, "a b c d -> a c b d")
        v = rearrange(v, "a b c d -> a c b d")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    for _ in range(5):
        o = globals()[func_name](q, k, v, *args, **kwargs)
    torch.cuda.synchronize()
    st = time.time()
    o = globals()[func_name](q, k, v, *args, **kwargs)
    torch.cuda.synchronize()
    tt = time.time() - st
    max_memory = torch.cuda.max_memory_allocated() // 2**20
    torch.cuda.empty_cache()
    print(o.size())
    if func_name in ["custom_attention", "pytorch_func"]:
        o = rearrange(o, "a c b d -> a b c d")

    return o, tt, max_memory

if __name__ == "__main__":
    test_num = 10
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    #torch.backends.cuda.enable_flash_sdp(False)
    #torch.backends.cuda.enable_mem_efficient_sdp(False)
    for idx in range(test_num):
        print(f"test {idx} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        bsz = random.randint(1, 64)
        sql = random.randint(1, 4096)
        nh = random.choice([8, 12, 16])
        hd = random.choice([64, 128])
        dtype = random.choice([torch.float16, torch.bfloat16])
        causal = random.choice([False, True])
        print(f"shape: ({bsz}, {sql}, {nh}, {hd}), dtype: {dtype}, causal: {causal}")
        q = torch.randn((bsz, sql, nh, hd)).to("cuda:0", dtype)
        k = torch.rand_like(q)
        v = torch.rand_like(q)

        o, t, m = test("custom_attention", q, k, v, causal=causal)
        print(f"custom pytorch time: {t:.6f}, peak memory: {m} MB")

        pf_o, pf_t, pf_m = test("pytorch_func", q, k, v, causal=causal)
        print(f"pytorch func time: {pf_t:.6f}, speedup: {t/pf_t:.2f}; peak memory: {pf_m} MB, save: {int((m-pf_m)/m*100)}%")
        assert torch.allclose(o, pf_o, rtol=1e-2, atol=1e-2)
        
        fa_o, fa_t, fa_m = test("flash_attention", q, k, v, causal=causal)
        print(f"flash attention time: {fa_t:.6f}, speedup: {t/fa_t:.2f}; peak memory: {fa_m} MB, save: {int((m-fa_m)/m*100)}%")
        assert torch.allclose(o, fa_o, rtol=1e-2, atol=1e-2)
