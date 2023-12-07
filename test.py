import math
import random
import time
import os
from einops import rearrange
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
import numpy as np

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")

#set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

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

#def pytorch_func(q, k, v, causal=False):
#    return F._scaled_dot_product_attention(q, k, v, is_caual=causal)

def flash_attention(q, k, v, causal=False):
    o = flash_attn_func(q, k, v, causal=causal)
    return o

def test(func_name, q, k, v, *args, **kwargs):
    if func_name in ["custom_attention"]:
        q = rearrange(q, "a b c d -> a c b d")
        k = rearrange(k, "a b c d -> a c b d")
        v = rearrange(v, "a b c d -> a c b d")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    #device_0 = torch.device("cuda:3")
    #device_1 = torch.device("cuda:4")
    
    #split_q = torch.chunk(q, 2, dim=3)
    #split_k = torch.chunk(k, 2, dim=3)
    #split_v = torch.chunk(v, 2, dim=3)

    #split_q0 = split_q[0].to(device_0)
    #split_q1 = split_q[1].to(device_1)
    #split_k0 = split_k[0].to(device_0)
    #split_k1 = split_k[1].to(device_1)
    #split_v0 = split_v[0].to(device_0)
    #split_v1 = split_v[1].to(device_1)
    
    for _ in range(5):
        o = globals()[func_name](q, k, v, *args, **kwargs)
        #split_o0 = globals()[func_name](split_q0, split_k0, split_v0, *args, **kwargs)
        #split_o1 = globals()[func_name](split_q1, split_k1, split_v1, *args, **kwargs)
        #split_o1 = split_o1.to(device_0)
        #o = torch.cat([split_o0, split_o1], dim=3)

    torch.cuda.synchronize()
    st = time.time() 
    
    o = globals()[func_name](q, k, v, *args, **kwargs)
    
    #split_o0 = globals()[func_name](split_q0, split_k0, split_v0, *args, **kwargs)
    #split_o1 = globals()[func_name](split_q1, split_k1, split_v1, *args, **kwargs)
    
    #split_o0 = split_o0.to("cpu")
    #split_o1 = split_o1.to("cpu")

    #o = torch.cat([split_o0, split_o1], dim=3)

    torch.cuda.synchronize()
    tt = time.time() - st
    max_memory = torch.cuda.max_memory_allocated() // 2**20
    torch.cuda.empty_cache()
    print(o.size())
    if func_name in ["custom_attention"]:
        o = rearrange(o, "a c b d -> a b c d")

    return o, tt, max_memory

if __name__ == "__main__":
    test_num = 10
    bl_t = 0
    my_t = 0
    bl_array = []
    my_array = []
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    #torch.backends.cuda.enable_flash_sdp(False)
    #torch.backends.cuda.enable_mem_efficient_sdp(False)
    for idx in range(test_num):
        print(f"test {idx} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        bsz = random.randint(1, 8)
        bsz = 4
        sql = random.randint(384, 1024)
        sql = 4096
        nh = random.choice([8, 12, 16])
        nh = 16
        hd = random.choice([64, 128])
        hd = 128
        #if bsz * nh * math.ceil(sql/128) > 82: #3090 number of SM
        #    print("too large")
        #    continue
        #if bsz * sql * nh * hd >= 23*1024*1024: #3090 RAM size
        #    print("too large")
        #    continue
        dtype = random.choice([torch.float16, torch.bfloat16])
        causal = random.choice([False, True])
        if idx%2==1: 
            causal = True
        else:
            causal = False
        print(f"shape: ({bsz}, {sql}, {nh}, {hd}), dtype: {dtype}, causal: {causal}")
        #device = torch.device("cuda:3")
        q = torch.randn((bsz, sql, nh, hd)).to("cuda:2", dtype)
        #k = torch.full([bsz, sql, nh, hd], 1.0).to("cuda:7", dtype)
        #v = torch.full_like(q, 0)
        #k = torch.full_like(q, 1)
        k = torch.randn((bsz, sql, nh, hd)).to("cuda:2", dtype)
        v = torch.rand_like(k)
        #for i in range(hd):
        #    for j in range(sql):
        #        v[0][0][0][i] = 100
        #        k[0][j][0][i] = 1/(i+j+1)       
        #            if j%2 == 0:
        #            v[0][j][0][i] = 2
        o, t, m = test("custom_attention", q, k, v, causal=True)
        #print(f"custom pytorch time: {t:.6f}, peak memory: {m} MB")
        #print(v)
        #pf_o, pf_t, pf_m = test("pytorch_func", q, k, v, causal=causal)
        #print(f"pytorch func time: {pf_t:.6f}, speedup: {t/pf_t:.2f}; peak memory: {pf_m} MB, save: {int((m-pf_m)/m*100)}%")
        #assert torch.allclose(o, pf_o, rtol=1e-2, atol=1e-2)
        
        fa_o, fa_t, fa_m = test("flash_attention", q, k, v, causal=causal)
        print(f"flash attention time: {fa_t:.6f}, peak memory: {fa_m} MB")#, save: {int((m-fa_m)/m*100)}%")
        #assert torch.allclose(o, fa_o, rtol=1e-2, atol=1e-2)
        for i in range(sql):
            if i%sql == sql - 1:
                print(i)
                print(torch.allclose(o[0][i][0],fa_o[0][i][0],rtol=0.01,atol=0.01))
                print(o[0][i][0])
                print(fa_o[0][i][0])
        #print(o)
        #print(fa_o)
        print(torch.allclose(o, fa_o, rtol=0.05, atol=0.05))
        if idx%2 == 1:
            my_t += fa_t
            my_array.append(fa_t)
        else:
            bl_t += fa_t
            bl_array.append(fa_t)
    print("bl_arr:")
    bl_array = ['{:.6f}'.format(x) for x in bl_array]
    print(bl_array)
    print("my_arr:")
    my_array = ['{:.6f}'.format(x) for x in my_array]
    print(my_array)
    print(f"avg base time: {bl_t/(test_num/2):.6f}, avg my time: {my_t/(test_num/2):.6f}, My speedup: {bl_t/my_t:.2f}")
