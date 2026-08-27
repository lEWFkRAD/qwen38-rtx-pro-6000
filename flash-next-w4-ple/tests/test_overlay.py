import torch
from types import SimpleNamespace

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.models.qwen4_exp import Qwen4ExpPackedPinnedHostEmbedding

rows = 128
dim = 160
fake = SimpleNamespace(
    quant_method=UnquantizedEmbeddingMethod(),
    weight=torch.nn.Parameter(torch.empty((rows,dim),device="cuda",dtype=torch.float8_e4m3fn),requires_grad=False),
    weight_scale=torch.tensor([0.00019931793212890625],device="cuda",dtype=torch.bfloat16),
    quant_config=None,
    enable_tp=False,
    use_attn_tp_group=False,
    tp_size=1,
    num_embeddings=rows,
    num_embeddings_padded=rows,
    org_vocab_size=rows,
    padding_size=0,
    num_added_embeddings=0,
    use_presharded_weights=False,
    org_vocab_size_padded=rows,
    shard_indices=SimpleNamespace(org_vocab_start_index=0,org_vocab_end_index=rows),
    embedding_dim=dim,
    num_embeddings_per_partition=rows,
    num_org_embeddings_per_partition=rows,
    num_added_embeddings_per_partition=0,
)
emb = Qwen4ExpPackedPinnedHostEmbedding(fake)
torch.manual_seed(7)
src = (torch.randn(rows, dim) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
emb.load_fp8_rows(src, 0)
ids = torch.tensor([0, 1, 7, 31, 63, 127], device="cuda", dtype=torch.int64)
got = emb.gather(ids).cpu()
x = src[ids.cpu()].float().reshape(-1, 10, 16)
maxp = x.clamp_min(0).amax(-1) / 7
maxn = (-x.clamp_max(0)).amax(-1) / 8
s = torch.maximum(maxp, maxn).clamp_min(2**-9).to(torch.float8_e4m3fn).float()
q = torch.round(x / s.unsqueeze(-1)).clamp(-8, 7)
ref = (q * s.unsqueeze(-1)).reshape(-1, dim).to(torch.bfloat16)
equal = torch.equal(got, ref)
max_abs = float((got.float() - ref.float()).abs().max())
cosine = float(
    torch.nn.functional.cosine_similarity(
        src[ids.cpu()].float(), got.float(), dim=1
    ).mean()
)
assert equal, f"overlay parity failed: max_abs={max_abs} cosine={cosine}"
print("OVERLAY_CLASS_OK", equal, "max_abs", max_abs)
print("cosine", cosine)
