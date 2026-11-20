import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen2Config,Qwen2ForCausalLM

from common.utils import load_json

ATTN = ("q_proj","k_proj","v_proj","o_proj")


def device():
    """Pick the gpu when one is present"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def revision(cfg):
    """Read the pinned model commit from the source revision lock"""
    return load_json(cfg["paths"]["lock"])["revisions"]["model_base"]


def load_tokenizer(cfg):
    """Load the tokenizer at the pinned commit"""
    return AutoTokenizer.from_pretrained(cfg["model"]["name"],revision=revision(cfg))


def load_model(cfg,tiny=False,dtype=None):
    """Load a fresh unchanged copy of the model or a tiny random one for smoke tests on the cpu"""
    if tiny:
        torch.manual_seed(cfg["seed"])
        config = Qwen2Config(vocab_size=151936,hidden_size=128,intermediate_size=256,num_hidden_layers=2,
                             num_attention_heads=2,num_key_value_heads=1,tie_word_embeddings=True,
                             max_position_embeddings=4096,attn_implementation=cfg["model"]["attn"])
        return Qwen2ForCausalLM(config).to(dtype or torch.float32).to(device()).eval().requires_grad_(False)
    dtype = dtype or getattr(torch,cfg["model"]["dtype"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"],revision=revision(cfg),dtype=dtype,
                                                 attn_implementation=cfg["model"]["attn"])
    return model.to(device()).eval().requires_grad_(False)


def projection(layer,name):
    """Return one of the seven projections of a block"""
    return getattr(layer.self_attn if name in ATTN else layer.mlp,name)


def eligible(model,cfg):
    """List every compressible projection as block index name and module"""
    return [(i,name,projection(layer,name)) for i,layer in enumerate(model.model.layers) for name in cfg["eligible"]]


def unique_parameters(model):
    """Count parameters with shared tensors counted once"""
    seen = {}
    for p in model.parameters():
        seen[p.data_ptr()] = p.numel()
    return sum(seen.values())


def fingerprint(model,cfg):
    """Summarize every eligible weight by two exact double precision sums"""
    out = {}
    for i,name,m in eligible(model,cfg):
        w = m.weight.detach().double()
        out[f"{i}.{name}"] = [w.sum().item(),w.abs().sum().item()]
    return out


def embed(model,ids):
    """Turn token ids into the first block input and the rotary tables"""
    hidden = model.model.embed_tokens(ids)
    positions = torch.arange(ids.shape[1],device=ids.device).unsqueeze(0)
    return hidden,model.model.rotary_emb(hidden,positions)


def run_blocks(model,hidden,rope,start=0):
    """Run the hidden states through every block from start to the end"""
    for layer in model.model.layers[start:]:
        hidden = layer(hidden,position_embeddings=rope)
    return hidden


def sequence_nll(model,hidden,ids):
    """Return the summed next token nll of each sequence from its last hidden states"""
    out = []
    for j in range(hidden.shape[0]):
        logits = model.lm_head(model.model.norm(hidden[j:j+1])).float()
        out.append(F.cross_entropy(logits[0,:-1],ids[j,1:],reduction="sum"))
    return torch.stack(out)


@torch.no_grad()
def nll_from_block(model,inputs,rope,ids,start,micro):
    """Pooled nll of the calibration sequences when the stored inputs of one block are run to the end"""
    total = 0.0
    for a in range(0,inputs.shape[0],micro):
        hidden = run_blocks(model,inputs[a:a+micro],rope,start)
        total += sequence_nll(model,hidden,ids[a:a+micro]).sum().item()
    return total/(ids.shape[0]*(ids.shape[1]-1))
