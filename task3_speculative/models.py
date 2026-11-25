import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen2Config,Qwen2ForCausalLM

from common.utils import load_json,repo_path

REV_KEY = {"target":"model_target","draft":"model_draft"}
STUDENTS = ("off_6304","off_6305","on_6304","on_6305")
DRAFTS = ("original",)+STUDENTS


def device():
    """Pick the gpu when one is present"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def revision(cfg,which):
    """Pinned commit of the target or of the original draft"""
    return load_json(cfg["paths"]["lock"])["revisions"][REV_KEY[which]]


def load_tokenizer(cfg,which="target"):
    """Load a tokenizer at its pinned commit"""
    return AutoTokenizer.from_pretrained(cfg["models"][which],revision=revision(cfg,which))


def artifacts(cfg,path=None):
    """Folder that holds adapters and merged checkpoints"""
    return repo_path(path or cfg["paths"]["artifacts_dir"])


def tiny_model(cfg,which):
    """Small random stand in with the real vocabulary for smoke tests on the cpu"""
    torch.manual_seed(cfg["seed"]+(which != "target"))
    dim = 64 if which == "target" else 32
    config = Qwen2Config(vocab_size=151936,hidden_size=dim,intermediate_size=2*dim,num_hidden_layers=2,num_attention_heads=2,
                         num_key_value_heads=1,tie_word_embeddings=True,max_position_embeddings=2048,
                         attn_implementation=cfg["models"]["attn"])
    model = Qwen2ForCausalLM(config)
    # sharpening the random distributions so that the two stand ins disagree and rejections happen
    model.model.embed_tokens.weight.data *= 15
    return model.half()


def load_model(cfg,which,tiny=False,art=None):
    """Load the target the original draft or one merged student in half precision"""
    dtype,attn = getattr(torch,cfg["models"]["dtype"]),cfg["models"]["attn"]
    which = "draft" if which == "original" else which
    if which in STUDENTS:
        model = AutoModelForCausalLM.from_pretrained(artifacts(cfg,art)/"merged"/which,dtype=dtype,attn_implementation=attn)
    elif tiny:
        model = tiny_model(cfg,which)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["models"][which],revision=revision(cfg,which),dtype=dtype,
                                                     attn_implementation=attn)
    return model.to(device()).eval().requires_grad_(False)


def count_parameters(model):
    """Parameters with shared tensors counted once"""
    return sum({p.data_ptr():p.numel() for p in model.parameters()}.values())
