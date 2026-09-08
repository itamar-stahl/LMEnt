"""RMU concept erasure for LMEnt models.

Grid from EMBER Appendix C.3, for Gemma-2-2B-it (26 layers) and
Llama-3.1-8B-Instruct (32 layers):
    lr        {1e-5, 1e-4, 3e-4}
    alpha     {10, 30, 50, 100}
    steering  {30, 100, 300, 1000}
    layers    Gemma (7,[5,6,7]) (8,[6,7,8]) (6,[4,5,6])
              Llama (7,[5,6,7]) (9,[7,8,9]) (11,[9,10,11])

Those absolute layer indices sit at 22-34% depth. The checkpoint this project
erases is OLMo-2 1B with **18 layers**, d_model 2048 and d_mlp 5632 -- read off
lment-1b-control-2e-b131k/config.json, not off the paper -- so the equivalent
band is (4,[2,3,4]), (5,[3,4,5]) and (6,[4,5,6]). Layers are picked from depth
automatically unless --layer-id/--layer-ids are given.

The published steering values were tuned for wider models. The --probe option
prints activation norms that help choose a useful scale for LMEnt.

Weights are fp32 by default, not bf16. Two reasons, both about this project.
The checkpoints on disk are fp32, and `compare_weights.py` checks that an
erased model differs from its control in the edited matrices *and nowhere
else* -- a bf16 save moves all 200 tensors and that check stops meaning
anything. And an AdamW step at lr 1e-4 on weights of magnitude ~2e-2 is a
relative change of ~5e-3, which is barely above bf16's 8-bit mantissa, so the
update arrives heavily quantised. Pass --dtype bf16 to get the old behaviour.

    python rmu.py --model <lment-1b-path> --concept <entity> \
        --concept-sentences chunks.json --neutral-sentences neutral.json --probe
    python rmu.py --model <lment-1b-path> --concept <entity> \
        --concept-sentences chunks.json --neutral-sentences neutral.json \
        --lr 1e-4 --alpha 100 --steering <from probe> --sanity
"""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_sentences(path, concept=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if concept is not None:
        for e in raw:
            if e.get("concept") == concept:
                return [s for s in e["sentences"] if s]
        raise KeyError(f"{concept!r} not in {path}")
    out = [o.get("sentence") if isinstance(o, dict) else o for o in raw]
    return [s for s in out if s]


def build_batches(forget, retain, batch_size, seed=42, max_len=2000):
    forget, retain = list(forget), list(retain)
    rnd = random.Random(seed)
    rnd.shuffle(forget)
    rnd.shuffle(retain)
    # characters, not tokens, same as the reference
    forget = [s for s in forget if len(s) <= max_len]
    retain = [s for s in retain if len(s) <= max_len]
    chop = lambda xs: [xs[i:i + batch_size] for i in range(0, len(xs), batch_size)
                       if xs[i:i + batch_size]]
    return chop(forget), chop(retain)


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_device(requested=None):
    if requested:
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(name, cache_dir=None, dtype=torch.float32, device=None):
    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    if tok.pad_token is None:
        # OLMo-2 ships one; this covers checkpoints that don't
        tok.pad_token = tok.eos_token
    # One device, not device_map="auto". RMU holds two copies of the model and
    # subtracts their activations; "auto" is free to shard the two copies
    # differently, and then the retain MSE gets its arguments on two devices.
    # A 1B in fp32 is ~4.4 GB, so both copies fit on any card in killable.
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map={"": resolve_device(device)}, dtype=dtype,
        cache_dir=cache_dir)
    if model.config.tie_word_embeddings:
        # EMBER only unties for gemma-2 by name. Keying on the config instead
        # covers OLMo-2, and matters once EMBER edits embeddings on top of this.
        with torch.no_grad():
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings = False
    model.config.use_cache = False
    return model, tok


def get_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList):
                return obj
        except AttributeError:
            continue
    raise AttributeError(f"no decoder layer list on {type(model).__name__}")


def down_proj_weights(model, layer_ids):
    # WMDP uses param_ids=[6], which lands on down_proj for Llama and Gemma-2
    # but on gate_proj for OLMo-2, where q_norm/k_norm shift the order.
    layers, out = get_layers(model), []
    for i in layer_ids:
        mlp = getattr(layers[i], "mlp", None) or layers[i].feed_forward
        for attr in ("down_proj", "c_proj", "dense_4h_to_h"):
            if hasattr(mlp, attr):
                out.append(getattr(mlp, attr).weight)
                break
        else:
            raise AttributeError(f"layer {i}: no down_proj in {list(dict(mlp.named_children()))}")
    return out


def param_index_report(model):
    names = [n for n, _ in get_layers(model)[0].named_parameters()]
    hits = [n for n in names if any(k in n for k in ("down_proj", "c_proj", "dense_4h_to_h"))]
    idx = names.index(hits[0]) if hits else None
    return {"arch": type(model).__name__, "down_proj_index": idx, "positional_ok": idx == 6}


def layers_by_depth(n_layers, fractions=(0.22, 0.28, 0.34)):
    """Paper's 22-34% depth band, mapped onto a model with n_layers."""
    out, seen = [], set()
    for f in fractions:
        lid = max(2, min(n_layers - 1, round(f * n_layers)))
        if lid not in seen:
            seen.add(lid)
            out.append((lid, [lid - 2, lid - 1, lid]))
    return out


def probe(model, tokenizer, batches, layer_id, max_length=512):
    """Mean L2 norm of residual activations at layer_id, to scale steering.

    Averaged over several batches: one batch of 16 sentences put the 1B's
    layer-5 norm anywhere in a ~10% band depending on which sentences it got,
    and the number is only useful as an order of magnitude for --steering.
    """
    total, count = 0.0, 0.0
    for batch in batches:
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(model.device)
        acts = forward_with_cache(model, inputs, get_layers(model)[layer_id])
        norms = acts.float().norm(dim=-1)
        mask = inputs["attention_mask"].to(norms.device, norms.dtype)
        total += (norms * mask).sum().item()
        count += mask.sum().item()
    return total / max(count, 1.0)


def forward_with_cache(model, inputs, module, no_grad=True):
    cache = []
    hook = module.register_forward_hook(
        lambda m, i, o: cache.append(o[0] if isinstance(o, tuple) else o))
    try:
        if no_grad:
            with torch.no_grad():
                model(**inputs)
        else:
            model(**inputs)
    finally:
        hook.remove()
    return cache[0]


def masked_mse(left, right, attention_mask):
    """MSE over real tokens only.

    The reference averages over the padded tensor, so the share of the loss
    spent on pad positions moves with how ragged each batch happens to be, and
    the forget term spends part of its gradient dragging pad activations onto
    the control vector. Both losses already carry the mask for the cosine
    diagnostics below. Accumulated in fp32 so --dtype bf16 does not lose the
    sum over ~8k positions.
    """
    mask = attention_mask.to(left.device).unsqueeze(-1).float()
    err = (left.float() - right.float()).pow(2) * mask
    return err.sum() / (mask.sum() * left.shape[-1]).clamp(min=1.0)


def masked_cosine(left, right, attention_mask):
    right = right.to(left.device)
    values = torch.nn.functional.cosine_similarity(
        left.float(), right.float(), dim=-1)
    mask = attention_mask.to(values.device, values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1)


def run_rmu(updated_model, frozen_model, tokenizer, forget_batches, retain_batches,
            layer_id=7, layer_ids=(5, 6, 7), alpha=100.0, steering=100.0, lr=1e-4,
            max_num_batches=150, max_length=512, seed=42, verbose=True):
    updated_model.train()
    frozen_model.eval()
    params = down_proj_weights(updated_model, layer_ids)
    for p in updated_model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr)

    up_mod = get_layers(updated_model)[layer_id]
    fr_mod = get_layers(frozen_model)[layer_id]

    # uniform, not gaussian, so the target lies in the positive orthant
    torch.manual_seed(seed)
    u = torch.rand(1, 1, updated_model.config.hidden_size,
                   dtype=updated_model.dtype, device=updated_model.device)
    control = u / torch.norm(u) * steering

    n = min(max_num_batches, len(forget_batches), len(retain_batches))
    if n == 0:
        raise ValueError("no batches")

    prev_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"
    trace = {"unlearn": [], "retain": [], "cos_forget": [], "cos_retain": []}
    try:
        for idx in range(n):
            f_in = tokenizer(forget_batches[idx], return_tensors="pt", padding=True,
                             truncation=True, max_length=max_length).to(updated_model.device)
            f_act = forward_with_cache(updated_model, f_in, up_mod, no_grad=False)
            control_here = control.to(f_act.device)
            # expand, not broadcast: mse_loss against a (1,1,d) target warns
            # about the size mismatch on every step and buries the log
            unlearn_loss = masked_mse(f_act, control_here.expand_as(f_act),
                                      f_in["attention_mask"])

            r_in = tokenizer(retain_batches[idx], return_tensors="pt", padding=True,
                             truncation=True, max_length=max_length).to(updated_model.device)
            r_act = forward_with_cache(updated_model, r_in, up_mod, no_grad=False)
            r_ref = forward_with_cache(frozen_model, r_in, fr_mod, no_grad=True)
            retain_loss = masked_mse(r_act, r_ref.to(r_act.device),
                                     r_in["attention_mask"]) * alpha

            loss = unlearn_loss + retain_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            trace["unlearn"].append(unlearn_loss.item())
            trace["retain"].append(retain_loss.item())
            with torch.no_grad():
                trace["cos_forget"].append(masked_cosine(
                    f_act, control_here, f_in["attention_mask"]).item())
                trace["cos_retain"].append(masked_cosine(
                    r_act, r_ref, r_in["attention_mask"]).item())
            if verbose and (idx % 5 == 0 or idx == n - 1):
                print(f"step {idx+1}/{n} loss={loss.item():.4g} "
                      f"unlearn={unlearn_loss.item():.4g} retain={retain_loss.item():.4g} "
                      f"cos_f={trace['cos_forget'][-1]:.3f} "
                      f"cos_r={trace['cos_retain'][-1]:.3f}")
    finally:
        tokenizer.truncation_side = prev_side
        updated_model.eval()
    return trace


SANITY_CHECKS = ("forget_rotated", "retain_preserved", "unlearn_loss_fell",
                 "spare_layer_frozen", "edited_layer_moved")


def sanity(trace, spare_before, spare_after, edited_before, edited_after):
    k = max(1, len(trace["cos_forget"]) // 5)
    start = sum(trace["cos_forget"][:k]) / k
    end = sum(trace["cos_forget"][-k:]) / k
    mean_r = sum(trace["cos_retain"]) / len(trace["cos_retain"])
    return {
        "cos_forget_start": round(start, 4),
        "cos_forget_end": round(end, 4),
        "forget_rotated": end > start + 0.05,
        "mean_cos_retain": round(mean_r, 4),
        "retain_preserved": mean_r > 0.9,
        "unlearn_loss_fell": trace["unlearn"][-1] < trace["unlearn"][0],
        "spare_layer_frozen": torch.equal(spare_before, spare_after),
        "edited_layer_moved": not torch.equal(edited_before, edited_after),
        # how far the edited matrix actually travelled, relative to its own
        # scale. Near zero with edited_layer_moved true means the optimiser
        # steps are being rounded away rather than applied -- the bf16 failure.
        "edited_rel_change": round(
            ((edited_after.float() - edited_before.float()).norm()
             / edited_before.float().norm().clamp(min=1e-12)).item(), 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--concept-sentences", required=True)
    ap.add_argument("--neutral-sentences", required=True)
    ap.add_argument("--out", default="runs/rmu")
    ap.add_argument("--cache-dir")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp32",
                    help="fp32 keeps the erased model comparable to its "
                         "control tensor-by-tensor; see the module docstring")
    ap.add_argument("--device", help="default cuda:0, or cpu without CUDA")
    ap.add_argument("--layer-id", type=int)
    ap.add_argument("--layer-ids", type=int, nargs="+")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--steering", type=float, default=100.0)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-num-batches", type=int, default=150)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe", action="store_true",
                    help="report layers and activation norms, then exit")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--save-model", action=argparse.BooleanOptionalAction,
                    default=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bs = a.batch_size or (8 if "llama" in a.model.lower() else 16)

    fb, rb = build_batches(load_sentences(a.concept_sentences, a.concept),
                           load_sentences(a.neutral_sentences), bs, a.seed)
    print(f"{sum(map(len, fb))} forget / {sum(map(len, rb))} retain, "
          f"{len(fb)} vs {len(rb)} batches")

    dtype, device = DTYPES[a.dtype], resolve_device(a.device)
    print(f"loading {a.model} as {a.dtype} on {device}")
    frozen, tok = load_model(a.model, a.cache_dir, dtype, device)
    print("params:", param_index_report(frozen))
    n_layers = len(get_layers(frozen))
    print(f"{n_layers} layers, d_model={frozen.config.hidden_size}, "
          f"depth-matched settings: {layers_by_depth(n_layers)}")

    if a.probe:
        report = {}
        for lid, lids in layers_by_depth(n_layers):
            norm = probe(frozen, tok, fb[:4], lid, a.max_length)
            report[lid] = {"layer_ids": lids, "mean_residual_norm": norm,
                           "steering_low": norm * 0.5, "steering_high": norm * 10}
            print(f"layer {lid}: mean residual norm {norm:.1f} "
                  f"-> try steering in [{norm*0.5:.0f}, {norm*10:.0f}]")
        (out / "probe.json").write_text(json.dumps(
            {"model": a.model, "n_layers": n_layers, "dtype": a.dtype,
             "layers": {str(k): v for k, v in report.items()}}, indent=2))
        print("wrote", out / "probe.json")
        return

    if (a.layer_id is None) != (a.layer_ids is None):
        raise SystemExit("provide --layer-id and --layer-ids together, or omit both")
    if a.layer_id is None:
        # default to the middle of the paper's 22-34% depth band for this model
        candidates = layers_by_depth(n_layers)
        a.layer_id, a.layer_ids = candidates[len(candidates) // 2]
        print(f"using layer_id={a.layer_id} layer_ids={a.layer_ids}")

    updated, _ = load_model(a.model, a.cache_dir, dtype, device)
    spare = next(i for i in range(n_layers) if i not in a.layer_ids)
    spare_before = down_proj_weights(updated, [spare])[0].detach().clone()
    edited_before = down_proj_weights(updated, [a.layer_ids[0]])[0].detach().clone()

    trace = run_rmu(updated, frozen, tok, fb, rb, a.layer_id, a.layer_ids, a.alpha,
                    a.steering, a.lr, a.max_num_batches, a.max_length, a.seed)

    (out / "trace.json").write_text(json.dumps(trace, indent=2))
    (out / "config.json").write_text(json.dumps(vars(a), indent=2, default=str))
    failed = []
    if a.sanity:
        res = sanity(trace, spare_before, down_proj_weights(updated, [spare])[0].detach(),
                     edited_before, down_proj_weights(updated, [a.layer_ids[0]])[0].detach())
        failed = [k for k in SANITY_CHECKS if not res[k]]
        res["failed"] = failed
        (out / "sanity.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        if failed:
            print("FAILED:", failed)
    if a.save_model:
        updated.save_pretrained(out / "model")
        tok.save_pretrained(out / "model")
    # A run that produced a model but moved nothing is the failure mode this
    # project keeps re-learning, so it must not exit 0 and look like a success.
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
