"""Sparse Semi-NMF erasure for MLP layers.

The factorization uses the Semi-NMF updates from Ding, Li and Jordan. Feature
selection uses the ratio pre-filter and the two-stage judge in Appendix A.3.

    python snmf.py factorize --model <path> --concept <name> \
        --concept-sentences chunks.json --neutral-sentences neutral.json \
        --out snmf_out
    python snmf.py select --out snmf_out --concept <name>
    python snmf.py erase --model <path> --out snmf_out \
        --delta-in 4 --delta-out 4 --save erased_model
    python snmf.py verify --model <path> --concept <name> --out snmf_out \
        --concept-sentences chunks.json --neutral-sentences neutral.json

Three things differ from the published recipe, all because of what this project
actually has.

**The judge is local.** A.3 runs two Gemini calls per feature and there is no
Gemini key on this cluster, which left `select` with only `--skip-llm` -- and
the note on that flag says plainly that it stops matching A.3. The EMBER fork
on this branch already solved the same problem for *embedding* features with a
pinned local `google/gemma-4-12B-it`, run in its own interpreter because gemma4
needs newer transformers than this pipeline. `--judge gemma` (the default) is
that same judge behind the same two seams, so MLP and embedding features are
interpreted by the same model. `--judge gemini` still works if a key appears.

**Layer ranges come from depth, not from Gemma.** The published defaults are
absolute indices for a 26-layer Gemma: in=(0,25), every layer, and out=(0,8),
the first third. Left as literals on the 18-layer LMEnt 1B they would skip
layers 16-17 on the in side and cover half the model on the out side, so
--layers-in/--layers-out now default to the same *fractions* of depth.

**Weights are fp32.** The checkpoints on disk are fp32 and
`compare_weights.py` checks that an erased model differs from its control in
the edited matrices and nowhere else; a bf16 save moves all 200 tensors and
that check stops meaning anything.
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_device(requested=None):
    if requested:
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(name, cache_dir=None, dtype=torch.float32, device=None):
    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # One device, not device_map="auto": ablate_layer edits up_proj and
    # down_proj against a feature direction moved onto W_in's device, and a
    # sharded model puts the two matrices of one MLP wherever it likes.
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map={"": resolve_device(device)}, dtype=dtype,
        cache_dir=cache_dir)
    model.config.use_cache = False
    model.eval()
    return model, tok


def get_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList):
                return obj
        except AttributeError:
            continue
    raise AttributeError(f"no decoder layer list on {type(model).__name__}")


def mlp_of(layer):
    return getattr(layer, "mlp", None) or layer.feed_forward


def default_rank(hidden_size):
    """Use k=100 below width 4096 and k=200 at width 4096 or above."""
    return 200 if hidden_size >= 4096 else 100


def default_layer_ranges(n_layers):
    """The publication's layer ranges, as fractions of depth.

    Gemma-2-2B has 26 layers and ships in=(0,25) and out=(0,8): all of it, and
    its first third. Reproducing those two fractions on any depth gives the
    26-layer numbers back exactly (round(0.34*26)-1 == 8) and does the right
    thing on the 18-layer LMEnt 1B, where the literals would have been wrong
    in both directions.
    """
    last = n_layers - 1
    return (0, last), (0, max(0, min(last, round(0.34 * n_layers) - 1)))


@torch.no_grad()
def collect_activations(model, tokenizer, concept_sents, neutral_sents,
                        layers, batch_size=8, max_length=256):
    """Post-nonlinearity MLP activations, i.e. the input to down_proj.

    Returns acts[layer] of shape (d_mlp, n_tok) and a bool mask marking which
    of those tokens came from S_C.
    """
    store = {l: [] for l in layers}
    labels, token_ids = [], []

    hooks = []
    for l in layers:
        def make(l):
            def hook(_m, inputs, _out):
                store[l].append(inputs[0].detach())   # down_proj input
            return hook
        hooks.append(mlp_of(get_layers(model)[l]).down_proj.register_forward_hook(make(l)))

    try:
        for sents, is_concept in ((concept_sents, True), (neutral_sents, False)):
            for i in range(0, len(sents), batch_size):
                batch = sents[i:i + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length).to(model.device)
                model(**enc)
                keep = enc["attention_mask"].bool().reshape(-1)   # drop pad positions
                labels.append(torch.full((int(keep.sum()),), is_concept, dtype=torch.bool))
                token_ids.append(enc["input_ids"].reshape(-1)[keep].cpu())
                for l in layers:
                    a = store[l][-1]
                    store[l][-1] = a.reshape(-1, a.shape[-1])[keep].float().cpu()
    finally:
        for h in hooks:
            h.remove()

    acts = {l: torch.cat(store[l], 0).T.contiguous() for l in layers}   # (d_mlp, n_tok)
    return acts, torch.cat(labels), torch.cat(token_ids)


def _wta_columns(Z, s):
    """Keep the ceil(s * d_mlp) largest-magnitude entries of each column."""
    keep = max(1, math.ceil(s * Z.shape[0]))
    if keep >= Z.shape[0]:
        return Z
    thresh = Z.abs().topk(keep, dim=0).values[-1:, :]
    return Z * (Z.abs() >= thresh)


def semi_nmf(A, k, s=0.01, lam=1e-4, max_iter=20000, patience=500, tol=1e-4,
             seed=42, device="cuda", verbose=True):
    """A ~ Z Y, Z unconstrained and column-sparse, Y >= 0. Returns (Z, Y)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    d, n = A.shape
    A = A.to(device, torch.float32)
    Z = torch.randn(d, k, generator=g).to(device)
    Yt = torch.rand(k, n, generator=g).to(device).T.contiguous()   # (n, k), strictly > 0
    eye = torch.eye(k, device=device)

    best, since = float("inf"), 0
    best_Z = best_Yt = None
    for it in range(max_iter):
        Y = Yt.T
        Z = A @ Y.T @ torch.linalg.inv(Y @ Y.T + lam * eye)
        Z = _wta_columns(Z, s)

        # multiplicative update, keeps Y non-negative at every step
        AtZ = A.T @ Z
        ZtZ = Z.T @ Z
        num = AtZ.clamp(min=0) + Yt @ ZtZ.clamp(max=0).neg()
        den = AtZ.clamp(max=0).neg() + Yt @ ZtZ.clamp(min=0)
        Yt = Yt * torch.sqrt((num + 1e-12) / (den + 1e-12))

        # rescale rows of Y to unit norm, absorb the scale into Z's columns
        norms = Yt.norm(dim=0).clamp(min=1e-8)
        Yt = Yt / norms
        Z = Z * norms

        err = (A - Z @ Yt.T).pow(2).sum().item()
        if err < best - tol:
            best, since = err, 0
            best_Z, best_Yt = Z.clone(), Yt.clone()
        else:
            since += 1
            if since >= patience:
                break
        if verbose and it % 500 == 0:
            print(f"  iter {it} recon {err:.4g}")
    if best_Z is None:
        # only happens if the reconstruction went non-finite on step 0
        raise RuntimeError("semi_nmf produced no finite reconstruction; "
                           "check the activations for NaN or inf")
    if verbose:
        print(f"  stopped at iter {it}, recon {best:.4g}")
    # return the best iterate, not the last one
    return best_Z.cpu(), best_Yt.T.cpu()


def rho_summary(rho, tau):
    """Where the mass ratios actually sit, so tau can be read off evidence.

    EMBER's embedding run on this same 1B died at its prefilter because the
    published ratio_thresh 2.0 sits *above* the maximum ratio a 1B produces --
    discovered only after the run. The same threshold gates this method, so
    every factorization records its own distribution whether or not anything
    clears tau.
    """
    values = sorted(float(v) for v in rho.tolist())
    n = len(values)
    pct = lambda q: values[min(n - 1, max(0, int(round(q * (n - 1)))))]
    return {
        "n_features": n, "tau": tau,
        "max": pct(1.0), "p99": pct(0.99), "p95": pct(0.95),
        "median": pct(0.5), "min": pct(0.0),
        "n_ge_tau": sum(v >= tau for v in values),
        "n_ge": {str(t): sum(v >= t for v in values)
                 for t in (1.1, 1.25, 1.5, 2.0, 3.0)},
    }


def mass_ratio(Y, is_concept, eps=1e-8):
    """rho_i, Eq. 3. Y is (k, n_tok); is_concept marks the S_C columns."""
    c = Y[:, is_concept].abs().mean(dim=1)
    n = Y[:, ~is_concept].abs().mean(dim=1)
    return c / (n + eps)


def top_tokens_per_feature(Y, token_strings, top=20):
    """Highest-coefficient tokens per feature, for the LLM interpretation step."""
    out = {}
    for i in range(Y.shape[0]):
        idx = Y[i].abs().topk(min(top, Y.shape[1])).indices.tolist()
        out[i] = [token_strings[j] for j in idx]
    return out


@torch.no_grad()
def projection_tokens_per_feature(model, tokenizer, layer_id, Z, feature_ids, top=30):
    """Top vocabulary tokens of U ln_final(W_out^T z), Appendix A.3."""
    mlp = mlp_of(get_layers(model)[layer_id])
    down = mlp.down_proj.weight.detach()             # (d_model, d_mlp)
    unembed = model.get_output_embeddings().weight.detach()  # (vocab, d_model)
    final_norm = getattr(getattr(model, "model", None), "norm", None)
    out = {}
    if not feature_ids:
        return out
    # 100,352 x 2,048 in fp32 is 822 MB; casting it once instead of once per
    # 16-feature chunk is the difference between one copy and dozens.
    down_f = down.float()
    unembed_f = unembed.float()
    for start in range(0, len(feature_ids), 16):
        ids = feature_ids[start:start + 16]
        z = Z[:, ids].T.to(down_f.device, torch.float32)
        directions = z @ down_f.T
        if final_norm is not None:
            norm_device = next(final_norm.parameters()).device
            directions = final_norm(directions.to(norm_device, down.dtype)).float()
        logits = directions.to(unembed_f.device) @ unembed_f.T
        token_ids = logits.topk(min(top, logits.shape[-1]), dim=-1).indices.cpu()
        for fid, row in zip(ids, token_ids):
            out[fid] = tokenizer.convert_ids_to_tokens(row.tolist())
    return out


def coverage_mask(Z, feature_ids, gamma=0.95):
    """Per-layer neuron filter, Eqs. 19-20.

    m_j = sum over concept features of |Z[j,i]| / ||z_i restricted to support||.
    Keep the smallest greedy set of neurons covering gamma of the total mass.
    """
    m = torch.zeros(Z.shape[0])
    for i in feature_ids:
        z = Z[:, i]
        nz = z[z != 0]
        if nz.numel() == 0:
            continue
        m += z.abs() / nz.norm().clamp(min=1e-8)
    order = m.argsort(descending=True)
    csum = m[order].cumsum(0)
    n_keep = int((csum < gamma * m.sum()).sum().item()) + 1
    keep = torch.zeros(Z.shape[0], dtype=torch.bool)
    keep[order[:n_keep]] = True
    return keep


@torch.no_grad()
def ablate_layer(model, layer_id, Z, feature_ids, delta_in, delta_out, gamma=0.95):
    """Project the concept directions out of up_proj and down_proj."""
    mlp = mlp_of(get_layers(model)[layer_id])
    # HF stores these transposed relative to the paper, where W_in is (d, d_mlp)
    # and W_out is (d_mlp, d). .T is a view, so the -= below edit in place.
    W_in = mlp.up_proj.weight.data.T
    W_out = mlp.down_proj.weight.data.T
    dev, dt = W_in.device, W_in.dtype

    keep = coverage_mask(Z, feature_ids, gamma).to(dev)
    # Directions come from the unedited matrices, before any update lands.
    prepared = []
    for i in feature_ids:
        z = Z[:, i].to(dev, dt)
        support = (z != 0) & keep
        if not support.any():
            continue
        f_in = W_in @ z if delta_in else None
        f_out = W_out.T @ z if delta_out else None
        if f_in is not None:
            f_in = f_in / f_in.norm().clamp(min=1e-8)
        if f_out is not None:
            f_out = f_out / f_out.norm().clamp(min=1e-8)
        prepared.append((support, f_in, f_out))

    n_edited = 0
    for support, f_in, f_out in prepared:
        n_edited += 1

        if f_out is not None:
            coeff = W_out @ f_out                       # (d_mlp,)
            upd = torch.outer(coeff * support.to(dt), f_out)
            W_out -= delta_out * upd

        if f_in is not None:
            coeff = f_in @ W_in                         # (d_mlp,)
            upd = torch.outer(f_in, coeff * support.to(dt))
            W_in -= delta_in * upd

    return n_edited, int(keep.sum())


def resolve_ranges(n_layers, layers_in, layers_out):
    """CLI ranges if given, else the depth-scaled publication defaults."""
    d_in, d_out = default_layer_ranges(n_layers)
    lo_in, hi_in = tuple(layers_in) if layers_in else d_in
    lo_out, hi_out = tuple(layers_out) if layers_out else d_out
    print(f"{n_layers} layers -> in [{lo_in},{hi_in}] out [{lo_out},{hi_out}]"
          f"{'' if layers_in and layers_out else ' (from model depth)'}")
    return (lo_in, hi_in), (lo_out, hi_out)


def load_sentences(path, concept=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if concept is not None:
        for e in raw:
            if e.get("concept") == concept:
                return [s for s in e["sentences"] if s]
        raise KeyError(f"{concept!r} not in {path}")
    return [s for s in (o.get("sentence") if isinstance(o, dict) else o for o in raw) if s]


def cmd_factorize(a):
    if a.layer_batch_size < 1:
        raise SystemExit("--layer-batch-size must be at least 1")
    dtype, device = DTYPES[a.dtype], resolve_device(a.device)
    model, tok = load_model(a.model, a.cache_dir, dtype, device)
    n_layers = len(get_layers(model))
    layers = list(range(n_layers)) if a.layers is None else a.layers
    k = a.k or default_rank(model.config.hidden_size)
    d_mlp = model.config.intermediate_size
    print(f"{n_layers} layers, d_model={model.config.hidden_size}, "
          f"d_mlp={d_mlp}, k={k}, {a.dtype} on {device}")

    concept_all = load_sentences(a.concept_sentences, a.concept)
    neutral_all = load_sentences(a.neutral_sentences)
    if not a.allow_short_data and (
            len(concept_all) < a.n_sentences or len(neutral_all) < a.n_sentences):
        raise SystemExit(
            f"expected {a.n_sentences} sentences per side; "
            f"found {len(concept_all)} concept and {len(neutral_all)} neutral. "
            "Pass --allow-short-data to continue with the available data.")
    concept = concept_all[:a.n_sentences]
    neutral = neutral_all[:a.n_sentences]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results, feature_tokens, projection_tokens, rho_stats = {}, {}, {}, {}
    # A few layers at a time; holding all of them at once eats host RAM.
    for start in range(0, len(layers), a.layer_batch_size):
        layer_group = layers[start:start + a.layer_batch_size]
        acts, is_concept, token_ids = collect_activations(
            model, tok, concept, neutral, layer_group, a.batch_size, a.max_length)
        print(f"{int(is_concept.sum())} concept tokens, "
              f"{int((~is_concept).sum())} neutral")
        # keep the tokenizer markers, they help when reading features
        token_strings = tok.convert_ids_to_tokens(token_ids.tolist())
        for l in layer_group:
            print(f"layer {l}: A {tuple(acts[l].shape)}")
            Z, Y = semi_nmf(acts[l], k, a.sparsity, a.ridge, a.max_iter,
                            seed=a.seed, device=dev)
            rho = mass_ratio(Y, is_concept)
            selected = (rho > a.tau).nonzero().flatten().tolist()
            rho_stats[str(l)] = rho_summary(rho, a.tau)
            print(f"  {len(selected)}/{k} features with rho > {a.tau} "
                  f"(max {rho_stats[str(l)]['max']:.4f}, "
                  f"median {rho_stats[str(l)]['median']:.4f})")
            results[l] = {"Z": Z, "rho": rho, "candidates": selected}
            tops = top_tokens_per_feature(Y, token_strings, a.top_tokens)
            feature_tokens[str(l)] = {str(i): tops[i] for i in selected}
            proj = projection_tokens_per_feature(
                model, tok, l, Z, selected, top=a.projection_tokens)
            projection_tokens[str(l)] = {str(i): proj[i] for i in selected}
        del acts
    with open(out / "features.pkl", "wb") as fh:
        pickle.dump({"layers": layers, "k": k, "results": results}, fh)
    json.dump({str(l): results[l]["candidates"] for l in layers},
              open(out / "candidates.json", "w"), indent=2)
    json.dump(feature_tokens, open(out / "feature_tokens.json", "w"),
              indent=2, ensure_ascii=False)
    json.dump(projection_tokens, open(out / "projection_tokens.json", "w"),
              indent=2, ensure_ascii=False)
    (out / "rho_stats.json").write_text(json.dumps(rho_stats, indent=2))
    total_candidates = sum(len(results[l]["candidates"]) for l in layers)
    best = max((s["max"], l) for l, s in rho_stats.items())
    print(f"{total_candidates} candidate features across {len(layers)} layers; "
          f"largest rho anywhere {best[0]:.4f} (layer {best[1]})")
    if total_candidates == 0:
        print(f"NOTHING CLEARS tau={a.tau}. rho_stats.json holds the measured "
              "distribution per layer; pick a threshold from it, and record "
              "that you did, rather than from whether the erasure works.")
    run_metadata = {
        "model": a.model, "concept": a.concept, "k": k,
        "dtype": a.dtype, "n_layers": n_layers, "d_mlp": d_mlp,
        "n_concept_sentences": len(concept), "n_neutral_sentences": len(neutral),
        "sparsity": a.sparsity, "ridge": a.ridge, "tau": a.tau,
        "max_iter": a.max_iter, "seed": a.seed,
        "activation_tokens_saved": a.top_tokens,
        "projection_tokens_saved": a.projection_tokens,
    }
    (out / "factorization_metadata.json").write_text(json.dumps(run_metadata, indent=2))
    print("wrote", out / "features.pkl", "and both token-evidence files")


def cmd_erase(a):
    model, tok = load_model(a.model, a.cache_dir, DTYPES[a.dtype],
                            resolve_device(a.device))
    blob = pickle.load(open(Path(a.out) / "features.pkl", "rb"))
    selected_path = Path(a.out) / "selected.json"
    if not selected_path.exists():
        raise SystemExit(
            "selected.json is missing. Run the select command first.")
    selected = json.load(open(selected_path))
    selection_meta_path = Path(a.out) / "selection_metadata.json"
    if not selection_meta_path.exists():
        raise SystemExit("selection_metadata.json is missing")
    selection_meta = json.load(open(selection_meta_path))
    print("selection method:", selection_meta.get("selection_method", "unknown"))

    (lo_in, hi_in), (lo_out, hi_out) = resolve_ranges(
        len(get_layers(model)), a.layers_in, a.layers_out)
    total = 0
    for l in blob["layers"]:
        feats = selected.get(str(l), [])
        if not feats:
            continue
        d_in = a.delta_in if lo_in <= l <= hi_in else 0.0
        d_out = a.delta_out if lo_out <= l <= hi_out else 0.0
        if not (d_in or d_out):
            continue
        n, kept = ablate_layer(model, l, blob["results"][l]["Z"], feats,
                               d_in, d_out, a.gamma)
        total += n
        print(f"layer {l}: {n} features, {kept} neurons kept, "
              f"delta_in={d_in} delta_out={d_out}")
    print(f"{total} feature ablations total")
    if total == 0:
        # Saving here would write a model byte-identical to its control and
        # call it erased. Nothing downstream could tell.
        raise SystemExit(
            "no feature was ablated: selected.json is empty for every layer "
            f"in [{lo_in},{hi_in}] u [{lo_out},{hi_out}], or both deltas are "
            "zero. Nothing saved.")
    if a.save:
        model.save_pretrained(a.save)
        tok.save_pretrained(a.save)
        erasure_meta = {
            "method": "SNMF",
            "selection": selection_meta,
            "delta_in": a.delta_in, "delta_out": a.delta_out,
            "layers_in": [lo_in, hi_in], "layers_out": [lo_out, hi_out],
            "layer_ranges_from": ("cli" if a.layers_in else "model depth"),
            "dtype": a.dtype,
            "gamma_cov": a.gamma, "feature_ablations": total,
        }
        (Path(a.save) / "snmf_erasure_metadata.json").write_text(
            json.dumps(erasure_meta, indent=2))
        print("saved", a.save)


STAGE1 = """You get tokens that represent a single feature vector (with some noise).
Infer the single most specific, cohesive concept shared by the relevant tokens.
Return only one concise sentence. No preface, no list, no caveats.

Example:
Tokens: ['▁tomorrow','▁tonight','▁yesterday','▁today','▁demain']
Explanation of feature behavior: this vector is related to specific dates and times (e.g., today/tomorrow/yesterday).

Tokens: {tokens}
Explanation of feature behavior:"""

STAGE2 = """You get a concept name and a feature's description.
Decide if this feature describes the given concept. Consider if the feature is
DISTINCTIVE for the concept (not broad like "sport" for "basketball") and not
too noisy. A feature should be marked true only if the concept is central and
dominant in the description, not just one of several themes.
Return ONLY JSON: {{"is_member": true|false, "confidence": 0..1}}

Example 1:
Concept: Charity
Description: This vector represents elements related to giving and donations to charitable causes.
Answer: {{"is_member": true, "confidence": 0.98}}

Example 2:
Concept: Charity
Description: This vector represents elements related to money, such as gambling, investing and charity.
Answer: {{"is_member": false, "confidence": 0.80}}

Concept: {concept}
Description: {description}
Answer:"""


# Judge defaults are the ones the EMBER run on this same 1B used
# (configs/ember_lment_rome_slurm.yaml): the pinned revision, the 23 GB
# already in hf_cache, its own interpreter, and the 3-hour startup budget that
# an unsharded 23.9 GB safetensors read off this filer actually needs.
GEMMA_JUDGE = "google/gemma-4-12B-it"
GEMMA_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
GEMMA_CACHE = "/home/dcor/galbarak2/hf_cache/hub"
GEMMA_PYTHON = "/home/dcor/galbarak2/conda_envs/gemma/bin/python"
GEMMA_STARTUP_TIMEOUT = 10800.0


class LocalGemmaClient:
    """The project's own judge, standing in for Gemini.

    Reuses ``ember.subprocess_judge.SubprocessJudge`` from the EMBER fork on
    this branch. Its ``describe_feature`` / ``classify_feature`` seams take a
    raw prompt and return text, and ``classify_feature`` already normalises
    the reply to ``{"is_member": bool, "confidence": float}`` -- exactly what
    ``_parse_membership`` below expects -- so STAGE1/STAGE2 are unchanged and
    the only thing that differs from the published recipe is which model
    answers them.
    """

    def __init__(self, model=GEMMA_JUDGE, revision=GEMMA_REVISION,
                 python_executable=GEMMA_PYTHON, cache_dir=GEMMA_CACHE,
                 device="cuda", max_new_tokens=256,
                 startup_timeout=GEMMA_STARTUP_TIMEOUT, fork_root=None):
        root = Path(fork_root) if fork_root else (
            Path(__file__).resolve().parent / "Ember-on-LMEnt")
        if not (root / "ember").is_dir():
            raise SystemExit(
                f"--judge gemma needs the EMBER fork: no ember/ package under "
                f"{root}. Pass --fork-root, or run from a checkout of "
                "itamars/Ember-on-LMEnt.")
        sys.path.insert(0, str(root))
        from ember.subprocess_judge import (SubprocessJudge,
                                            resolve_judge_directory)
        model_path = resolve_judge_directory(
            model, revision=revision, cache_dir=cache_dir,
            local_files_only=True)
        print(f"judge: {model_path} via {python_executable}")
        self._judge = SubprocessJudge(
            model_path=model_path, python_executable=python_executable,
            device=device, max_new_tokens=max_new_tokens, cache_dir=cache_dir,
            local_files_only=True, project_root=root,
            startup_timeout=startup_timeout)
        self.metadata = {"judge_backend": "gemma-subprocess",
                         "judge_model": str(model), "judge_revision": revision,
                         "judge_path": str(model_path)}

    def describe(self, prompt):
        return self._judge.describe_feature(prompt)

    def classify(self, prompt):
        return self._judge.classify_feature(prompt)

    def close(self):
        self._judge.close()


class GeminiClient:
    def __init__(self, model_name, max_retries=3, sleep_seconds=5.0):
        key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_TOKEN")
               or os.getenv("GEMINI_API_KEY"))
        if not key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY, GEMINI_API_TOKEN, or GEMINI_API_KEY")
        try:
            import google.generativeai as gai
        except ImportError as exc:
            raise RuntimeError("Install google-generativeai") from exc
        gai.configure(api_key=key)
        self.model = gai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.metadata = {"judge_backend": "gemini", "judge_model": model_name}

    def describe(self, prompt):
        return self.generate(prompt)

    def classify(self, prompt):
        return self.generate(prompt)

    def close(self):
        return None

    def generate(self, prompt):
        last = None
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                text = getattr(response, "text", "")
                if not text or not text.strip():
                    raise RuntimeError("Gemini returned empty text")
                return text.strip()
            except Exception as exc:
                last = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.sleep_seconds)
        raise RuntimeError(f"Gemini failed after {self.max_retries} attempts: {last}")


def _parse_membership(text):
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in Gemini response: {text!r}")
    obj = json.loads(match.group(0))
    return bool(obj["is_member"]), float(obj["confidence"])


def _judge_evidence(client, concept, tokens, top_k, confidence_threshold):
    used = tokens[:top_k]
    description = client.describe(STAGE1.format(tokens=repr(used))).strip("` \n")
    raw = client.classify(STAGE2.format(concept=concept, description=description))
    member, confidence = _parse_membership(raw)
    return {
        "tokens": used, "description": description, "is_member": member,
        "confidence": confidence,
        "accepted": member and confidence >= confidence_threshold,
        "raw_stage2": raw,
    }


def cmd_select(a):
    """Select the concept features used by the weight update."""
    out = Path(a.out)
    candidates = json.load(open(out / "candidates.json"))

    # --skip-llm keeps every rho > tau candidate instead of running the
    # Gemini filter. Factorization and ablation are unchanged; only this
    # selection step stops matching Appendix A.3.
    if a.skip_llm:
        selected = {
            layer: [int(feature) for feature in features]
            for layer, features in candidates.items()
        }
        (out / "selected.json").write_text(json.dumps(selected, indent=2))
        metadata = {
            "selection_method": "ratio_only",
            "llm_filter_used": False,
            "tau": json.load(open(out / "factorization_metadata.json"))["tau"],
            "selected_count": sum(map(len, selected.values())),
        }
        (out / "selection_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"selected {metadata['selected_count']} ratio-filtered features")
        return

    if not a.concept:
        raise SystemExit("--concept is needed unless --skip-llm is used")

    activating = json.load(open(out / "feature_tokens.json"))
    projecting_path = out / "projection_tokens.json"
    if not projecting_path.exists():
        raise SystemExit(
            "projection_tokens.json is missing; rerun factorize with this version. "
            "Both token sources are needed for feature selection.")
    projecting = json.load(open(projecting_path))
    if sum(map(len, candidates.values())) == 0:
        raise SystemExit(
            "no candidate features to judge: every layer's rho fell short of "
            "tau at factorization time. Read out/rho_stats.json before "
            "changing tau, and write down the rule you used.")
    if a.judge == "gemini":
        client = GeminiClient(a.gemini_model, a.max_retries, a.retry_sleep)
    else:
        client = LocalGemmaClient(
            model=a.judge_model, revision=a.judge_revision,
            python_executable=a.judge_python, cache_dir=a.judge_cache_dir,
            device=a.judge_device, max_new_tokens=a.judge_max_new_tokens,
            startup_timeout=a.judge_startup_timeout, fork_root=a.fork_root)
    selected = {layer: [] for layer in candidates}
    records = []
    interpretations_path = out / "interpretations.json"
    try:
        for layer, ids in candidates.items():
            for fid_value in ids:
                fid = str(fid_value)
                record = {"layer": int(layer), "feature": int(fid)}
                for source, data in (("activation", activating),
                                     ("projection", projecting)):
                    print(f"judge layer={layer} feature={fid} source={source}",
                          flush=True)
                    record[source] = _judge_evidence(
                        client, a.concept, data[layer][fid],
                        a.judge_top_tokens, a.confidence_threshold)
                if (record["activation"]["accepted"]
                        or record["projection"]["accepted"]):
                    selected[layer].append(int(fid))
                records.append(record)
                interpretations_path.write_text(
                    json.dumps(records, indent=2, ensure_ascii=False))
    finally:
        client.close()
    (out / "selected.json").write_text(json.dumps(selected, indent=2))
    metadata = {
        "selection_method": f"ratio_then_{a.judge}",
        "llm_filter_used": True,
        "confidence_threshold": a.confidence_threshold,
        "judge_top_tokens": a.judge_top_tokens,
        "sources": ["activation", "projection"],
        "union_rule": "accepted by either independently judged source",
        "selected_count": sum(map(len, selected.values())),
        "n_candidates_judged": len(records),
    }
    metadata.update(client.metadata)
    (out / "selection_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"selected {metadata['selected_count']} of {len(records)} candidates "
          f"({a.judge} judge)")


@torch.no_grad()
def feature_activation(acts, Z, feature_ids):
    """Mean projection of activations onto each concept feature direction."""
    if not feature_ids:
        return torch.zeros(0)
    Zs = Z[:, feature_ids].to(acts.dtype)
    Zs = Zs / Zs.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return (Zs.T @ acts).abs().mean(dim=1)          # (n_features,)


def cmd_verify(a):
    """Compare feature activation before and after the weight update."""
    blob = pickle.load(open(Path(a.out) / "features.pkl", "rb"))
    sel_path = Path(a.out) / "selected.json"
    if not sel_path.exists():
        raise SystemExit("selected.json is missing; choose a feature-selection method first")
    selected = json.load(open(sel_path))
    layers = [l for l in blob["layers"] if selected.get(str(l))]
    if not layers:
        raise SystemExit("no selected features in any layer")

    model, tok = load_model(a.model, a.cache_dir, DTYPES[a.dtype],
                            resolve_device(a.device))
    concept = load_sentences(a.concept_sentences, a.concept)[:a.n_sentences]
    neutral = load_sentences(a.neutral_sentences)[:a.n_sentences]

    def measure():
        acts, is_c, _ = collect_activations(model, tok, concept, neutral, layers,
                                            a.batch_size, a.max_length)
        return {l: (feature_activation(acts[l][:, is_c], blob["results"][l]["Z"],
                                       selected[str(l)]).mean().item(),
                    feature_activation(acts[l][:, ~is_c], blob["results"][l]["Z"],
                                       selected[str(l)]).mean().item())
                for l in layers}

    before = measure()
    (lo_in, hi_in), (lo_out, hi_out) = resolve_ranges(
        len(get_layers(model)), a.layers_in, a.layers_out)
    for l in layers:
        d_in = a.delta_in if lo_in <= l <= hi_in else 0.0
        d_out = a.delta_out if lo_out <= l <= hi_out else 0.0
        if d_in or d_out:
            ablate_layer(model, l, blob["results"][l]["Z"], selected[str(l)],
                         d_in, d_out, a.gamma)
    after = measure()

    print(f"{'layer':>6} {'concept before':>15} {'after':>10} {'drop':>7} "
          f"{'neutral before':>15} {'after':>10} {'drop':>7}")
    drops_c, drops_n = [], []
    for l in layers:
        cb, nb = before[l]
        ca, na = after[l]
        dc = 1 - ca / max(cb, 1e-12)
        dn = 1 - na / max(nb, 1e-12)
        drops_c.append(dc)
        drops_n.append(dn)
        print(f"{l:>6} {cb:>15.4g} {ca:>10.4g} {dc:>6.1%} "
              f"{nb:>15.4g} {na:>10.4g} {dn:>6.1%}")
    mc = sum(drops_c) / len(drops_c)
    mn = sum(drops_n) / len(drops_n)
    print(f"\nmean concept drop {mc:.1%}, mean neutral drop {mn:.1%}, "
          f"selectivity {mc - mn:+.1%}")
    if mc < 0.2:
        print("concept drop is small; the edit is not doing much. Check the "
              "up_proj/down_proj transposes and raise delta.")
    if mn > mc * 0.7:
        print("neutral drops nearly as much as concept; the features are not "
              "concept-specific. Raise tau or apply the LLM filter.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("factorize")
    f.add_argument("--model", required=True)
    f.add_argument("--concept", required=True)
    f.add_argument("--concept-sentences", required=True)
    f.add_argument("--neutral-sentences", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--cache-dir")
    f.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    f.add_argument("--device", help="default cuda:0, or cpu without CUDA")
    f.add_argument("--layers", type=int, nargs="+")
    f.add_argument("--k", type=int)
    f.add_argument("--n-sentences", type=int, default=300)
    f.add_argument("--allow-short-data", action="store_true")
    f.add_argument("--sparsity", type=float, default=0.01)
    f.add_argument("--ridge", type=float, default=1e-4)
    f.add_argument("--tau", type=float, default=2.0)
    f.add_argument("--max-iter", type=int, default=20000)
    f.add_argument("--batch-size", type=int, default=8)
    f.add_argument("--layer-batch-size", type=int, default=1)
    f.add_argument("--max-length", type=int, default=256)
    f.add_argument("--top-tokens", type=int, default=30)
    f.add_argument("--projection-tokens", type=int, default=30)
    f.add_argument("--seed", type=int, default=42)
    f.set_defaults(func=cmd_factorize)

    e = sub.add_parser("erase")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--cache-dir")
    e.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    e.add_argument("--device", help="default cuda:0, or cpu without CUDA")
    e.add_argument("--delta-in", type=float, default=4.0)
    e.add_argument("--delta-out", type=float, default=4.0)
    # default None, resolved from the model's depth; see default_layer_ranges
    e.add_argument("--layers-in", type=int, nargs=2)
    e.add_argument("--layers-out", type=int, nargs=2)
    e.add_argument("--gamma", type=float, default=0.95)
    e.add_argument("--save")
    e.set_defaults(func=cmd_erase)

    p = sub.add_parser("select")
    p.add_argument("--out", required=True)
    p.add_argument("--concept")
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--judge", choices=("gemma", "gemini"), default="gemma",
                   help="gemma is the local pinned judge this project already "
                        "uses for embedding features; gemini needs a key")
    p.add_argument("--judge-model", default=GEMMA_JUDGE)
    p.add_argument("--judge-revision", default=GEMMA_REVISION)
    p.add_argument("--judge-python", default=GEMMA_PYTHON)
    p.add_argument("--judge-cache-dir", default=GEMMA_CACHE)
    p.add_argument("--judge-device", default="cuda")
    p.add_argument("--judge-max-new-tokens", type=int, default=256)
    p.add_argument("--judge-startup-timeout", type=float,
                   default=GEMMA_STARTUP_TIMEOUT)
    p.add_argument("--fork-root",
                   help="path to the Ember-on-LMEnt checkout holding ember/")
    p.add_argument("--gemini-model", default="models/gemini-2.5-flash-lite")
    p.add_argument("--judge-top-tokens", type=int, default=20)
    p.add_argument("--confidence-threshold", type=float, default=0.85)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=5.0)
    p.set_defaults(func=cmd_select)

    v = sub.add_parser("verify")
    v.add_argument("--model", required=True)
    v.add_argument("--concept", required=True)
    v.add_argument("--concept-sentences", required=True)
    v.add_argument("--neutral-sentences", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--cache-dir")
    v.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    v.add_argument("--device", help="default cuda:0, or cpu without CUDA")
    v.add_argument("--delta-in", type=float, default=4.0)
    v.add_argument("--delta-out", type=float, default=4.0)
    v.add_argument("--layers-in", type=int, nargs=2)
    v.add_argument("--layers-out", type=int, nargs=2)
    v.add_argument("--gamma", type=float, default=0.95)
    v.add_argument("--n-sentences", type=int, default=300)
    v.add_argument("--batch-size", type=int, default=8)
    v.add_argument("--max-length", type=int, default=256)
    v.set_defaults(func=cmd_verify)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
