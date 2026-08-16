"""Sparse Semi-NMF erasure for MLP layers.

The factorization uses the Semi-NMF updates from Ding, Li and Jordan. Feature
selection uses the ratio pre-filter and two-stage Gemini filter in Appendix A.3.

    python snmf.py factorize --model <path> --concept <name> \
        --concept-sentences chunks.json --neutral-sentences neutral.json \
        --out snmf_out
    python snmf.py select --out snmf_out --concept <name>
    # Without Gemini credits:
    # python snmf.py select --out snmf_out --skip-llm
    python snmf.py erase --model <path> --out snmf_out \
        --delta-in 4 --delta-out 4 --layers-in 0 15 --layers-out 0 7 \
        --save erased_model
"""
import argparse
import json
import math
import os
import pickle
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name, cache_dir=None, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="auto", torch_dtype=dtype, cache_dir=cache_dir)
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
    for start in range(0, len(feature_ids), 16):
        ids = feature_ids[start:start + 16]
        z = Z[:, ids].T.to(down.device, torch.float32)
        directions = z @ down.float().T
        if final_norm is not None:
            norm_device = next(final_norm.parameters()).device
            directions = final_norm(directions.to(norm_device, down.dtype)).float()
        logits = directions.to(unembed.device) @ unembed.float().T
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
    model, tok = load_model(a.model, a.cache_dir)
    n_layers = len(get_layers(model))
    layers = list(range(n_layers)) if a.layers is None else a.layers
    k = a.k or default_rank(model.config.hidden_size)
    print(f"{n_layers} layers, d_model={model.config.hidden_size}, k={k}")

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
    results, feature_tokens, projection_tokens = {}, {}, {}
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
            print(f"  {len(selected)}/{k} features with rho > {a.tau}")
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
    run_metadata = {
        "model": a.model, "concept": a.concept, "k": k,
        "n_concept_sentences": len(concept), "n_neutral_sentences": len(neutral),
        "sparsity": a.sparsity, "ridge": a.ridge, "tau": a.tau,
        "max_iter": a.max_iter, "seed": a.seed,
        "activation_tokens_saved": a.top_tokens,
        "projection_tokens_saved": a.projection_tokens,
    }
    (out / "factorization_metadata.json").write_text(json.dumps(run_metadata, indent=2))
    print("wrote", out / "features.pkl", "and both token-evidence files")


def cmd_erase(a):
    model, tok = load_model(a.model, a.cache_dir)
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

    lo_in, hi_in = a.layers_in
    lo_out, hi_out = a.layers_out
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
    if a.save:
        model.save_pretrained(a.save)
        tok.save_pretrained(a.save)
        erasure_meta = {
            "method": "SNMF",
            "selection": selection_meta,
            "delta_in": a.delta_in, "delta_out": a.delta_out,
            "layers_in": a.layers_in, "layers_out": a.layers_out,
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
    description = client.generate(STAGE1.format(tokens=repr(used))).strip("` \n")
    raw = client.generate(STAGE2.format(concept=concept, description=description))
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
    client = GeminiClient(a.gemini_model, a.max_retries, a.retry_sleep)
    selected = {layer: [] for layer in candidates}
    records = []
    interpretations_path = out / "interpretations.json"
    for layer, ids in candidates.items():
        for fid_value in ids:
            fid = str(fid_value)
            record = {"layer": int(layer), "feature": int(fid)}
            for source, data in (("activation", activating), ("projection", projecting)):
                print(f"judge layer={layer} feature={fid} source={source}")
                record[source] = _judge_evidence(
                    client, a.concept, data[layer][fid], a.judge_top_tokens,
                    a.confidence_threshold)
            if record["activation"]["accepted"] or record["projection"]["accepted"]:
                selected[layer].append(int(fid))
            records.append(record)
            interpretations_path.write_text(
                json.dumps(records, indent=2, ensure_ascii=False))
    (out / "selected.json").write_text(json.dumps(selected, indent=2))
    metadata = {
        "selection_method": "ratio_then_gemini",
        "llm_filter_used": True,
        "judge_model": a.gemini_model,
        "confidence_threshold": a.confidence_threshold,
        "judge_top_tokens": a.judge_top_tokens,
        "sources": ["activation", "projection"],
        "union_rule": "accepted by either independently judged source",
        "selected_count": sum(map(len, selected.values())),
    }
    (out / "selection_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"selected {metadata['selected_count']} Gemini-filtered features")


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

    model, tok = load_model(a.model, a.cache_dir)
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
    lo_in, hi_in = a.layers_in
    lo_out, hi_out = a.layers_out
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
    e.add_argument("--delta-in", type=float, default=4.0)
    e.add_argument("--delta-out", type=float, default=4.0)
    e.add_argument("--layers-in", type=int, nargs=2, default=[0, 15])
    e.add_argument("--layers-out", type=int, nargs=2, default=[0, 7])
    e.add_argument("--gamma", type=float, default=0.95)
    e.add_argument("--save")
    e.set_defaults(func=cmd_erase)

    p = sub.add_parser("select")
    p.add_argument("--out", required=True)
    p.add_argument("--concept")
    p.add_argument("--skip-llm", action="store_true")
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
    v.add_argument("--delta-in", type=float, default=4.0)
    v.add_argument("--delta-out", type=float, default=4.0)
    v.add_argument("--layers-in", type=int, nargs=2, default=[0, 15])
    v.add_argument("--layers-out", type=int, nargs=2, default=[0, 7])
    v.add_argument("--gamma", type=float, default=0.95)
    v.add_argument("--n-sentences", type=int, default=300)
    v.add_argument("--batch-size", type=int, default=8)
    v.add_argument("--max-length", type=int, default=256)
    v.set_defaults(func=cmd_verify)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
