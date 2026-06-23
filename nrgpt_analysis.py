"""Utilities for energy and surprisal analysis of NRGPT on word-aligned text."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NRGPTContext:
    """Bundle of model components needed for per-word analysis."""

    model: nn.Module       # full NRGPT model (provides ln_f, lm_head)
    tokenizer: object      # transformers tokenizer
    block: nn.Module       # shared BlockGrad_* block (single recurrent unit)
    wte: nn.Embedding      # word token embeddings
    wpe: nn.Embedding      # position embeddings
    block_size: int        # max sequence length (== wpe.num_embeddings)


def make_context(model, tokenizer) -> NRGPTContext:
    """Discover wte, wpe, block from a loaded NRGPT model.

    Mirrors the heuristic discovery used in the notebooks so callers don't
    have to know specific attribute paths.
    """
    block = next(
        m for _, m in model.named_modules()
        if m.__class__.__name__.startswith("BlockGrad")
    )
    embs = sorted(
        (m for _, m in model.named_modules() if isinstance(m, nn.Embedding)),
        key=lambda m: -m.num_embeddings,
    )
    wte, wpe = embs[0], embs[1]
    return NRGPTContext(
        model=model,
        tokenizer=tokenizer,
        block=block,
        wte=wte,
        wpe=wpe,
        block_size=wpe.num_embeddings,
    )


def tokenize_word_by_word(
    words: List[str],
    tokenizer,
    max_tokens: Optional[int] = None,
) -> Tuple[List[int], List[Tuple[int, int, str]]]:
    """Encode a list of words token-by-token and record each word's BPE span.

    Each word is encoded with a leading space except word 0 (matches GPT-2 BPE
    convention: word-initial tokens carry the space). If ``max_tokens`` is given,
    truncates as soon as adding the next word would exceed the limit.

    Returns
    -------
    ids : list[int]
        Concatenated BPE token IDs.
    ranges : list[(start, end, word)]
        For each retained word, the half-open BPE span ``[start, end)`` into ``ids``
        and the original word string.
    """
    ids: List[int] = []
    ranges: List[Tuple[int, int, str]] = []
    for i, w in enumerate(words):
        prefix = "" if i == 0 else " "
        tok = tokenizer.encode(prefix + w, add_special_tokens=False)
        if max_tokens is not None and len(ids) + len(tok) > max_tokens:
            break
        ranges.append((len(ids), len(ids) + len(tok), w))
        ids.extend(tok)
    return ids, ranges


def _energies_at(x: torch.Tensor, ctx: NRGPTContext):
    """Return (E_attn, E_ff, E_total) at every (batch, position) of state ``x``.

    Reproduces the original notebook helper: energies are evaluated in the same
    layer-normed space the block uses for its gradient. ``E_ff`` falls back to
    ``-(net_out ** 2).sum`` for FF heads that don't return into the residual space.
    """
    g = ctx.block.ln(x)
    E_attn = ctx.block.attn.energy(g)
    net_out = ctx.block.ffwd.net(g)
    if net_out.shape[-1] == g.shape[-1]:
        E_ff = -(g * net_out).sum(dim=-1)
    else:
        E_ff = -(net_out ** 2).sum(dim=-1)
    return E_attn, E_ff, E_attn + E_ff


def _attention_stats_at(x: torch.Tensor, ctx: NRGPTContext):
    """Return per-position (expected_score, entropy) summed over heads.

    For each head ``h`` and query position ``A``, computes the causal softmax
    ``p^h_B`` over the attention scores ``s^h_B`` (the entries of ``wei[A, :]``
    inside ``EnergyHead_H``), and then ``<s>_p = sum_B p^h_B s^h_B`` and
    ``H(p) = -sum_B p^h_B log p^h_B``. The two returned tensors are the sums
    of these quantities over heads.

    The NRGPT codebase uses unscaled attention scores (no 1/sqrt(d) factor),
    i.e. effectively ``beta = 1``, so the identity in the main paper specialises to
        E_attn = -(expected_score + entropy),
    where ``E_attn`` is the value returned by ``_energies_at``. This makes
    ``-expected_score`` and ``-entropy`` an exact additive decomposition of the
    attention energy.
    """
    g = ctx.block.ln(x)
    B_size, T, _C = g.shape
    expected_score = torch.zeros(B_size, T, device=g.device, dtype=g.dtype)
    entropy = torch.zeros(B_size, T, device=g.device, dtype=g.dtype)

    for head in ctx.block.attn.heads:
        xH = head.H(g)                              # (B, T, C)
        wei = g @ xH.transpose(-2, -1)              # (B, T, T): score from query A to key B
        mask = head.tril[:T, :T]                    # (T, T): 1 allowed, 0 masked
        wei_masked = wei.masked_fill(mask == 0, float("-inf"))

        # F.softmax(-inf, dim=-1) gives 0 for masked entries; rows that are
        # entirely masked yield NaN, which we zero out below.
        p = F.softmax(wei_masked, dim=-1)           # (B, T, T)
        all_mask_rows = torch.all(mask == 0, dim=-1)  # (T,)
        if all_mask_rows.any():
            zero_row = torch.zeros_like(p)
            p = torch.where(all_mask_rows.view(1, T, 1), zero_row, p)

        # Expected score: dot wei (with masked entries treated as 0 in the
        # product, since p there is 0; avoid 0 * -inf = NaN).
        wei_for_sum = torch.where(mask == 0, torch.zeros_like(wei), wei)
        e_score_h = (p * wei_for_sum).sum(dim=-1)   # (B, T)

        # Entropy: 0 log 0 = 0.
        log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
        H_h = -(p * log_p).sum(dim=-1)              # (B, T)

        expected_score = expected_score + e_score_h
        entropy = entropy + H_h

    return expected_score, entropy


def per_word_energies(
    words: List[str],
    ctx: NRGPTContext,
    layers: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
    max_tokens: Optional[int] = None,
) -> Tuple[List[Dict], List[Tuple[int, int, str]]]:
    """Per-word same-position energies summed across BPE tokens, at each requested layer.

    For each word at BPE span ``[s, e)`` and each ``k`` in ``layers``, records:
    ``E_attn_k``, ``E_ff_k``, ``E_total_k`` (each summed over ``j in [s, e)``).

    Layer 0 is the energy of the initial embedding state (no block iterations);
    layer k>0 is the energy after k recurrent block iterations. The trained
    readout endpoint is layer 6.

    Returns
    -------
    records : list[dict]
        One dict per word with keys ``word``, ``n_bpe``, ``start``, plus
        ``E_attn_k`` / ``E_ff_k`` / ``E_total_k`` for each k.
    ranges : list[(start, end, word)]
        As returned by ``tokenize_word_by_word``.
    """
    if max_tokens is None:
        max_tokens = ctx.block_size

    layer_set = set(layers)
    max_layer = max(layer_set)

    ids, ranges = tokenize_word_by_word(words, ctx.tokenizer, max_tokens=max_tokens)
    input_ids = torch.tensor([ids])
    T = input_ids.shape[1]

    records: List[Dict] = [
        {"word": w, "n_bpe": e - s, "start": s} for s, e, w in ranges
    ]

    with torch.no_grad():
        pos = torch.arange(T, device=input_ids.device)
        x = ctx.wte(input_ids) + ctx.wpe(pos)
        for k in range(max_layer + 1):
            if k in layer_set:
                E_attn, E_ff, E_total = _energies_at(x, ctx)
                for i, (s, e, _w) in enumerate(ranges):
                    records[i][f"E_attn_{k}"] = E_attn[0, s:e].sum().item()
                    records[i][f"E_ff_{k}"] = E_ff[0, s:e].sum().item()
                    records[i][f"E_total_{k}"] = E_total[0, s:e].sum().item()
            if k < max_layer:
                x = ctx.block(x)

    return records, ranges


def per_word_preceding_energies(
    words: List[str],
    ctx: NRGPTContext,
    layers: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
    max_tokens: Optional[int] = None,
) -> Tuple[List[Dict], List[Tuple[int, int, str]]]:
    """Per-position energies at every preceding BPE position, at each requested layer.

    For each word at BPE span ``[s, e)`` and each ``k`` in ``layers``, records the
    energy at every position ``p in [0, s)`` as length-``s`` lists:
    ``E_attn_pre_k``, ``E_ff_pre_k``, ``E_total_pre_k``.

    Intended as a convergence diagnostic for the precondition of Proposition 2.1:
    compute per-position differences ``|E_*_pre_k[p] - E_*_pre_{k-1}[p]|`` for
    ``p < s`` and check that the max over ``p`` is small at the chosen
    measurement layer.

    Word 0 (``s == 0``) yields empty lists.

    Returns
    -------
    records : list[dict]
        One dict per word with keys ``word``, ``n_bpe``, ``start``, plus
        ``E_attn_pre_k`` / ``E_ff_pre_k`` / ``E_total_pre_k`` for each k.
    ranges : list[(start, end, word)]
        As returned by ``tokenize_word_by_word``.
    """
    if max_tokens is None:
        max_tokens = ctx.block_size

    layer_set = set(layers)
    max_layer = max(layer_set)

    ids, ranges = tokenize_word_by_word(words, ctx.tokenizer, max_tokens=max_tokens)
    input_ids = torch.tensor([ids])
    T = input_ids.shape[1]

    records: List[Dict] = [
        {"word": w, "n_bpe": e - s, "start": s} for s, e, w in ranges
    ]

    with torch.no_grad():
        pos = torch.arange(T, device=input_ids.device)
        x = ctx.wte(input_ids) + ctx.wpe(pos)
        for k in range(max_layer + 1):
            if k in layer_set:
                E_attn, E_ff, E_total = _energies_at(x, ctx)
                for i, (s, _e, _w) in enumerate(ranges):
                    records[i][f"E_attn_pre_{k}"]  = E_attn[0, :s].tolist()
                    records[i][f"E_ff_pre_{k}"]    = E_ff[0, :s].tolist()
                    records[i][f"E_total_pre_{k}"] = E_total[0, :s].tolist()
            if k < max_layer:
                x = ctx.block(x)

    return records, ranges


def per_word_attention_stats(
    words: List[str],
    ctx: NRGPTContext,
    layers: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
    max_tokens: Optional[int] = None,
) -> Tuple[List[Dict], List[Tuple[int, int, str]]]:
    """Per-word attention statistics: expected attention score and entropy.

    Decomposes the attention energy at each word's BPE positions into the two
    components of the log-sum-exp / free-energy identity (see Section "Energy
    and attention entropy" of the paper):
        E_attn = -(1/beta) * (expected_score + entropy),
    summed over heads. With beta = 1 (NRGPT codebase uses unscaled scores),
    this becomes ``E_attn = -(expected_score + entropy)`` exactly.

    For each word at BPE span ``[s, e)`` and each ``k`` in ``layers``, records:

    - ``expected_score_k`` : sum over heads of the expected attention score
      ``<s>_p`` at positions ``s, ..., e-1``, summed over those positions.
    - ``entropy_k`` : sum over heads of the attention entropy ``H(p)`` at
      positions ``s, ..., e-1``, summed over those positions.

    The contributions to attention energy are
        E_attn_contribution_from_score  = -expected_score_k,
        E_attn_contribution_from_entropy = -entropy_k,
    and their sum equals ``E_attn_k`` from ``per_word_energies`` up to
    floating-point error.

    Returns
    -------
    records : list[dict]
        One dict per word with keys ``word``, ``n_bpe``, ``start``, plus
        ``expected_score_k`` and ``entropy_k`` for each ``k``.
    ranges : list[(start, end, word)]
        As returned by ``tokenize_word_by_word``.
    """
    if max_tokens is None:
        max_tokens = ctx.block_size

    layer_set = set(layers)
    max_layer = max(layer_set)

    ids, ranges = tokenize_word_by_word(words, ctx.tokenizer, max_tokens=max_tokens)
    input_ids = torch.tensor([ids])
    T = input_ids.shape[1]

    records: List[Dict] = [
        {"word": w, "n_bpe": e - s, "start": s} for s, e, w in ranges
    ]

    with torch.no_grad():
        pos = torch.arange(T, device=input_ids.device)
        x = ctx.wte(input_ids) + ctx.wpe(pos)
        for k in range(max_layer + 1):
            if k in layer_set:
                e_score, ent = _attention_stats_at(x, ctx)
                for i, (s, e, _w) in enumerate(ranges):
                    records[i][f"expected_score_{k}"] = e_score[0, s:e].sum().item()
                    records[i][f"entropy_{k}"]        = ent[0, s:e].sum().item()
            if k < max_layer:
                x = ctx.block(x)

    return records, ranges


def per_word_predictive_energies(
    words: List[str],
    ctx: NRGPTContext,
    layers: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
    max_tokens: Optional[int] = None,
) -> Tuple[List[Dict], List[Tuple[int, int, str]]]:
    """Per-word predictive energies at each requested layer.

    For each word at BPE span ``[s, e)`` and each ``k`` in ``layers``, records two
    flavors of predictive energy:

    - ``E_*_pred_k``      = energy at position ``s - 1`` only (single 'pre-word' state).
      Strict pre-word measurement. For multi-BPE words this captures less than
      surprisal does, so it is typically used with a single-BPE filter.

    - ``E_*_pred_sum_k``  = sum over positions ``[s - 1, e - 1)``, the energy
      chain-rule analog of surprisal. Matches surprisal's scope on all words.
      The structural parallel to surprisal is exact (each position in the sum is
      the state whose output scores the corresponding within-word BPE), but the
      summed quantity only has a probabilistic interpretation to the extent that
      F.1 holds (E ~ -log P).

    Additionally, the function records the attention-energy decomposition
    (see ``_attention_stats_at`` and the "Energy and attention entropy"
    section of the paper):

    - ``expected_score_pred_k``, ``entropy_pred_k`` : the expected attention
      score ``<s>_p`` and the attention entropy ``H(p)``, summed over heads,
      at position ``s - 1``.
    - ``expected_score_pred_sum_k``, ``entropy_pred_sum_k`` : the same
      quantities summed over positions ``[s - 1, e - 1)``.

    With the codebase's beta = 1 these satisfy
        E_attn_pred_k = -(expected_score_pred_k + entropy_pred_k),
    so the two new fields exactly decompose the attention-energy predictor
    into a match-quality term (``-expected_score``) and the Ryu+25
    attention-entropy term (``-entropy``).

    Word 0 (s == 0) has no pre-word position, so all fields are NaN there.

    Returns
    -------
    records : list[dict]
        One dict per word with keys ``word``, ``n_bpe``, ``start``, plus
        ``E_attn_pred_k``, ``E_ff_pred_k``, ``E_total_pred_k``,
        ``expected_score_pred_k``, ``entropy_pred_k``, and the matching
        ``_pred_sum_k`` fields for each k in ``layers``.
    ranges : list[(start, end, word)]
        As returned by ``tokenize_word_by_word``.
    """
    if max_tokens is None:
        max_tokens = ctx.block_size

    layer_set = set(layers)
    max_layer = max(layer_set)

    ids, ranges = tokenize_word_by_word(words, ctx.tokenizer, max_tokens=max_tokens)
    input_ids = torch.tensor([ids])
    T = input_ids.shape[1]

    records: List[Dict] = [
        {"word": w, "n_bpe": e - s, "start": s} for s, e, w in ranges
    ]

    nan = float("nan")
    nan_keys_per_layer = [
        "E_attn_pred", "E_ff_pred", "E_total_pred",
        "E_attn_pred_sum", "E_ff_pred_sum", "E_total_pred_sum",
        "expected_score_pred", "entropy_pred",
        "expected_score_pred_sum", "entropy_pred_sum",
    ]

    with torch.no_grad():
        pos = torch.arange(T, device=input_ids.device)
        x = ctx.wte(input_ids) + ctx.wpe(pos)
        for k in range(max_layer + 1):
            if k in layer_set:
                E_attn, E_ff, E_total = _energies_at(x, ctx)
                e_score, ent = _attention_stats_at(x, ctx)
                for i, (s, e, _w) in enumerate(ranges):
                    if s == 0:
                        for key in nan_keys_per_layer:
                            records[i][f"{key}_{k}"] = nan
                        continue
                    records[i][f"E_attn_pred_{k}"]  = E_attn[0, s - 1].item()
                    records[i][f"E_ff_pred_{k}"]    = E_ff[0, s - 1].item()
                    records[i][f"E_total_pred_{k}"] = E_total[0, s - 1].item()
                    records[i][f"E_attn_pred_sum_{k}"]  = E_attn[0, s - 1:e - 1].sum().item()
                    records[i][f"E_ff_pred_sum_{k}"]    = E_ff[0, s - 1:e - 1].sum().item()
                    records[i][f"E_total_pred_sum_{k}"] = E_total[0, s - 1:e - 1].sum().item()
                    records[i][f"expected_score_pred_{k}"]     = e_score[0, s - 1].item()
                    records[i][f"entropy_pred_{k}"]            = ent[0, s - 1].item()
                    records[i][f"expected_score_pred_sum_{k}"] = e_score[0, s - 1:e - 1].sum().item()
                    records[i][f"entropy_pred_sum_{k}"]        = ent[0, s - 1:e - 1].sum().item()
            if k < max_layer:
                x = ctx.block(x)

    return records, ranges


def per_word_predictive_conditional_energies(
    words: List[str],
    ctx: NRGPTContext,
    layers: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
    max_tokens: Optional[int] = None,
) -> Tuple[List[Dict], List[Tuple[int, int, str]]]:
    """Per-word token-conditional predictive energies at each requested layer.

    "Token-conditional" means: at the predictive position ``j``, we substitute the
    actual upcoming token's input embedding (``wte[ids[j+1]] + wpe[j]``), then
    compute the energy of that substituted state using context-aware keys from the
    rest of the original forward pass (the self-key at position ``j`` is replaced
    by the substituted state's own key, matching the original notebook recipe).

    The resulting quantity depends on both the context AND the identity of the
    actual next token, the same dependency surprisal has. This is the closest
    energy-analog of surprisal.

    For each word at BPE span ``[s, e)`` and each ``k`` in ``layers``:

    - ``E_*_token_k``     : substitution at position ``s - 1`` only (substituting
      token ``ids[s]``). Strict single-slot measurement.
    - ``E_*_token_sum_k`` : sum over predictive positions ``j in [s - 1, e - 1)``,
      substituting token ``ids[j + 1]`` at each ``j``. Chain-rule analog of
      surprisal.

    For single-BPE words the two flavors are identical. Word 0 (s == 0) is NaN.

    Returns
    -------
    records : list[dict]
        One dict per word with keys ``word``, ``n_bpe``, ``start``, plus
        ``E_attn_token_k``, ``E_ff_token_k``, ``E_total_token_k`` and matching
        ``_token_sum_k`` fields for each ``k``.
    ranges : list[(start, end, word)]
        As returned by ``tokenize_word_by_word``.
    """
    if max_tokens is None:
        max_tokens = ctx.block_size

    layer_set = set(layers)
    max_layer = max(layer_set)

    ids, ranges = tokenize_word_by_word(words, ctx.tokenizer, max_tokens=max_tokens)

    input_ids = torch.tensor([ids])
    T = input_ids.shape[1]

    records: List[Dict] = [
        {"word": w, "n_bpe": e - s, "start": s} for s, e, w in ranges
    ]

    nan = float("nan")
    nan_keys = [
        "E_attn_token", "E_ff_token", "E_total_token",
        "E_attn_token_sum", "E_ff_token_sum", "E_total_token_sum",
    ]
    neg_inf = float("-inf")

    with torch.no_grad():
        pos = torch.arange(T, device=input_ids.device)
        x = ctx.wte(input_ids) + ctx.wpe(pos)
        for k in range(max_layer + 1):
            if k in layer_set:
                g_layer = ctx.block.ln(x)[0]
                all_xH = [head.H(g_layer.unsqueeze(0))[0] for head in ctx.block.attn.heads]

                for i, (s, e, _w) in enumerate(ranges):
                    if s == 0:
                        for key in nan_keys:
                            records[i][f"{key}_{k}"] = nan
                        continue

                    E_attn_first = E_ff_first = None
                    E_attn_sum = 0.0
                    E_ff_sum = 0.0

                    for j in range(s - 1, e - 1):
                        sub_token = ids[j + 1]
                        x_subst = ctx.wte.weight[sub_token] + ctx.wpe.weight[j]
                        g_query = ctx.block.ln(x_subst.unsqueeze(0).unsqueeze(0))[0, 0]

                        E_attn_j = 0.0
                        for h_idx, head in enumerate(ctx.block.attn.heads):
                            xH_self = head.H(g_query.unsqueeze(0))[0]
                            wei = (g_query @ all_xH[h_idx].t()).clone()
                            wei[j] = g_query @ xH_self
                            mask = head.tril[j, :T]
                            wei = wei.masked_fill(mask == 0, neg_inf)
                            E_attn_j += -torch.logsumexp(wei, dim=0).item()

                        net_out = ctx.block.ffwd.net(g_query.unsqueeze(0).unsqueeze(0))[0, 0]
                        if net_out.shape[-1] == g_query.shape[-1]:
                            E_ff_j = -(g_query * net_out).sum().item()
                        else:
                            E_ff_j = -(net_out ** 2).sum().item()

                        if E_attn_first is None:
                            E_attn_first = E_attn_j
                            E_ff_first = E_ff_j
                        E_attn_sum += E_attn_j
                        E_ff_sum += E_ff_j

                    records[i][f"E_attn_token_{k}"]      = E_attn_first
                    records[i][f"E_ff_token_{k}"]        = E_ff_first
                    records[i][f"E_total_token_{k}"]     = E_attn_first + E_ff_first
                    records[i][f"E_attn_token_sum_{k}"]  = E_attn_sum
                    records[i][f"E_ff_token_sum_{k}"]    = E_ff_sum
                    records[i][f"E_total_token_sum_{k}"] = E_attn_sum + E_ff_sum

            if k < max_layer:
                x = ctx.block(x)

    return records, ranges


def plot_energy_landscape_2d(
    words: List[str],
    ctx: NRGPTContext,
    word_idx: int,
    n_steps: int = 6,
    start_step: int = 0,
    grid_size: int = 40,
    margin: float = 0.5,
    max_tokens: Optional[int] = None,
    show_predictions: bool = False,
    label_bbox: bool = False,
    overlay_color: str = "black",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax=None,
) -> Dict:
    """Plot the 2D PCA energy landscape at a word's first BPE, with the layer trajectory overlaid.

    Procedure:
      1. Forward pass; save the state ``x`` at every layer ``k`` in ``0..n_steps``.
      2. Pick BPE position ``p`` = first token of word ``word_idx``.
         Trajectory := [x_k[p] for k in start_step..n_steps]  (each a vector in R^D).
      3. PCA on the trajectory (top 2 PCs). Project the trajectory onto the PC1-PC2 plane.
      4. Grid the plane around the trajectory's bounding box.
      5. For each grid point (a, b), lift to high-D as
         ``x_p = mean(traj) + a * PC1 + b * PC2`` and compute the energy
         ``E_total`` at position p using the layer-``n_steps`` context (keys at all
         other positions taken from the final forward-pass state).
      6. Plot ``E_total`` as a filled contour and overlay the 2D-projected trajectory.

    The trajectory points are shown at their 2D projections; the residual (PC3+)
    is not visualised. The landscape is "what the energy at position p looks like
    when we restrict x[p] to the trajectory's principal plane, using the trained
    L6 context."

    ``start_step`` lets you hide the initial layers from the plot AND from the PCA
    fit (so the bounding box and basis match what is drawn). Set ``start_step=1``
    to exclude L0 (the raw embedding, before any block iteration).

    ``show_predictions=True`` annotates each Lk dot with the top-1 token the trained
    readout (``ln_f`` + ``lm_head``) would pick from that intermediate state. Only
    Lk = ``n_steps`` matches the model's actual prediction; earlier Lk readouts are
    off-distribution for the head and are a "logit lens" diagnostic.

    ``label_bbox=True`` puts a translucent box behind each label for legibility
    against the contour; the box uses a colour contrasting with ``overlay_color``.

    ``overlay_color`` sets the colour of the trajectory line and the Lk labels
    (default ``"black"``). Set to ``"white"`` if you flip the cmap to one with a
    light low-energy region.

    ``vmin``, ``vmax`` fix the energy colour scale across plots (use the same pair
    on multiple panels for a shared scale). Values outside the range are shown
    with the extreme colours via ``extend="both"``. If omitted, the scale is set
    per-plot.

    Returns
    -------
    dict with keys:
        ``fig``, ``ax``, ``traj_2d``, ``traj_energies_true``, ``traj_energies_lifted``,
        ``grid_xs``, ``grid_ys``, ``energy_grid``, ``pc1``, ``pc2``, ``var_explained``,
        ``word``, ``position``, ``start_step``, ``n_steps``, ``predictions``
        (list[str] of top-1 readout words at L`start_step`..L`n_steps`, or None
        if ``show_predictions=False``).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if max_tokens is None:
        max_tokens = ctx.block_size

    if not 0 <= start_step <= n_steps:
        raise ValueError(f"start_step must be in [0, n_steps]; got start_step={start_step}, n_steps={n_steps}")
    if n_steps - start_step < 2:
        raise ValueError("Need at least 2 visible trajectory points for PCA")

    ids, ranges = tokenize_word_by_word(words, ctx.tokenizer, max_tokens=max_tokens)
    if word_idx >= len(ranges):
        raise ValueError(f"word_idx {word_idx} out of range (got {len(ranges)} words)")

    s, e, target_word = ranges[word_idx]
    p = s

    input_ids = torch.tensor([ids])
    T = input_ids.shape[1]

    with torch.no_grad():
        pos = torch.arange(T, device=input_ids.device)
        x = ctx.wte(input_ids) + ctx.wpe(pos)
        states = [x.clone()]
        for _ in range(n_steps):
            x = ctx.block(x)
            states.append(x.clone())

        traj = torch.stack([st[0, p] for st in states[start_step:]], dim=0)
        D = traj.shape[1]

        center = traj.mean(dim=0)
        traj_c = traj - center
        _U, S, Vh = torch.linalg.svd(traj_c, full_matrices=False)
        pc1, pc2 = Vh[0], Vh[1]
        var_explained = (S[:2] ** 2 / (S ** 2).sum()).cpu().numpy()

        basis = torch.stack([pc1, pc2], dim=1)
        traj_2d = traj_c @ basis

        x_min, x_max = traj_2d[:, 0].min().item(), traj_2d[:, 0].max().item()
        y_min, y_max = traj_2d[:, 1].min().item(), traj_2d[:, 1].max().item()
        x_pad = max(x_max - x_min, 1e-6) * margin
        y_pad = max(y_max - y_min, 1e-6) * margin
        x_min -= x_pad; x_max += x_pad
        y_min -= y_pad; y_max += y_pad

        xs = torch.linspace(x_min, x_max, grid_size)
        ys = torch.linspace(y_min, y_max, grid_size)
        coords = torch.cartesian_prod(xs, ys)
        G = center.unsqueeze(0) + coords[:, 0:1] * pc1.unsqueeze(0) + coords[:, 1:2] * pc2.unsqueeze(0)

        def _energy_at_p_for_batch(X_batch, context_keys_per_head):
            """X_batch: (N, D) candidate states for position p. Returns (N,) E_total."""
            G_ln = ctx.block.ln(X_batch.unsqueeze(0))[0]
            E_attn = torch.zeros(X_batch.shape[0])
            for h_idx, head in enumerate(ctx.block.attn.heads):
                xH_self = head.H(G_ln.unsqueeze(0))[0]
                wei = G_ln @ context_keys_per_head[h_idx].t()
                wei[:, p] = (G_ln * xH_self).sum(dim=-1)
                mask = head.tril[p, :T]
                wei = wei.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))
                E_attn = E_attn - torch.logsumexp(wei, dim=-1)
            net_out = ctx.block.ffwd.net(G_ln.unsqueeze(0))[0]
            if net_out.shape[-1] == G_ln.shape[-1]:
                E_ff = -(G_ln * net_out).sum(dim=-1)
            else:
                E_ff = -(net_out ** 2).sum(dim=-1)
            return E_attn + E_ff

        x_final = states[n_steps]
        g_final = ctx.block.ln(x_final)[0]
        keys_final = [head.H(g_final.unsqueeze(0))[0] for head in ctx.block.attn.heads]

        E_grid_flat = _energy_at_p_for_batch(G, keys_final)
        energy_grid = E_grid_flat.reshape(grid_size, grid_size)

        traj_lifted = center.unsqueeze(0) + traj_2d[:, 0:1] * pc1.unsqueeze(0) + traj_2d[:, 1:2] * pc2.unsqueeze(0)
        traj_E_lifted = _energy_at_p_for_batch(traj_lifted, keys_final)
        traj_E_true = _energy_at_p_for_batch(traj, keys_final)

        predictions = None
        if show_predictions:
            g_traj = ctx.model.ln_f(traj.unsqueeze(0))
            logits = ctx.model.lm_head(g_traj)[0]
            top_ids = logits.argmax(dim=-1).tolist()
            predictions = [ctx.tokenizer.decode([tid]).strip() for tid in top_ids]

    xs_np = xs.cpu().numpy()
    ys_np = ys.cpu().numpy()
    energy_np = energy_grid.T.cpu().numpy()
    traj_2d_np = traj_2d.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    X_mesh, Y_mesh = np.meshgrid(xs_np, ys_np)
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 26)
        cs = ax.contourf(X_mesh, Y_mesh, energy_np, levels=levels, cmap="viridis", extend="both")
    else:
        cs = ax.contourf(X_mesh, Y_mesh, energy_np, levels=25, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=f"Total energy (L6 context)")

    ax.plot(traj_2d_np[:, 0], traj_2d_np[:, 1], color=overlay_color, lw=1.2, alpha=0.8)
    n_visible = traj_2d_np.shape[0]
    colors = plt.cm.plasma(np.linspace(0, 1, n_visible))
    bbox_color = "white" if overlay_color == "black" else "black"
    for k_offset, (px, py) in enumerate(traj_2d_np):
        k = start_step + k_offset
        ax.scatter(px, py, color=colors[k_offset], edgecolor="black",
                   s=70 if k in (start_step, n_steps) else 40, zorder=5)
        label = f"L{k}: {predictions[k_offset]}" if predictions is not None else f"L{k}"
        bbox = dict(facecolor=bbox_color, alpha=0.25, pad=1, edgecolor="none") if label_bbox else None
        if k in (4, 5):
            offset, ha, va = (8, 0), "left", "center"
        elif k == 3:
            offset, ha, va = (4, -8), "left", "center"
        else:
            offset, ha, va = (0, -10), "center", "top"
        ax.annotate(label, (px, py), color=overlay_color, fontsize=11,
                    xytext=offset, textcoords="offset points",
                    ha=ha, va=va, bbox=bbox)

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)")
    ax.set_title(f"Energy landscape at word '{target_word}' (BPE pos {p})")

    return {
        "fig": fig,
        "ax": ax,
        "traj_2d": traj_2d_np,
        "traj_energies_true":   traj_E_true.cpu().numpy(),
        "traj_energies_lifted": traj_E_lifted.cpu().numpy(),
        "grid_xs": xs_np,
        "grid_ys": ys_np,
        "energy_grid": energy_np,
        "pc1": pc1.cpu().numpy(),
        "pc2": pc2.cpu().numpy(),
        "var_explained": var_explained,
        "word": target_word,
        "position": p,
        "start_step": start_step,
        "n_steps": n_steps,
        "predictions": predictions,
    }


class TiktokenWrapper:
    """Thin tiktoken wrapper with a HuggingFace-compatible encode interface."""

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)
        self.max_token_value = self._enc.max_token_value

    def encode(self, text: str, add_special_tokens: bool = False, return_tensors=None):
        ids = self._enc.encode(text)
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids

    def decode(self, ids) -> str:
        return self._enc.decode([t for t in ids if t <= self.max_token_value])


def load_nrgpt(model_name: str = "nrgpt_local"):
    """Load an NRGPT model and a compatible tokenizer.

    Parameters
    ----------
    model_name : str
        ``"nrgpt_local"`` loads the HuggingFace model from ``./nrgpt_local``.
        Any other string is treated as a path to a raw ``.pt`` training checkpoint
        (e.g. the OWT02 best-checkpoint file).
    """
    if model_name == "nrgpt_local":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("./nrgpt_local", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("./nrgpt_local")
        model.eval()
        return model, tokenizer

    import sys
    sys.path.insert(0, "nrgpt")
    sys.path.insert(0, "nrgpt/models")
    from model_config import ModelConfig
    from models.energy_models import NRGPT_H_FF2W

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_name, map_location=device)
    config = ModelConfig(**checkpoint["model_args"])
    model = NRGPT_H_FF2W(config).to(device)
    state_dict = checkpoint["model"]
    for prefix in ["_orig_mod.", "module."]:
        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {model_name}")
    return model, TiktokenWrapper("gpt2")


def plot_energy_landscape_pca(
    words: List[str],
    ctx: NRGPTContext,
    word_idx: int,
    layer: int = 5,
    grid_size: int = 40,
    margin: float = 0.5,
    max_tokens: Optional[int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax=None,
):
    """Plot the energy landscape at a fixed layer, using PCA over all token states.

    The 2D plane is the top-2 PCA components of all token states at ``layer``.
    For each grid point (lifted back to D-dim), the energy at position ``p``
    (first BPE token of ``word_idx``) is computed using the context from that layer.
    Current token (position p) and next token (position p+1) are marked on the plot.

    Parameters
    ----------
    words : list[str]
    ctx : NRGPTContext
    word_idx : int
        Index into ``words`` of the token to analyse.
    layer : int
        Number of recurrent block iterations before measuring (default 5).
    grid_size : int
        Resolution of the energy grid along each axis.
    margin : float
        Fractional padding around the token cloud bounding box.
    vmin, vmax : float, optional
        Fix the colour scale across plots. Values outside the range are shown
        with the extreme colours. If omitted, the scale is set per-plot.
    ax : matplotlib Axes, optional
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if max_tokens is None:
        max_tokens = ctx.block_size

    ids, ranges = tokenize_word_by_word(words, ctx.tokenizer, max_tokens=max_tokens)
    if word_idx >= len(ranges):
        raise ValueError(f"word_idx {word_idx} out of range (got {len(ranges)} words)")

    s, _e, current_word = ranges[word_idx]
    p = s
    p_next = p + 1

    input_ids = torch.tensor([ids])
    T = input_ids.shape[1]
    neg_inf = float("-inf")

    with torch.no_grad():
        pos_idx = torch.arange(T, device=input_ids.device)
        x = ctx.wte(input_ids) + ctx.wpe(pos_idx)
        for _ in range(layer):
            x = ctx.block(x)
        states = x[0]  # (T, D)

        # PCA over all token states
        center = states.mean(dim=0)
        states_c = states - center
        _, _, Vh = torch.linalg.svd(states_c, full_matrices=False)
        pc1, pc2 = Vh[0], Vh[1]

        basis = torch.stack([pc1, pc2], dim=1)  # (D, 2)
        states_2d = states_c @ basis             # (T, 2)

        # Grid bounds from the token cloud
        x_min, x_max = states_2d[:, 0].min().item(), states_2d[:, 0].max().item()
        y_min, y_max = states_2d[:, 1].min().item(), states_2d[:, 1].max().item()
        x_pad = max(x_max - x_min, 1e-6) * margin
        y_pad = max(y_max - y_min, 1e-6) * margin
        xs = torch.linspace(x_min - x_pad, x_max + x_pad, grid_size)
        ys = torch.linspace(y_min - y_pad, y_max + y_pad, grid_size)
        coords = torch.cartesian_prod(xs, ys)           # (grid_size^2, 2)
        G = center + coords[:, 0:1] * pc1 + coords[:, 1:2] * pc2  # (N, D)

        # Precompute context keys from the actual forward-pass states
        g_ctx = ctx.block.ln(x)[0]  # (T, D)
        keys_per_head = [head.H(g_ctx.unsqueeze(0))[0] for head in ctx.block.attn.heads]

        # Energy at position p for each grid point
        def _energy_batch(G_batch: torch.Tensor) -> torch.Tensor:
            G_ln = ctx.block.ln(G_batch.unsqueeze(0))[0]  # (N, D)
            E_attn = torch.zeros(G_batch.shape[0], device=G_batch.device)
            for h_idx, head in enumerate(ctx.block.attn.heads):
                xH_self = head.H(G_ln.unsqueeze(0))[0]          # (N, D)
                wei = G_ln @ keys_per_head[h_idx].t()            # (N, T)
                wei[:, p] = (G_ln * xH_self).sum(dim=-1)
                mask = head.tril[p, :T]
                wei = wei.masked_fill(mask.unsqueeze(0) == 0, neg_inf)
                E_attn -= torch.logsumexp(wei, dim=-1)
            net_out = ctx.block.ffwd.net(G_ln.unsqueeze(0))[0]  # (N, D or D*ff)
            if net_out.shape[-1] == G_ln.shape[-1]:
                E_ff = -(G_ln * net_out).sum(dim=-1)
            else:
                E_ff = -(net_out ** 2).sum(dim=-1)
            return E_attn + E_ff

        E_flat = _energy_batch(G)
        energy_grid = E_flat.reshape(grid_size, grid_size)

    xs_np = xs.cpu().numpy()
    ys_np = ys.cpu().numpy()
    energy_np = energy_grid.T.cpu().numpy()
    states_2d_np = states_2d.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    X_mesh, Y_mesh = np.meshgrid(xs_np, ys_np)
    import numpy as np
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 26)
        cs = ax.contourf(X_mesh, Y_mesh, energy_np, levels=levels, cmap="RdBu_r", extend="both")
    else:
        cs = ax.contourf(X_mesh, Y_mesh, energy_np, levels=25, cmap="RdBu_r")
    fig.colorbar(cs, ax=ax, label=f"E_total at position {p}")

    # Mark current token
    ax.scatter(*states_2d_np[p], color="white", s=140, zorder=5,
               edgecolor="black", linewidths=1.5, label=f"current: '{current_word}'")

    # Mark next token
    if p_next < T:
        next_label = words[word_idx + 1] if word_idx + 1 < len(words) else f"pos {p_next}"
        ax.scatter(*states_2d_np[p_next], color="yellow", s=140, zorder=5,
                   edgecolor="black", linewidths=1.5, marker="^",
                   label=f"next: '{next_label}'")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Energy landscape — layer {layer} — word '{current_word}'")
    ax.legend(framealpha=0.7)

    return fig, ax


def load_gpt2(model_name: str = "gpt2"):
    """Load a HuggingFace GPT-2 model and its tokenizer.

    Parameters
    ----------
    model_name : str
        ``"gpt2"`` (124M, default), ``"gpt2-medium"`` (355M), ``"gpt2-large"`` (774M),
        or ``"gpt2-xl"`` (1.5B). Any HuggingFace-compatible path is also accepted.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def per_word_surprisal_gpt2(
    words: List[str],
    model,
    tokenizer,
    max_tokens: Optional[int] = None,
) -> Tuple[List[float], List[Tuple[int, int, str]]]:
    """Return ``-sum_j log P(token_j | prefix)`` over each word's BPE tokens, using GPT-2.

    Mirrors ``per_word_surprisal`` for a HuggingFace GPT-2 model. Word 0 has no
    prior context, so its surprisal is NaN.

    Parameters
    ----------
    words : list[str]
        Words in order, one per zone.
    model : GPT2LMHeadModel
        A loaded HuggingFace GPT-2 (or compatible) causal LM.
    tokenizer : GPT2Tokenizer / GPT2TokenizerFast
        The matching tokenizer.
    max_tokens : int, optional
        Truncation limit. Defaults to ``model.config.n_positions``.

    Returns
    -------
    surprisals : list[float]
        One value per word in ``ranges``.
    ranges : list[(start, end, word)]
        As returned by ``tokenize_word_by_word``.
    """
    if max_tokens is None:
        max_tokens = model.config.n_positions

    ids, ranges = tokenize_word_by_word(words, tokenizer, max_tokens=max_tokens)
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], device=device)

    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = F.log_softmax(logits, -1)

    out: List[float] = []
    for s, e, _w in ranges:
        if s == 0:
            out.append(float("nan"))
            continue
        out.append(sum(-log_probs[0, j - 1, ids[j]].item() for j in range(s, e)))
    return out, ranges


def per_word_surprisal(
    words: List[str],
    ctx: NRGPTContext,
    n_steps: int = 6,
    max_tokens: Optional[int] = None,
) -> Tuple[List[float], List[Tuple[int, int, str]]]:
    """Return ``-sum_j log P(token_j | prefix)`` over each word's BPE tokens.

    Word 0 has no prior context, so its surprisal is NaN.

    Parameters
    ----------
    words : list[str]
        Words in order, one per zone.
    ctx : NRGPTContext
        Bundled model and tokenizer.
    n_steps : int
        Number of recurrent block iterations before reading out the lm_head
        (the trained endpoint is 6).
    max_tokens : int, optional
        Truncation limit. Defaults to ``ctx.block_size``.

    Returns
    -------
    surprisals : list[float]
        One value per word in ``ranges``.
    ranges : list[(start, end, word)]
        As returned by ``tokenize_word_by_word``.
    """
    if max_tokens is None:
        max_tokens = ctx.block_size

    ids, ranges = tokenize_word_by_word(words, ctx.tokenizer, max_tokens=max_tokens)
    input_ids = torch.tensor([ids])
    T = input_ids.shape[1]

    with torch.no_grad():
        pos = torch.arange(T, device=input_ids.device)
        x = ctx.wte(input_ids) + ctx.wpe(pos)
        for _ in range(n_steps):
            x = ctx.block(x)
        g = ctx.model.ln_f(x)
        logits = ctx.model.lm_head(g)
        log_probs = F.log_softmax(logits, -1)

    out: List[float] = []
    for s, e, _w in ranges:
        if s == 0:
            out.append(float("nan"))
            continue
        out.append(sum(-log_probs[0, j - 1, ids[j]].item() for j in range(s, e)))
    return out, ranges


# ----------------------------------------------------------------------------
# Example: per-word convergence diagnostic.
# For each word, show its own energy trajectory across all layers, and the
# layer-to-layer change. Paste into a Jupyter cell.
# ----------------------------------------------------------------------------
#
# import numpy as np
# import matplotlib.pyplot as plt
# from nrgpt_analysis import load_nrgpt, make_context, per_word_energies
#
# model, tokenizer = load_nrgpt("nrgpt_local")
# ctx = make_context(model, tokenizer)
#
# words = "The bus driver who the kids followed wondered about the hotel".split()
# layers = (0, 1, 2, 3, 4, 5, 6)
#
# # E_total per word (summed over its BPE span) at every layer.
# records, ranges = per_word_energies(words, ctx, layers=layers)
#
# # Stack into arrays of shape (n_words, n_layers).
# E = np.array([[r[f"E_total_{k}"] for k in layers] for r in records])
# dE = np.abs(np.diff(E, axis=1))  # shape (n_words, n_layers - 1)
#
# # Print per-word trajectory and layer-to-layer change.
# header = "word".rjust(12) + " | " + " ".join(f"L{k}".rjust(9) for k in layers)
# print(header)
# print("-" * len(header))
# for r, row in zip(records, E):
#     print(f"{r['word']:>12s} | " + " ".join(f"{v:9.2f}" for v in row))
#
# print()
# header2 = "word".rjust(12) + " | " + " ".join(
#     f"L{a}->L{b}".rjust(9) for a, b in zip(layers[:-1], layers[1:])
# )
# print(header2)
# print("-" * len(header2))
# for r, row in zip(records, dE):
#     print(f"{r['word']:>12s} | " + " ".join(f"{v:9.2f}" for v in row))
#
# # Plot: trajectories on the left, layer-to-layer changes on the right.
# fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# cmap = plt.cm.viridis(np.linspace(0, 1, len(records)))
#
# for i, (r, row) in enumerate(zip(records, E)):
#     axes[0].plot(layers, row, marker="o", color=cmap[i], label=r["word"])
# axes[0].set_xlabel("layer (block iteration)")
# axes[0].set_ylabel("E_total (per word, summed over BPE)")
# axes[0].set_title("Energy trajectory per word")
# axes[0].legend(loc="best", fontsize=8, ncol=2)
#
# for i, (r, row) in enumerate(zip(records, dE)):
#     axes[1].plot(layers[1:], row, marker="o", color=cmap[i], label=r["word"])
# axes[1].set_xlabel("layer k")
# axes[1].set_ylabel(r"$|E^{(k)} - E^{(k-1)}|$")
# axes[1].set_title("Layer-to-layer change in E_total per word")
# axes[1].set_yscale("log")
# axes[1].legend(loc="best", fontsize=8, ncol=2)
#
# plt.tight_layout()
# plt.show()
