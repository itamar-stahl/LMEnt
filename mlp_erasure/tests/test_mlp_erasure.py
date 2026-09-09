"""Fast tests for the two MLP erasure scripts.

    python -m unittest discover -s mlp_erasure/tests -p 'test_*.py' -v

No checkpoint and no GPU: everything here runs against a 6-layer Olmo2 built
from a config, which is enough to pin the two things that were actually wrong
about OLMo-2 -- where down_proj sits, and how deep the layer bands are.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (Olmo2Config, Olmo2ForCausalLM,
                          PreTrainedTokenizerFast)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import rmu    # noqa: E402
import snmf   # noqa: E402


def tiny_olmo2(n_layers=6, hidden=32, mlp=88):
    torch.manual_seed(0)
    return Olmo2ForCausalLM(Olmo2Config(
        vocab_size=64, hidden_size=hidden, intermediate_size=mlp,
        num_hidden_layers=n_layers, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=32,
        tie_word_embeddings=False))


def tiny_tokenizer():
    """A throwaway tokenizer, so the round-trip test needs no checkpoint.

    load_model() loads a tokenizer beside the weights, and a directory holding
    only safetensors makes AutoTokenizer try every slow->fast converter it has
    before failing.
    """
    tok = Tokenizer(models.WordLevel(
        vocab={f"t{i}": i for i in range(64)}, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="t0",
                                   eos_token="t1", pad_token="t2")


class LayerSelection(unittest.TestCase):
    def test_snmf_ranges_reproduce_the_published_gemma_indices(self):
        # ember/erasure/methods/snmf.py records Gemma-2-2B (26 layers) as
        # in=(0,25) and out=(0,8). The fractions have to give those back.
        self.assertEqual(snmf.default_layer_ranges(26), ((0, 25), (0, 8)))

    def test_snmf_ranges_on_the_lment_1b(self):
        # 18 layers, from lment-1b-control-2e-b131k/config.json. The old
        # literals were (0,15) and (0,7).
        self.assertEqual(snmf.default_layer_ranges(18), ((0, 17), (0, 5)))

    def test_snmf_ranges_never_leave_the_model(self):
        for n in range(2, 65):
            (lo_in, hi_in), (lo_out, hi_out) = snmf.default_layer_ranges(n)
            self.assertEqual((lo_in, hi_in), (0, n - 1))
            self.assertTrue(0 <= lo_out <= hi_out <= n - 1, n)

    def test_resolve_ranges_prefers_the_cli(self):
        got = snmf.resolve_ranges(18, [3, 4], None)
        self.assertEqual(got, ((3, 4), (0, 5)))

    def test_rmu_bands_sit_in_the_published_depth_window(self):
        for n in (18, 26, 32):
            for layer_id, layer_ids in rmu.layers_by_depth(n):
                self.assertTrue(0.20 <= layer_id / n <= 0.36, (n, layer_id))
                self.assertEqual(layer_ids, [layer_id - 2, layer_id - 1, layer_id])

    def test_rmu_bands_on_the_lment_1b(self):
        self.assertEqual(rmu.layers_by_depth(18),
                         [(4, [2, 3, 4]), (5, [3, 4, 5]), (6, [4, 5, 6])])


class DownProjLookup(unittest.TestCase):
    """The comment in rmu.py that WMDP's param_ids=[6] misses OLMo-2."""

    def setUp(self):
        self.model = tiny_olmo2()

    def test_selects_down_proj_by_name(self):
        got = rmu.down_proj_weights(self.model, [2])[0]
        self.assertIs(got, self.model.model.layers[2].mlp.down_proj.weight)

    def test_positional_param_id_6_is_wrong_for_olmo2(self):
        report = rmu.param_index_report(self.model)
        self.assertFalse(report["positional_ok"])
        self.assertNotEqual(report["down_proj_index"], 6)


class MaskedLoss(unittest.TestCase):
    def test_matches_plain_mse_when_nothing_is_padded(self):
        left = torch.randn(3, 5, 8)
        right = torch.randn(3, 5, 8)
        mask = torch.ones(3, 5, dtype=torch.long)
        self.assertAlmostEqual(
            rmu.masked_mse(left, right, mask).item(),
            torch.nn.functional.mse_loss(left, right).item(), places=5)

    def test_pad_positions_do_not_reach_the_loss(self):
        left = torch.zeros(1, 4, 2)
        right = torch.zeros(1, 4, 2)
        left[0, 2:] = 100.0          # only the padded tail disagrees
        mask = torch.tensor([[1, 1, 0, 0]])
        self.assertEqual(rmu.masked_mse(left, right, mask).item(), 0.0)
        self.assertGreater(
            torch.nn.functional.mse_loss(left, right).item(), 0.0)


class RhoSummary(unittest.TestCase):
    def test_counts_use_the_same_comparison_as_selection(self):
        rho = torch.tensor([0.5, 2.0, 2.5, 3.0])
        got = snmf.rho_summary(rho, 2.0)
        # cmd_factorize selects on rho > tau, so 2.0 itself is not a candidate
        self.assertEqual(got["n_gt_tau"], 2)
        self.assertEqual(got["max"], 3.0)
        self.assertEqual(got["n_features"], 4)
        self.assertEqual(got["n_ge"]["1.5"], 3)


class AblateLayer(unittest.TestCase):
    def test_edits_both_projections_and_leaves_the_rest_alone(self):
        model = tiny_olmo2()
        mlp = model.model.layers[1].mlp
        before = {name: w.detach().clone()
                  for name, w in model.named_parameters()}
        d_mlp = model.config.intermediate_size
        Z = torch.zeros(d_mlp, 2)
        Z[:8, 0] = torch.randn(8)        # one sparse feature, support of 8
        n, kept = snmf.ablate_layer(model, 1, Z, [0], 4.0, 4.0)
        self.assertEqual(n, 1)
        self.assertGreater(kept, 0)
        moved = {name for name, w in model.named_parameters()
                 if not torch.equal(w.detach(), before[name])}
        self.assertEqual(moved, {"model.layers.1.mlp.up_proj.weight",
                                 "model.layers.1.mlp.down_proj.weight"})

    def test_zero_deltas_change_nothing(self):
        model = tiny_olmo2()
        before = {name: w.detach().clone()
                  for name, w in model.named_parameters()}
        Z = torch.zeros(model.config.intermediate_size, 1)
        Z[:8, 0] = torch.randn(8)
        snmf.ablate_layer(model, 1, Z, [0], 0.0, 0.0)
        for name, w in model.named_parameters():
            self.assertTrue(torch.equal(w.detach(), before[name]), name)


class SaveComparability(unittest.TestCase):
    """The reason both scripts default to fp32.

    compare_weights.py's whole job is showing that an erasure moved the
    matrices it claims to move and nothing else. That only works if the save
    round-trips every untouched tensor.
    """

    def _round_trip(self, dtype):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src, dest = tmp / "src", tmp / "dest"
            tiny_olmo2().to(torch.float32).save_pretrained(src)
            tiny_tokenizer().save_pretrained(src)
            base = load_file(src / "model.safetensors")

            model, _ = snmf.load_model(str(src), None, snmf.DTYPES[dtype], "cpu")
            Z = torch.zeros(model.config.intermediate_size, 1)
            Z[:8, 0] = torch.randn(8)
            snmf.ablate_layer(model, 1, Z, [0], 4.0, 4.0)
            model.save_pretrained(dest)

            got = load_file(dest / "model.safetensors")
            return base, {name for name, w in base.items()
                          if not torch.equal(w, got[name].to(w.dtype))}

    def test_fp32_moves_only_the_edited_matrices(self):
        base, differ = self._round_trip("fp32")
        self.assertEqual(differ, {"model.layers.1.mlp.up_proj.weight",
                                  "model.layers.1.mlp.down_proj.weight"})

    def test_bf16_moves_most_of_the_model(self):
        base, differ = self._round_trip("bf16")
        self.assertGreater(len(differ), 10, "bf16 should be visibly lossy")
        self.assertIn("model.embed_tokens.weight", differ)


class DeltaScalingLaw(unittest.TestCase):
    """ablate_layer applies (I - delta*P), so the component scales by |1-delta|.

    This is why job 871607 reported a -183.8% "drop" at delta 4: the metric
    takes .abs() of the component, so its reported drop is 1 - |1-delta|, which
    is -200% at delta 4. Negative drops above delta 2 are arithmetic, not a
    transpose bug, and raising delta makes them worse. Pinned here so the
    relationship cannot drift silently.
    """

    def _ratio(self, delta):
        torch.manual_seed(0)
        model = tiny_olmo2()
        d_mlp = model.config.intermediate_size
        Z = torch.zeros(d_mlp, 1)
        Z[:8, 0] = torch.randn(8)
        mlp = snmf.mlp_of(snmf.get_layers(model)[1])
        W_in = mlp.up_proj.weight.data.T
        z = Z[:, 0]
        support = z != 0
        f = W_in @ z
        f = f / f.norm()
        before = (f @ W_in)[support].norm().item()
        snmf.ablate_layer(model, 1, Z, [0], delta, 0.0)
        after = (f @ mlp.up_proj.weight.data.T)[support].norm().item()
        return after / before

    def test_delta_one_removes_the_component(self):
        self.assertLess(self._ratio(1.0), 0.2)

    def test_delta_two_preserves_magnitude(self):
        self.assertAlmostEqual(self._ratio(2.0), 1.0, places=2)

    def test_delta_four_triples_it(self):
        self.assertAlmostEqual(self._ratio(4.0), 3.0, delta=0.1)

    def test_the_law_holds_across_deltas(self):
        for delta in (0.0, 0.5, 1.5, 2.0, 4.0):
            self.assertAlmostEqual(self._ratio(delta), abs(1.0 - delta),
                                   delta=0.05, msg=f"delta={delta}")


class OneSidedErasureGuard(unittest.TestCase):
    """Reproduces job 871388: the judge kept features only outside the output band.

    On 18 layers the output range is [0,5]. Selection accepted 0 of layer 4's
    candidates and 13 across layers 9 and 14, so every selected feature sat
    outside [0,5]. `total` was 13, the total==0 guard stayed quiet, and the
    erase would have applied delta_in at two layers and delta_out nowhere.
    """

    RANGES = ((0, 17), (0, 5))
    SELECTED_871388 = {"4": [], "9": [6, 41, 42, 46, 67, 82],
                       "14": [0, 4, 31, 41, 56, 83, 90]}

    def test_requesting_output_side_and_getting_none_is_fatal(self):
        with self.assertRaises(SystemExit) as cm:
            snmf.check_both_sides_applied(
                applied_in=2, applied_out=0, delta_in=4.0, delta_out=4.0,
                ranges=self.RANGES, selected=self.SELECTED_871388, cmd="erase")
        msg = str(cm.exception)
        self.assertIn("down_proj", msg)
        self.assertIn("[0,5]", msg)
        self.assertIn("[9, 14]", msg)

    def test_deliberate_one_sided_edit_is_allowed(self):
        snmf.check_both_sides_applied(
            applied_in=2, applied_out=0, delta_in=4.0, delta_out=0.0,
            ranges=self.RANGES, selected=self.SELECTED_871388, cmd="erase")

    def test_both_sides_applied_passes(self):
        snmf.check_both_sides_applied(
            applied_in=2, applied_out=2, delta_in=4.0, delta_out=4.0,
            ranges=self.RANGES, selected=self.SELECTED_871388, cmd="erase")

    def test_missing_input_side_is_also_caught(self):
        with self.assertRaises(SystemExit) as cm:
            snmf.check_both_sides_applied(
                applied_in=0, applied_out=1, delta_in=4.0, delta_out=4.0,
                ranges=self.RANGES, selected=self.SELECTED_871388, cmd="verify")
        self.assertIn("up_proj", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
