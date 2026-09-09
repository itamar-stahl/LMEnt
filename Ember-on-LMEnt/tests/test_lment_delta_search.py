import unittest

from ember.lment_pipeline import _paired_margin_report, search_deltas


class LMEntDeltaSearchTests(unittest.TestCase):
    def test_paired_margin_report_preserves_question_level_deltas(self) -> None:
        baseline = {
            "qa": {
                "records": [
                    {"question": "q1", "correct_letter": "A", "margin": 0.4},
                    {"question": "q2", "correct_letter": "B", "margin": -0.1},
                ],
            },
        }
        edited = {
            "qa": {
                "records": [
                    {"question": "q1", "correct_letter": "A", "margin": 0.1},
                    {"question": "q2", "correct_letter": "B", "margin": -0.2},
                ],
            },
        }

        paired = _paired_margin_report(baseline, edited)

        self.assertEqual(paired["qa"]["n"], 2)
        self.assertAlmostEqual(paired["qa"]["mean_delta"], -0.2)
        first = paired["qa"]["records"][0]
        self.assertEqual({key: first[key] for key in (
            "question", "correct_letter", "baseline_margin", "edited_margin",
        )}, {
            "question": "q1",
            "correct_letter": "A",
            "baseline_margin": 0.4,
            "edited_margin": 0.1,
        })
        self.assertAlmostEqual(first["delta"], -0.3)

    def test_each_delta_starts_pristine_and_ties_choose_the_smaller_delta(self) -> None:
        state = {"weight": -1.0}
        apply_starts = []
        evaluated = []

        def restore_pristine():
            state["weight"] = 10.0

        def apply_delta(delta):
            apply_starts.append(state["weight"])
            state["weight"] -= delta
            return {"edited_token_ids": [1]}

        def evaluate_train():
            delta = 10.0 - state["weight"]
            evaluated.append(delta)
            if delta in (0.5, 1.0):
                return {
                    "qa": {"accuracy": 0.25, "mean_margin": -0.2},
                    "simdom": {"accuracy": 0.50, "mean_margin": 0.1},
                }
            return {
                "qa": {"accuracy": 0.50, "mean_margin": 0.2},
                "simdom": {"accuracy": 0.25, "mean_margin": -0.1},
            }

        result = search_deltas(
            deltas=[2.0, 1.0, 0.5],
            baseline={
                "qa": {"accuracy": 0.50, "mean_margin": 0.3},
                "simdom": {"accuracy": 0.50, "mean_margin": 0.3},
            },
            restore_pristine=restore_pristine,
            apply_delta=apply_delta,
            evaluate_train=evaluate_train,
        )

        self.assertEqual(apply_starts, [10.0, 10.0, 10.0])
        self.assertEqual(evaluated, [2.0, 1.0, 0.5])
        self.assertEqual(result.chosen_delta, 0.5)
        self.assertEqual(state["weight"], 10.0)
        self.assertEqual(len(result.candidates), 3)


if __name__ == "__main__":
    unittest.main()
