import unittest

from arc.transformer_archspecs import archspec2_list, archspec_list, archspecs, archspecs2


class TestArchSpecsIntegrity(unittest.TestCase):

    def test_archspec_keys_consistency(self):
        """Ensure every archspec has the 4 required keys for TransformerX."""
        required_keys = {
            "positional_encoding",
            "layer_method",
            "norm_post_layers",
            "aggr_attention_method",
        }

        for name, spec in archspecs.items():
            with self.subTest(archspec=name):
                # Check that all keys exist
                self.assertEqual(
                    set(spec.keys()), required_keys, f"{name} is missing keys or has extra ones."
                )

                # Check that values are valid types (Enum-like check)
                self.assertIn(spec["positional_encoding"], ["rope", None])
                self.assertIn(spec["layer_method"], ["torch_transformer", None])
                self.assertIsInstance(spec["norm_post_layers"], bool)
                self.assertIn(spec["aggr_attention_method"], ["view", "sum", "cls_token"])

    def test_archspec2_logic(self):
        """Ensure MLP configurations follow the naming convention (L{depth}_{norm})."""
        valid_norms = ["bn", "ln", "nn"]
        valid_calls = ["L4", "L3", "L2", "L1"]

        for name, spec in archspecs2.items():
            with self.subTest(archspec2=name):
                self.assertIn(spec["call"], valid_calls)
                self.assertIn(spec["n"], valid_norms)

                # Logic check: Does the key 'L4_bn' match the content {'call': 'L4', 'n': 'bn'}?
                expected_key = f"{spec['call']}_{spec['n']}"
                self.assertEqual(
                    name, expected_key, f"Key name '{name}' does not match content '{expected_key}'"
                )

    def test_list_sync(self):
        """Verify that the helper lists are in sync with the actual dictionary keys."""
        # Check archspec_list
        self.assertEqual(
            set(archspec_list),
            set(archspecs.keys()),
            "archspec_list is out of sync with archspecs dictionary.",
        )

        # Check archspec2_list
        self.assertEqual(
            set(archspec2_list),
            set(archspecs2.keys()),
            "archspec2_list is out of sync with archspecs2 dictionary.",
        )

    def test_no_duplicate_configs(self):
        """Ensure no two named specs have identical configurations (redundancy check)."""
        seen_configs = []
        for name, spec in archspecs.items():
            # Convert dict to a frozenset of items to make it hashable/comparable
            config_tuple = tuple(sorted(spec.items()))
            self.assertNotIn(
                config_tuple, seen_configs, f"Redundant configuration found: {name} is a duplicate."
            )
            seen_configs.append(config_tuple)


if __name__ == "__main__":
    unittest.main()
