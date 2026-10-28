import copy
import unittest

from official_models import derive_account_type, parse_catalog


def fixture():
    return {
        "config_schema": "v2",
        "organization_model_policy": None,
        "models": {
            "pplx_pro": {"label": "Best", "mode": "search", "provider": None},
            "fresh_v9": {"label": "Fresh 9", "mode": "search", "provider": "NEW_PROVIDER"},
            "fresh_v9_reason": {"label": "Fresh 9 Thinking", "mode": "search", "provider": "NEW_PROVIDER"},
            "reason_only": {"label": "Reasoner", "mode": "search", "provider": "OTHER"},
            "max_model": {"label": "Max model", "mode": "search", "provider": "OTHER"},
            "browser_model": {"label": "Browser", "mode": "browser_agent", "provider": "OTHER"},
            "retired_model": {"label": "Retired", "mode": "search", "provider": "OTHER"},
        },
        "default_models": {"search": "pplx_pro"},
        "search_config": [
            {"label": "Fresh 9", "subscription_tier": "pro", "non_reasoning_model": "fresh_v9", "reasoning_model": "fresh_v9_reason", "text_only_model": False, "audience": None},
            {"label": "Reasoner", "subscription_tier": "pro", "non_reasoning_model": None, "reasoning_model": "reason_only", "text_only_model": True, "audience": None},
            {"label": "Max model", "subscription_tier": "max", "non_reasoning_model": "max_model", "reasoning_model": None, "audience": None},
            {"label": "Browser", "subscription_tier": "pro", "non_reasoning_model": "browser_model", "reasoning_model": None, "audience": None},
        ],
    }


class OfficialCatalogTests(unittest.TestCase):
    def test_picker_controls_membership_not_complete_metadata_registry(self):
        catalog=parse_catalog(fixture())
        self.assertEqual(list(catalog["models"]), ["auto", "fresh_v9", "reason_only"])
        self.assertEqual(catalog["default"], "pplx_pro")
        self.assertEqual(catalog["models"]["auto"]["pref"], "pplx_pro")
        self.assertEqual(catalog["models"]["fresh_v9"]["provider"], "NEW_PROVIDER")
        self.assertEqual(catalog["models"]["fresh_v9"]["thinking_pref"], "fresh_v9_reason")

    def test_reasoning_only_models_keep_actual_preference_and_text_only_marker(self):
        model=parse_catalog(fixture())["models"]["reason_only"]
        self.assertEqual(model["pref"], "reason_only")
        self.assertEqual(model["thinking_pref"], "reason_only")
        self.assertTrue(model["text_only"])

    def test_account_tier_filters_locked_options(self):
        self.assertEqual(list(parse_catalog(fixture(), "free")["models"]), ["auto"])
        self.assertIn("max_model", parse_catalog(fixture(), "max")["models"])

    def test_removed_picker_option_disappears_even_when_metadata_remains(self):
        payload=fixture()
        payload["search_config"]=payload["search_config"][1:]
        self.assertNotIn("fresh_v9", parse_catalog(payload)["models"])

    def test_backend_supplied_audience_is_not_a_client_availability_gate(self):
        payload=fixture()
        payload["search_config"][0]["audience"]="internal"
        self.assertIn("fresh_v9", parse_catalog(payload)["models"])

    def test_unknown_tier_is_not_assumed_to_be_available(self):
        payload=fixture()
        payload["search_config"][0]["subscription_tier"]="future_paid_tier"
        self.assertNotIn("fresh_v9", parse_catalog(payload, "max")["models"])

    def test_unrecognized_fast_or_speed_variants_reject_partial_catalog(self):
        for field,value in [("fast_model", "future_fast_model"), ("speed_options", [{"model": "future_fast_model"}])]:
            for location in ["root", "option"]:
                with self.subTest(field=field, location=location):
                    payload=fixture()
                    target=payload if location == "root" else payload["search_config"][0]
                    target[field]=value
                    with self.assertRaisesRegex(ValueError, "fast/speed"):
                        parse_catalog(payload)

    def test_empty_fast_or_speed_markers_do_not_change_current_catalog(self):
        payload=fixture()
        payload["search_config"][0].update(fast_model=None, speed_options=[])
        self.assertEqual(parse_catalog(payload), parse_catalog(fixture()))

    def test_identical_duplicate_picker_entries_are_deduplicated(self):
        payload=fixture()
        payload["search_config"].append(copy.deepcopy(payload["search_config"][0]))
        self.assertEqual(len(parse_catalog(payload)["models"]), 3)

    def test_conflicting_duplicate_picker_entries_reject_refresh(self):
        payload=fixture()
        duplicate=copy.deepcopy(payload["search_config"][0])
        duplicate["reasoning_model"]=None
        payload["search_config"].append(duplicate)
        with self.assertRaises(ValueError):
            parse_catalog(payload)

    def test_invalid_replacements_raise_without_mutating_input(self):
        mutations=[
            lambda p: p.update(config_schema="v3"),
            lambda p: p.update(organization_model_policy={"unknown": True}),
            lambda p: p.update(models=[]),
            lambda p: p.update(search_config=[]),
            lambda p: p.update(search_config=[None]),
            lambda p: p.update(default_models={}),
            lambda p: p["default_models"].update(search="missing"),
            lambda p: p["default_models"].update(search="browser_model"),
            lambda p: p["search_config"][0].update(non_reasoning_model="missing"),
            lambda p: p["search_config"][0].update(reasoning_model="missing"),
            lambda p: p["search_config"][0].update(reasoning_model="browser_model"),
            lambda p: p["search_config"][0].update(text_only_model="false"),
            lambda p: p["search_config"][0].update(non_reasoning_model=None, reasoning_model=None),
            lambda p: p["search_config"][0].update(non_reasoning_model="bad\nmodel"),
            lambda p: p["search_config"][0].update(label=""),
            lambda p: p["models"]["fresh_v9"].update(provider=[]),
        ]
        for mutate in mutations:
            payload=fixture()
            mutate(payload)
            before=copy.deepcopy(payload)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                parse_catalog(payload)
            self.assertEqual(payload, before)

    def test_no_supported_search_options_rejects_replacement(self):
        payload=fixture()
        payload["search_config"]=[payload["search_config"][-1]]
        with self.assertRaises(ValueError):
            parse_catalog(payload)

    def test_bounds_reject_unexpectedly_large_picker(self):
        payload=fixture()
        payload["search_config"] *= 65
        with self.assertRaises(ValueError):
            parse_catalog(payload)

    def test_input_and_account_type_validation(self):
        for payload in [None, [], "v2"]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_catalog(payload)
        for tier in ["enterprise", [], None]:
            with self.subTest(tier=tier), self.assertRaises(ValueError):
                parse_catalog(fixture(), tier)


class AccountTypeTests(unittest.TestCase):
    def test_paid_subscription_tiers(self):
        for status in ["active", "trialing", "white_glove_past_due"]:
            with self.subTest(status=status):
                self.assertEqual(derive_account_type({"subscription_status": status, "subscription_tier": "max"}), "max")
                self.assertEqual(derive_account_type({"subscription_status": status, "subscription_tier": "pro"}), "pro")
                self.assertEqual(derive_account_type({"subscription_status": status, "subscription_tier": "viewer"}), "free")

    def test_inactive_or_missing_subscription_is_free(self):
        for user in [{}, {"subscription_tier": "max"}, {"subscription_status": "canceled", "subscription_tier": "max"}]:
            with self.subTest(user=user):
                self.assertEqual(derive_account_type(user), "free")
        with self.assertRaises(ValueError):
            derive_account_type(None)
        with self.assertRaises(ValueError):
            derive_account_type({"subscription_status": []})
        with self.assertRaises(ValueError):
            derive_account_type({"subscription_tier": {}})


if __name__ == "__main__":
    unittest.main()
