"""Exercise the website catalog through the existing API/model resolution paths."""

import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

import server
from model_catalog import ModelCatalog, cookie_owner
from official_models import parse_catalog
from test_official_models import fixture


class CatalogIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.patches=ExitStack()
        payload=fixture()
        payload["search_config"][0]["label"]="GPT-9.7"
        self.catalog=ModelCatalog(Path(self.directory.name)/"catalog.json", cookie_owner("test-cookie"), Mock())
        self.catalog.catalog=parse_catalog(payload, "max")
        self.catalog.account_type="max"
        self.catalog.updated_at=1
        self.legacy={"retired_model": ("pro", "retired_pref"), "gpt-5.5": ("pro", "retired_gpt")}
        for name, value in {
            "_official_catalog": self.catalog,
            "ACCOUNT_TYPE": "pro",
            "API_KEY": "",
            "MODEL_MAP": self.legacy,
            "_rate_limit": {"remaining_pro": None, "updated_at": 0, "last_error": None},
        }.items():
            self.patches.enter_context(patch.object(server, name, value))
        self.http=httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.http.aclose()
        self.patches.close()
        self.directory.cleanup()

    async def test_public_models_use_canonical_current_ids_and_website_labels(self):
        response=await self.http.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        body=response.json()
        self.assertEqual(body["object"], "list")
        models={entry["id"]: entry for entry in body["data"]}
        self.assertEqual(set(models), {"auto", "fresh_v9", "reason_only", "max_model"})
        self.assertEqual(models["fresh_v9"]["name"], "GPT-9.7")
        self.assertEqual(models["fresh_v9"]["internal_pref"], "fresh_v9")
        self.assertTrue(models["fresh_v9"]["thinking_supported"])
        self.assertTrue(models["reason_only"]["text_only"])
        self.assertFalse(models["max_model"]["thinking_supported"])
        self.assertNotIn("gpt", models)
        self.assertNotIn("gpt-9.7", models)

    async def test_aliases_target_current_models_and_detected_tier_overrides_manual_tier(self):
        model_map=server.get_model_map()
        self.assertEqual(model_map["gpt"], ("pro", "fresh_v9"))
        self.assertEqual(model_map["gpt-9.7"], ("pro", "fresh_v9"))
        self.assertEqual(model_map["max_model"], ("pro", "max_model"))
        self.assertNotIn("retired_model", model_map)
        self.assertNotIn("gpt-5.5", model_map)
        self.assertEqual(server.ACCOUNT_TYPE, "pro")
        self.assertEqual(server.check_tier("max_model"), "")
        admin=await self.http.get("/admin/models")
        self.assertEqual(admin.status_code, 200)
        self.assertEqual(admin.json()["account_type"], "max")

    async def test_responses_thinking_uses_official_preferences_for_canonical_ids_and_aliases(self):
        for model in ["fresh_v9", "gpt", "gpt-9.7"]:
            with self.subTest(model=model):
                self.assertEqual(server._responses_resolve_model(model, True), (model, "pro", "fresh_v9_reason"))
                self.assertEqual(server._responses_resolve_model(model, False), (model, "pro", "fresh_v9"))
        for thinking in [False, True]:
            with self.subTest(thinking=thinking):
                self.assertEqual(server._responses_resolve_model("reason_only", thinking), ("reason_only", "pro", "reason_only"))
        self.assertEqual(server._responses_resolve_model("auto", True), ("auto", "auto", "pplx_pro"))
        with self.assertRaises(server.OpenAIAPIError) as error:
            server._responses_resolve_model("retired_model", False)
        self.assertEqual(error.exception.status_code, 400)

    async def test_manual_discovery_synchronizes_catalog_without_guessing_or_generation_probes(self):
        synchronize=AsyncMock(return_value=self.catalog.status())
        with patch.object(server, "_sync_official_models", synchronize), patch.object(server, "get_client") as client, patch.object(server, "probe_model_status", AsyncMock()) as probe:
            response=await self.http.post("/admin/discover-models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["catalog"]["model_count"], 4)
        self.assertIsNone(response.json()["catalog"]["last_error"])
        synchronize.assert_awaited_once_with()
        client.assert_not_called()
        probe.assert_not_awaited()

    async def test_manual_update_is_explicitly_rejected_without_changing_either_catalog(self):
        previous_catalog=copy.deepcopy(self.catalog.catalog)
        previous_legacy=copy.deepcopy(server.MODEL_MAP)
        with patch.object(server, "save_model_map") as save:
            response=await self.http.post("/admin/update-models", json={"models": {"new": ["pro", "new_pref"]}, "merge": False})
        self.assertEqual(response.status_code, 409)
        self.assertIn("MODEL_CATALOG_SOURCE=legacy", response.json()["error"]["message"])
        self.assertEqual(self.catalog.catalog, previous_catalog)
        self.assertEqual(server.MODEL_MAP, previous_legacy)
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
