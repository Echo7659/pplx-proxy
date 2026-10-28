import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import model_catalog
from model_catalog import CATALOG_URL, SESSION_URL, ModelCatalog, cookie_owner
from test_official_models import fixture


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self.payload=payload
        self.status_code=status

    def json(self):
        return copy.deepcopy(self.payload)


class FakeSession:
    def __init__(self, payload=None, catalog_status=200, session_status=200, failure=None):
        self.payload=fixture() if payload is None else payload
        self.catalog_status=catalog_status
        self.session_status=session_status
        self.failure=failure
        self.calls=[]
        self.entered=None
        self.release=None

    async def get(self, url, timeout):
        self.calls.append((url, timeout))
        if self.failure:
            raise self.failure
        if url == SESSION_URL:
            if self.entered:
                self.entered.set()
                await self.release.wait()
            return FakeResponse({"user": {"subscription_status": "active", "subscription_tier": "max"}}, self.session_status)
        if url == CATALOG_URL:
            return FakeResponse(self.payload, self.catalog_status)
        raise AssertionError("Unexpected test URL")


class FakeClient:
    def __init__(self, **session_args):
        self.session=FakeSession(**session_args)
        self.cookies={"__Secure-next-auth.session-token": "test-cookie"}
        self.init_calls=0

    async def init(self):
        self.init_calls+=1


def write_snapshot(path, snapshot):
    temporary=path.with_suffix(".next")
    temporary.write_text(json.dumps(snapshot))
    temporary.replace(path)


class ModelCatalogTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/"catalog.json"
        self.owner=cookie_owner("test-cookie")
        self.manager=ModelCatalog(self.path, self.owner, write_snapshot)

    async def synchronize(self, client=None):
        client=client or FakeClient()
        result=await self.manager.refresh(client, "http://solver:8191")
        self.assertEqual(result["status"], "synced")
        return client

    async def test_refresh_persists_and_reloads_exact_catalog(self):
        client=await self.synchronize()
        self.assertEqual(client.init_calls, 1)
        self.assertEqual([url for url, _ in client.session.calls], [SESSION_URL, CATALOG_URL])
        self.assertTrue(all(timeout == 20 for _, timeout in client.session.calls))
        snapshot=json.loads(self.path.read_text())
        self.assertEqual(snapshot["owner"], self.owner)
        self.assertNotIn("test-cookie", self.path.read_text())
        reloaded=ModelCatalog(self.path, self.owner, write_snapshot)
        self.assertEqual(reloaded.catalog, self.manager.catalog)
        self.assertEqual(reloaded.status(), self.manager.status())
        self.assertEqual(reloaded.account_type, "max")

    async def test_empty_and_malformed_refresh_keep_prior_memory_and_disk(self):
        await self.synchronize()
        previous=copy.deepcopy(self.manager.catalog)
        disk=self.path.read_bytes()
        malformed=fixture()
        malformed["search_config"][0]["reasoning_model"]="missing"
        for payload in [{}, [], {**fixture(), "search_config": []}, malformed]:
            with self.subTest(payload=payload):
                result=await self.manager.refresh(FakeClient(payload=payload), "http://solver:8191")
                self.assertEqual(result["status"], "error")
                self.assertEqual(self.manager.catalog, previous)
                self.assertEqual(self.path.read_bytes(), disk)

    async def test_write_failure_does_not_activate_unpersisted_models(self):
        await self.synchronize()
        previous=copy.deepcopy(self.manager.catalog)
        timestamp=self.manager.updated_at
        disk=self.path.read_bytes()
        replacement=fixture()
        replacement["search_config"]=[replacement["search_config"][1]]

        def fail_write(_path, _snapshot):
            raise OSError("Sensitive filesystem details")

        self.manager.writer=fail_write
        result=await self.manager.refresh(FakeClient(payload=replacement), "http://solver:8191")
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.manager.catalog, previous)
        self.assertEqual(self.manager.updated_at, timestamp)
        self.assertEqual(self.path.read_bytes(), disk)
        self.assertNotIn("Sensitive", result["last_error"])

    async def test_network_failure_preserves_restart_fallback_and_redacts_error(self):
        await self.synchronize()
        previous=copy.deepcopy(self.manager.catalog)
        result=await self.manager.refresh(FakeClient(failure=TimeoutError("token=test-cookie")), "http://solver:8191")
        self.assertEqual(result["status"], "error")
        self.assertIn("TimeoutError", result["last_error"])
        self.assertNotIn("test-cookie", result["last_error"])
        self.assertEqual(self.manager.catalog, previous)
        self.assertEqual(ModelCatalog(self.path, self.owner, write_snapshot).catalog, previous)
        await self.synchronize()
        self.assertIsNone(self.manager.last_error)

    async def test_overlapping_refreshes_share_one_network_attempt(self):
        client=FakeClient()
        client.session.entered=asyncio.Event()
        client.session.release=asyncio.Event()
        first=asyncio.create_task(self.manager.refresh(client, "http://solver:8191"))
        await client.session.entered.wait()
        second=asyncio.create_task(self.manager.refresh(client, "http://solver:8191"))
        await asyncio.sleep(0)
        client.session.release.set()
        results=await asyncio.gather(first, second)
        self.assertEqual(results[0], results[1])
        self.assertEqual(client.init_calls, 1)
        self.assertEqual(len(client.session.calls), 2)
        await self.synchronize(client)
        self.assertEqual(client.init_calls, 2)

    async def test_cookie_owner_mismatch_and_empty_owner_ignore_cache(self):
        await self.synchronize()
        for owner in [cookie_owner("different-cookie"), ""]:
            with self.subTest(owner=owner):
                manager=ModelCatalog(self.path, owner, write_snapshot)
                self.assertIsNone(manager.catalog)
                self.assertEqual(manager.status()["status"], "pending")

    async def test_invalid_cache_shapes_do_not_crash_or_activate_models(self):
        for data in [[], None, "bad", {"version": 1, "owner": self.owner}, {"version": 1, "owner": self.owner, "config": {}, "account_type": "max", "updated_at": 1}]:
            with self.subTest(data=data):
                self.path.write_text(json.dumps(data))
                manager=ModelCatalog(self.path, self.owner, write_snapshot)
                self.assertIsNone(manager.catalog)
                self.assertEqual(manager.status()["status"], "error")

    async def test_aliases_resolve_only_current_canonical_models_and_explicit_thinking(self):
        payload=fixture()
        payload["search_config"][0]["label"]="GPT-9.7"
        await self.synchronize(FakeClient(payload=payload))
        model_map=self.manager.model_map()
        self.assertEqual(model_map["gpt"], ("pro", "fresh_v9"))
        self.assertEqual(model_map["gpt-9.7"], ("pro", "fresh_v9"))
        self.assertEqual(self.manager.metadata("gpt")["thinking_pref"], "fresh_v9_reason")
        self.assertEqual(self.manager.metadata("reason_only")["thinking_pref"], "reason_only")
        self.assertNotIn("best", model_map)
        self.assertNotIn("retired_model", model_map)
        self.assertNotIn("gpt-5.5", model_map)
        payload["search_config"]=payload["search_config"][1:]
        await self.synchronize(FakeClient(payload=payload))
        self.assertNotIn("gpt", self.manager.model_map())
        self.assertNotIn("gpt-9.7", self.manager.model_map())
        self.assertIsNone(self.manager.metadata("gpt-9.7"))

    async def test_generic_alias_never_shadows_a_canonical_id(self):
        payload=fixture()
        payload["search_config"][0]["label"]="GPT-9.7"
        payload["models"]["gpt"]={"label": "Canonical custom", "mode": "search", "provider": "OTHER"}
        payload["search_config"].append({"label": "Canonical custom", "subscription_tier": "pro", "non_reasoning_model": "gpt", "reasoning_model": None})
        await self.synchronize(FakeClient(payload=payload))
        self.assertEqual(self.manager.model_map()["gpt"], ("pro", "gpt"))
        self.assertEqual(self.manager.metadata("gpt")["pref"], "gpt")

    async def test_old_account_refresh_cannot_write_or_dedupe_new_account_refresh(self):
        await self.synchronize()
        writes=[]

        def record_write(path, snapshot):
            writes.append(copy.deepcopy(snapshot))
            write_snapshot(path, snapshot)

        self.manager.writer=record_write
        old_client=FakeClient()
        old_client.session.entered=asyncio.Event()
        old_client.session.release=asyncio.Event()
        old_request=asyncio.create_task(self.manager.refresh(old_client, "http://solver:8191"))
        await old_client.session.entered.wait()
        new_owner=cookie_owner("new-cookie")
        self.manager.owner=new_owner
        self.manager.catalog=None
        self.manager.account_type=None
        self.manager.updated_at=None
        self.manager.last_error=None
        new_payload=fixture()
        new_payload["search_config"]=new_payload["search_config"][1:]
        new_client=FakeClient(payload=new_payload)
        new_client.cookies={"__Secure-next-auth.session-token": "new-cookie"}
        new_request=asyncio.create_task(self.manager.refresh(new_client, "http://solver:8191"))
        await asyncio.sleep(0)
        old_client.session.release.set()
        old_result,new_result=await asyncio.gather(old_request, new_request)
        self.assertEqual(old_result["status"], "pending")
        self.assertEqual(new_result["status"], "synced")
        self.assertEqual(new_client.init_calls, 1)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["owner"], new_owner)
        self.assertEqual(json.loads(self.path.read_text())["owner"], new_owner)
        self.assertNotIn("fresh_v9", self.manager.catalog["models"])
        self.assertEqual(self.manager.last_attempt_owner, new_owner)

    async def test_403_uses_mocked_browser_fallback(self):
        thread=AsyncMock(return_value=fixture())
        client=FakeClient(catalog_status=403)
        with patch.object(model_catalog.asyncio, "to_thread", thread), patch.object(model_catalog, "fetch_browser_catalog") as browser:
            await self.synchronize(client)
        thread.assert_awaited_once_with(browser, client.cookies, "http://solver:8191")
        browser.assert_not_called()

    async def test_browser_failure_preserves_previous_snapshot(self):
        await self.synchronize()
        previous=copy.deepcopy(self.manager.catalog)
        with patch.object(model_catalog.asyncio, "to_thread", AsyncMock(side_effect=ValueError("challenge"))):
            result=await self.manager.refresh(FakeClient(catalog_status=403), "http://solver:8191")
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.manager.catalog, previous)

    async def test_auth_failure_never_attempts_catalog_or_browser(self):
        client=FakeClient(session_status=401)
        with patch.object(model_catalog.asyncio, "to_thread", AsyncMock()) as thread:
            result=await self.manager.refresh(client, "http://solver:8191")
        self.assertEqual(result["status"], "error")
        self.assertEqual([url for url, _ in client.session.calls], [SESSION_URL])
        thread.assert_not_awaited()
        self.assertIsNone(self.manager.catalog)


if __name__ == "__main__":
    unittest.main()
