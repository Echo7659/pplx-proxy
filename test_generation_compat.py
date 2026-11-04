"""Client generation settings are validated and dropped before website requests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import server


CONTROLS={
    "temperature": 0.7, "top_p": 0.9,
    "presence_penalty": 0.4, "frequency_penalty": -0.5,
    "max_tokens": 512, "max_completion_tokens": 1024, "max_output_tokens": 2048,
}
PROMPT="Explain temperature and max_tokens as API fields."
ANSWER="The website decides sampling and output length."


class WebsiteResponse:
    status_code=200

    async def aclose(self):
        pass

    async def aiter_lines(self, delimiter):
        for progress in ("IN_PROGRESS", "DONE"):
            payload={"blocks": [{"intended_usage": "ask_text", "markdown_block": {
                "progress": progress, "chunks": [ANSWER]}}]}
            yield "event: message"
            yield "data: "+json.dumps(payload)
            yield ""
        yield "event: end_of_stream"
        yield ""


class WebsiteSession:
    def __init__(self):
        self.payloads=[]

    async def post(self, url, **kwargs):
        self.payloads.append(kwargs["json"])
        return WebsiteResponse()

    async def close(self):
        pass


class GenerationCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.website=WebsiteSession()
        self.client=server.PerplexityClient({})
        self.client.session=self.website
        self.client._initialized=True
        server._responses_reset_memory()
        self.patches=[
            patch.object(server, "API_KEY", ""),
            patch.object(server, "_official_catalog", None),
            patch.object(server, "MODEL_MAP", {"auto": ("pro", "pplx_pro")}),
            patch.object(server, "get_client", return_value=self.client),
            patch.object(server, "_session_cache", {}),
            patch.object(server, "_response_suffix", return_value=""),
            patch.object(server, "_rate_limit", {"remaining_pro": None, "updated_at": 0, "last_error": None}),
            patch.object(server, "RESPONSES_FILE", Path(self.temp.name)/"responses.json"),
        ]
        for item in self.patches:
            item.start()
        self.http=httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.http.aclose()
        await self.client.close()
        for item in reversed(self.patches):
            item.stop()
        server._responses_reset_memory()
        self.temp.cleanup()

    def request_body(self, path, **fields):
        source={"messages": [{"role": "user", "content": PROMPT}]} if path.endswith("completions") else {"input": PROMPT, "store": False}
        return {"model": "auto", **source, **fields}

    def assert_clean_website_payload(self):
        payload=self.website.payloads[-1]
        self.assertTrue(CONTROLS.keys().isdisjoint(payload))
        self.assertTrue(CONTROLS.keys().isdisjoint(payload["params"]))
        self.assertEqual(payload["params"]["search_focus"], "internet")
        self.assertEqual(payload["params"]["model_preference"], "pplx_pro")
        self.assertEqual(json.loads(payload["query_str"])["query"], PROMPT)

    async def test_chat_settings_are_ignored_in_complete_and_streaming_website_requests(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                response=await self.http.post("/v1/chat/completions", json=self.request_body("/v1/chat/completions", stream=stream, **CONTROLS))
                self.assertEqual(response.status_code, 200, response.text)
                if stream:
                    self.assertIn("[DONE]", response.text)
                    self.assertNotIn('"error"', response.text)
                else:
                    choice=response.json()["choices"][0]
                    self.assertEqual(choice["finish_reason"], "stop")
                    self.assertEqual(choice["message"]["content"], ANSWER)
                self.assert_clean_website_payload()
        self.assertEqual(len(self.website.payloads), 2)

    async def test_responses_settings_are_dropped_before_output_metadata_and_streaming(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                response=await self.http.post("/v1/responses", json=self.request_body("/v1/responses", stream=stream, **CONTROLS))
                self.assertEqual(response.status_code, 200, response.text)
                if stream:
                    completed=[json.loads(line[6:])["response"] for line in response.text.splitlines()
                               if line.startswith("data: ") and json.loads(line[6:]).get("type") == "response.completed"]
                    self.assertEqual(len(completed), 1, response.text)
                    record=completed[0]
                else:
                    record=response.json()
                self.assertEqual(record["status"], "completed")
                self.assertEqual(record["output_text"], ANSWER)
                self.assertEqual(record["temperature"], 1)
                self.assertEqual(record["top_p"], 1)
                self.assertIsNone(record["max_output_tokens"])
                self.assert_clean_website_payload()

    async def test_null_settings_are_accepted_without_leaking_to_upstream(self):
        for path in ("/v1/chat/completions", "/v1/responses"):
            with self.subTest(path=path):
                response=await self.http.post(path, json=self.request_body(path, **dict.fromkeys(CONTROLS)))
                self.assertEqual(response.status_code, 200, response.text)
                self.assert_clean_website_payload()

    async def test_invalid_ignored_settings_still_fail_before_upstream(self):
        for path in ("/v1/chat/completions", "/v1/responses"):
            for name in CONTROLS:
                invalid=[True, "0.7", [], {}, float("nan"), float("inf")]
                if name.startswith("max_"):
                    invalid.extend([0, -1, 1.5])
                for value in invalid:
                    with self.subTest(path=path, name=name, value=value):
                        response=await self.http.post(path, content=json.dumps(self.request_body(path, **{name: value})), headers={"Content-Type": "application/json"})
                        self.assertEqual(response.status_code, 400, response.text)
                        self.assertEqual(response.json()["error"]["param"], name)
        self.assertEqual(self.website.payloads, [])

    async def test_responses_function_branch_receives_normalized_settings(self):
        tool={"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}
        with patch.object(server, "_responses_tools_api", AsyncMock(return_value={"status": "completed"})) as tools:
            response=await self.http.post("/v1/responses", json=self.request_body("/v1/responses", tools=[tool], **CONTROLS))
        self.assertEqual(response.status_code, 200, response.text)
        tools.assert_awaited_once()
        self.assertTrue(CONTROLS.keys().isdisjoint(tools.await_args.args[0]))
        self.assertEqual(self.website.payloads, [])


if __name__ == "__main__":
    unittest.main()
