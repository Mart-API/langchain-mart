import asyncio
import os
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.tools import ToolException
from pydantic import ValidationError
from langchain_mart import MartToolkit


PERSON = "https://www.linkedin.com/in/example-person"
COMPANY = "https://www.linkedin.com/company/example-company"
KEY = "fixture-key-not-a-secret"
CLIENT = httpx.Client
ASYNC_CLIENT = httpx.AsyncClient


class MartToolkitTests(unittest.TestCase):
    def setUp(self):
        self.toolkit = MartToolkit(api_key=KEY)
        self.tools = {tool.name: tool for tool in self.toolkit.get_tools()}

    def client(self, handler):
        return patch("langchain_mart.toolkit.httpx.Client", side_effect=lambda **kwargs: CLIENT(
            transport=httpx.MockTransport(handler), **kwargs))

    def test_four_operations_have_correct_routes_and_headers(self):
        routes = [("mart_enrich_person", "profile", PERSON), ("mart_enrich_company", "company", COMPANY),
                  ("mart_refresh_contact", "refresh", PERSON), ("mart_get_posts", "posts", PERSON)]
        for name, operation, url in routes:
            seen = []
            def handler(request):
                seen.append(request)
                return httpx.Response(200, json={"accessible": True, "type": operation})
            with self.subTest(tool=name), self.client(handler):
                response = self.tools[name].invoke({"url": url})
                self.assertEqual(response["type"], operation)
                self.assertEqual(len(seen), 1)
                self.assertEqual(seen[0].url.host, "api.mart.dev")
                self.assertEqual(seen[0].url.params["type"], operation)
                self.assertEqual(seen[0].url.params["url"], url)
                self.assertEqual(seen[0].headers["x-api-key"], KEY)
                self.assertNotIn(KEY, str(seen[0].url))

    def test_key_is_not_in_model_arguments_or_serialization(self):
        self.assertNotIn(KEY, repr(self.toolkit))
        self.assertNotIn(KEY, self.toolkit.model_dump_json())
        for tool in self.tools.values():
            self.assertNotIn("api_key", tool.get_input_schema().model_fields)
            self.assertNotIn(KEY, repr(tool))

    def test_email_is_opt_in_and_only_sent_to_profile(self):
        for enabled in [False, True]:
            toolkit = MartToolkit(api_key=KEY, include_email=enabled)
            tools = {tool.name: tool for tool in toolkit.get_tools()}
            for name, url in [("mart_enrich_person", PERSON), ("mart_refresh_contact", PERSON),
                              ("mart_enrich_company", COMPANY), ("mart_get_posts", PERSON)]:
                def handler(request):
                    expected = enabled and name == "mart_enrich_person"
                    self.assertEqual(request.url.params.get("email"), "true" if expected else None)
                    return httpx.Response(200, json={"accessible": True})
                with self.subTest(enabled=enabled, tool=name), self.client(handler):
                    tools[name].invoke({"url": url})

    def test_agent_cannot_override_email_configuration(self):
        with patch("langchain_mart.toolkit.httpx.Client") as client:
            for field in ["email", "include_email"]:
                with self.subTest(field=field), self.assertRaises(ValidationError):
                    self.tools["mart_enrich_person"].invoke({"url": PERSON, field: True})
            client.assert_not_called()
        for value in ["false", "true", 0, 1, None]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                MartToolkit(api_key=KEY, include_email=value)

    def test_email_result_is_preserved_for_every_status(self):
        tool = MartToolkit(api_key=KEY, include_email=True).get_tools()[0]
        fixture = json.loads((Path(__file__).parents[1] / "examples/responses/profile-with-email.json").read_text())
        for status, match, address in [("deliverable", "inferred", "ava@meridian.example"),
                                       ("unknown", None, None), ("verified", "independent", "ava@meridian.example")]:
            payload = json.loads(json.dumps(fixture))
            payload["profile"].update(workEmail=address, emailStatus=status, emailMatch=match,
                                      emailConfidence=92 if address else None)
            with self.subTest(status=status), self.client(lambda request: httpx.Response(200, json=payload)):
                self.assertEqual(tool.invoke({"url": PERSON}), payload)

    def test_async_email_matches_sync_wire_contract(self):
        async def exercise():
            def handler(request):
                self.assertEqual(request.url.params["email"], "true")
                self.assertEqual(request.url.params["type"], "profile")
                return httpx.Response(200, json={"profile": {"workEmail": None, "emailStatus": "unknown"}})
            with patch("langchain_mart.toolkit.httpx.AsyncClient", side_effect=lambda **kwargs: ASYNC_CLIENT(
                    transport=httpx.MockTransport(handler), **kwargs)):
                result = await MartToolkit(api_key=KEY, include_email=True).get_tools()[0].ainvoke({"url": PERSON})
                self.assertIsNone(result["profile"]["workEmail"])
        asyncio.run(exercise())

    def test_missing_or_empty_key_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            MartToolkit()
        with self.assertRaises(ValidationError):
            MartToolkit(api_key=" ")

    def test_wrong_url_kind_and_extra_arguments_are_rejected_before_request(self):
        with patch("langchain_mart.toolkit.httpx.Client") as client:
            for args in [{"url": COMPANY}, {"url": PERSON, "api_key": KEY}, {"url": "https://example.com/in/person"}]:
                with self.subTest(args=args), self.assertRaises(ValidationError):
                    self.tools["mart_enrich_person"].invoke(args)
            client.assert_not_called()

    def test_empty_result_and_availability_are_preserved(self):
        payload = {"accessible": False, "profileState": "exists_not_public", "profile": None}
        with self.client(lambda request: httpx.Response(200, json=payload)):
            self.assertEqual(self.tools["mart_enrich_person"].invoke({"url": PERSON}), payload)

    def test_complete_response_and_unlisted_fields_are_preserved(self):
        payload = {"accessible": True, "profileState": "accessible", "resolvedUrl": PERSON,
                   "profile": {"name": "Example Person", "currentTitle": None,
                               "education": [{"school": "Example University"}],
                               "futureField": {"zero": 0, "flag": False}},
                   "futureRootField": [1, 2], "creditsUsed": 1}
        with self.client(lambda request: httpx.Response(200, json=payload)):
            self.assertEqual(self.tools["mart_enrich_person"].invoke({"url": PERSON}), payload)

    def test_agent_tool_call_returns_complete_json_in_tool_message(self):
        payload = {"accessible": True, "profile": {"name": "Example", "nested": [None, False, 0]}}
        with self.client(lambda request: httpx.Response(200, json=payload)):
            message = self.tools["mart_enrich_person"].invoke({
                "type": "tool_call", "name": "mart_enrich_person", "id": "person-1", "args": {"url": PERSON}})
        self.assertEqual(message.tool_call_id, "person-1")
        self.assertEqual(json.loads(message.content), payload)

    def test_posts_budget_is_enforced_before_network(self):
        with patch("langchain_mart.toolkit.httpx.Client") as client, self.assertRaises(ValidationError):
            self.tools["mart_get_posts"].invoke({"url": PERSON, "limit": 4})
        client.assert_not_called()

    def test_posts_default_is_three(self):
        def handler(request):
            self.assertEqual(request.url.params["limit"], "3")
            return httpx.Response(200, json={"posts": []})
        with self.client(handler):
            self.tools["mart_get_posts"].invoke({"url": PERSON})

    def test_lower_post_budget_changes_schema_and_default_request(self):
        toolkit = MartToolkit(api_key=KEY, max_posts=1)
        posts = {tool.name: tool for tool in toolkit.get_tools()}["mart_get_posts"]
        limit_schema = posts.get_input_schema().model_json_schema()["properties"]["limit"]
        self.assertEqual(limit_schema["default"], 1)
        self.assertEqual(limit_schema["maximum"], 1)
        def handler(request):
            self.assertEqual(request.url.params["limit"], "1")
            return httpx.Response(200, json={"posts": []})
        with self.client(handler):
            posts.invoke({"url": PERSON})
        with patch("langchain_mart.toolkit.httpx.Client") as client, self.assertRaises(ValidationError):
            posts.invoke({"url": PERSON, "limit": 2})
        client.assert_not_called()

    def test_http_errors_do_not_echo_body_or_key(self):
        for status in [401, 429, 500]:
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(status, text=KEY, headers={"Retry-After": "4"})
            with self.subTest(status=status), self.client(handler), self.assertRaises(ToolException) as caught:
                self.tools["mart_enrich_person"].invoke({"url": PERSON})
            self.assertNotIn(KEY, str(caught.exception))
            self.assertEqual(len(calls), 1)
            self.assertIn(str(status), str(caught.exception))

    def test_redirect_does_not_send_key_to_another_host(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(302, headers={"Location": "https://example.com/collect"})
        with self.client(handler), self.assertRaises(ToolException):
            self.tools["mart_enrich_person"].invoke({"url": PERSON})
        self.assertEqual(len(calls), 1)

    def test_invalid_json_is_a_controlled_error(self):
        for body in [KEY, "[]"]:
            with self.subTest(body=body), self.client(lambda request: httpx.Response(200, text=body)), self.assertRaises(ToolException):
                self.tools["mart_enrich_person"].invoke({"url": PERSON})

    def test_timeout_is_redacted(self):
        def handler(request):
            raise httpx.ReadTimeout(KEY, request=request)
        with self.client(handler), self.assertRaises(ToolException) as caught:
            self.tools["mart_enrich_person"].invoke({"url": PERSON})
        self.assertNotIn(KEY, str(caught.exception))

    def test_async_works_and_cancellation_propagates(self):
        async def exercise():
            with patch("langchain_mart.toolkit.httpx.AsyncClient", side_effect=lambda **kwargs: ASYNC_CLIENT(
                    transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"accessible": True})), **kwargs)):
                result = await self.tools["mart_enrich_person"].ainvoke({"url": PERSON})
                self.assertTrue(result["accessible"])
            async def cancelled(request):
                raise asyncio.CancelledError
            with patch("langchain_mart.toolkit.httpx.AsyncClient", side_effect=lambda **kwargs: ASYNC_CLIENT(
                    transport=httpx.MockTransport(cancelled), **kwargs)), self.assertRaises(asyncio.CancelledError):
                await self.tools["mart_enrich_person"].ainvoke({"url": PERSON})
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
