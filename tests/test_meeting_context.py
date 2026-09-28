"""Offline checks of the runnable example's success and partial-failure paths."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import httpx
from langchain_mart import MartToolkit

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("meeting_context", ROOT / "examples/meeting_context.py")
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)
CLIENT = httpx.Client
URL = "https://www.linkedin.com/in/example-person"


class MeetingContextTests(unittest.TestCase):
    def exercise(self, profile, posts_status=200, include_email=False):
        calls = []
        posts = {"posts": [{"text": "Public post", "nested": {"keep": True}}], "creditsUsed": 1}

        def handler(request):
            calls.append(dict(request.url.params))
            if request.url.params["type"] == "profile":
                return httpx.Response(200, json=profile)
            return httpx.Response(posts_status, json=posts)

        with patch("langchain_mart.toolkit.httpx.Client", side_effect=lambda **kw: CLIENT(
                transport=httpx.MockTransport(handler), **kw)):
            result = EXAMPLE.collect_context(URL, MartToolkit(api_key="fixture-key", include_email=include_email))
        return result, calls, posts

    def test_full_responses_survive_the_meeting_bundle(self):
        profile = {"accessible": True, "profile": {"name": "Example", "future": [False, None]}}
        result, calls, posts = self.exercise(profile, include_email=True)
        self.assertEqual(result["profileResponse"], profile)
        self.assertEqual(result["postsResponse"], posts)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["email"], "true")
        self.assertNotIn("email", calls[1])
        self.assertEqual(calls[1]["limit"], "3")

    def test_posts_failure_keeps_the_profile_without_retry(self):
        profile = {"accessible": True, "profile": {"name": "Example"}}
        result, calls, _ = self.exercise(profile, posts_status=503)
        self.assertEqual(result["profileResponse"], profile)
        self.assertEqual(result["postsResponse"]["status"], "unavailable")
        self.assertEqual(len(calls), 2)

    def test_unavailable_profile_skips_posts(self):
        profile = {"accessible": False, "profile": None, "profileState": "exists_not_public"}
        result, calls, _ = self.exercise(profile)
        self.assertEqual(result["profileResponse"], profile)
        self.assertEqual(result["postsResponse"]["status"], "skipped_profile_unavailable")
        self.assertEqual(len(calls), 1)

    def test_demo_needs_no_key_and_is_labeled(self):
        with patch.dict("os.environ", {}, clear=True):
            result = subprocess.run([sys.executable, str(ROOT / "examples/meeting_context.py"), "--demo"],
                                    capture_output=True, text=True, check=True)
        output = json.loads(result.stdout)
        self.assertTrue(output["demo"])
        self.assertIn("profile", output["profileResponse"])
        self.assertIn("posts", output["postsResponse"])

    def test_email_flag_is_rejected_for_non_person_cli_actions(self):
        result = subprocess.run([sys.executable, str(ROOT / "examples/full_responses.py"),
                                 "company", "https://www.linkedin.com/company/example", "--email"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--email is available only for the person action", result.stderr)


if __name__ == "__main__":
    unittest.main()
