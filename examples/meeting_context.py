"""Collect meeting source data. Keep the API key in MART_API_KEY."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from langchain_core.tools import ToolException
from langchain_mart import MartToolkit


def collect_context(url: str, toolkit: MartToolkit) -> dict:
    """Return full source responses, retaining the profile when posts fail."""
    tools = {tool.name: tool for tool in toolkit.get_tools()}
    profile = tools["mart_enrich_person"].invoke({"url": url})
    posts = {"posts": [], "status": "skipped_profile_unavailable"}
    if profile.get("accessible") is True:
        try:
            posts = tools["mart_get_posts"].invoke({"url": url, "limit": min(3, toolkit.max_posts)})
        except ToolException:
            posts = {"posts": [], "status": "unavailable"}
    return {"requestedUrl": url, "fetchedAt": datetime.now(timezone.utc).isoformat(),
            "profileResponse": profile, "postsResponse": posts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="Public LinkedIn profile URL")
    parser.add_argument("--email", action="store_true", help="Also search for a work email")
    parser.add_argument("--demo", action="store_true", help="Use fictional local fixtures, without a key or network calls")
    args = parser.parse_args()
    if args.demo:
        if args.url or args.email:
            parser.error("Use --demo alone; it does not perform a lookup")
        fixtures = Path(__file__).with_name("responses")
        profile = json.loads((fixtures / "profile.json").read_text())
        result = {"demo": True, "requestedUrl": profile["url"], "profileResponse": profile,
                  "postsResponse": json.loads((fixtures / "posts.json").read_text())}
    else:
        if not args.url:
            parser.error("Provide a LinkedIn URL or use --demo")
        result = collect_context(args.url, MartToolkit(max_posts=3, include_email=args.email))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
