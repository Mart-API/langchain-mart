"""Print the full JSON response from one Mart tool without projecting fields."""

import argparse
import json

from langchain_mart import MartToolkit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["person", "company", "refresh", "posts"])
    parser.add_argument("url", help="Public LinkedIn URL for this action")
    parser.add_argument("--limit", type=int, choices=range(1, 11), default=3,
                        help="Maximum posts for the posts action (default: 3)")
    parser.add_argument("--email", action="store_true", help="Search for a work email with the person action")
    args = parser.parse_args()
    if args.email and args.action != "person":
        parser.error("--email is available only for the person action")
    names = {"person": "mart_enrich_person", "company": "mart_enrich_company",
             "refresh": "mart_refresh_contact", "posts": "mart_get_posts"}
    tools = {tool.name: tool for tool in MartToolkit(max_posts=args.limit, include_email=args.email).get_tools()}
    params = {"url": args.url}
    if args.action == "posts":
        params["limit"] = args.limit
    result = tools[names[args.action]].invoke(params)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
