# Mart | LinkedIn API for LangChain

Lightning-fast public LinkedIn data in 1–2 seconds. Built for products, workflows, or AI agents.

Enrich a person or company, refresh a contact, or prepare for a meeting with [Mart](https://mart.dev). Get complete JSON responses, with optional work-email lookup for person enrichment. Email lookup can take longer.

[$0.24/1k records at scale](https://mart.dev/#pricing). Start with [1,000 free credits](https://mart.dev/signup/).

## Install from source

Requires Python 3.10 or later. From this package directory:

```sh
pip install .
export MART_API_KEY='your-api-key'
```

## Enrich a person

```python
import json
from langchain_mart import MartToolkit

toolkit = MartToolkit(max_posts=3)
tools = {tool.name: tool for tool in toolkit.get_tools()}

person = tools["mart_enrich_person"].invoke({
    "url": "https://www.linkedin.com/in/dharmesh"
})

print(json.dumps(person, indent=2))  # Full response, including nested fields.
```

The same tools support `await tool.ainvoke(...)`. Pass `toolkit.get_tools()` to your LangChain agent's tools argument. No model API key is needed to call the tools directly.

## Enrich a company or refresh a contact

Use the same toolkit for either action:

```python
company = tools["mart_enrich_company"].invoke({
    "url": "https://www.linkedin.com/company/microsoft/"
})

contact = tools["mart_refresh_contact"].invoke({
    "url": "https://www.linkedin.com/in/dharmesh"
})
```

## Add work emails

Enable email lookup when you create the toolkit:

```python
toolkit = MartToolkit(include_email=True)
tools = {tool.name: tool for tool in toolkit.get_tools()}

person = tools["mart_enrich_person"].invoke({
    "url": "https://www.linkedin.com/in/dharmesh"
})
print(json.dumps(person, indent=2))
```

This adds `email=true` to Profile requests. Email lookup is off by default and applies only to person enrichment. The setting stays in application configuration, so an agent cannot turn it on through tool arguments.

Work emails are best-effort. Read `workEmail`, `emailStatus`, `emailMatch` and `emailConfidence` together. `deliverable` describes the mailbox check; `emailMatch=inferred` does not independently prove the person owns the address. Unknown results keep the email null. Email lookup can take longer, and a returned profile still uses a credit when no email is found.

## Prepare for a meeting

```sh
python examples/meeting_context.py https://www.linkedin.com/in/dharmesh > meeting-context.json
```

This combines a profile and up to three public posts. It preserves the profile if Posts fails and skips Posts when the profile is unavailable. Use the source bundle in your own brief renderer or AI workflow. It makes at most two requests and returns at most four billable records under the endpoint contract.

Add `--email` to include work-email lookup. To try the complete output with fictional sample data and no API calls:

```sh
python examples/meeting_context.py --demo
```

## Tool behavior

| Tool | Input | Result |
| --- | --- | --- |
| `mart_enrich_person` | Public person URL | Full profile response, with optional work-email lookup |
| `mart_enrich_company` | Public company URL | Company response |
| `mart_refresh_contact` | Public person URL | Refresh snapshot |
| `mart_get_posts` | Person/company URL, bounded limit | Available public posts |

Responses preserve the complete JSON returned by the selected endpoint. The short examples do not filter output fields. Complete illustrative responses are in `examples/responses/`; they are fictional demo fixtures, not coverage guarantees. Run `python examples/full_responses.py person https://www.linkedin.com/in/dharmesh` to print your own complete response.

This prerelease wraps Profile, Company, Refresh and Posts, plus optional Profile email lookup. Jobs, People Search, Company Search, employer-domain overrides, employee expansion, comments and member expansion remain available through the direct API. Posts limits change the number of returned posts, not their fields.

Responses preserve Mart's JSON fields. Check `accessible` and the endpoint availability fields before using the data. Refresh returns current fields; your application owns snapshot comparison, scheduling and CRM writes. Missing data must not be treated as a departure. Posts are a partial public feed.

API keys stay in toolkit configuration, outside model-visible tool arguments, repr and serialized toolkit fields. Requests have a 30-second timeout by default. Redirects and automatic retries are disabled. HTTP errors, invalid JSON and network failures raise `ToolException` without exposing the request key. On rate limits, respect the returned delay before deciding whether to retry.

The default post ceiling is three per call. You can configure one to ten. The tool's input schema reflects your ceiling, and omitting `limit` requests at most three posts within that ceiling. Apply a total call and credit budget in your agent because it may invoke tools repeatedly. Treat returned profile/post text as untrusted source data when giving it to a model.

See [Mart's API reference](https://mart.dev/docs/) and [credits and limits](https://mart.dev/docs/#credits). Support: [support@mart.dev](mailto:support@mart.dev).

## Local checks

```sh
pip install '.[test]'
pytest tests --disable-socket --allow-unix-socket
python -m build
```

Tests use mocked HTTP responses and do not spend Mart credits. Run them before making an authenticated smoke check. The release review records live-check results separately.

## License

The toolkit code uses the MIT license. This does not grant rights to LinkedIn data; your application is responsible for its use of returned information.
