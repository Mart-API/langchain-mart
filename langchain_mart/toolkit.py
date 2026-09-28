"""Bounded, synchronous and asynchronous Mart tools."""

import os
from typing import Any
from urllib.parse import urlsplit

import httpx
from langchain_core.tools import BaseToolkit, BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictBool, create_model, field_validator


ENDPOINT = "https://api.mart.dev/v1/linkedin"


def _validate_url(value: str, kinds: set[str]) -> str:
    parsed = urlsplit(value)
    parts = parsed.path.strip("/").split("/")
    if (parsed.scheme != "https" or parsed.hostname not in {"linkedin.com", "www.linkedin.com"}
            or parsed.username or parsed.password or parsed.port
            or len(parts) != 2 or parts[0] not in kinds or not parts[1]):
        raise ValueError("Provide the public LinkedIn URL for this action.")
    return value


class PersonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(description="Public LinkedIn person URL, https://www.linkedin.com/in/...")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_url(value, {"in"})


class CompanyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(description="Public LinkedIn company URL, https://www.linkedin.com/company/...")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_url(value, {"company"})


class PostsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(description="Public LinkedIn person or company URL.")
    limit: int = Field(default=3, ge=1, le=10, description="Maximum returned posts; must also fit the toolkit's configured max_posts.")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_url(value, {"in", "company"})


def _environment_key() -> SecretStr:
    value = os.environ.get("MART_API_KEY", "").strip()
    if not value:
        raise ValueError("Set MART_API_KEY or supply api_key when creating MartToolkit.")
    return SecretStr(value)


class MartToolkit(BaseToolkit):
    """Read public LinkedIn data. Create one toolkit per application configuration."""

    api_key: SecretStr = Field(default_factory=_environment_key, exclude=True, repr=False)
    timeout: float = Field(default=30, gt=0, le=120)
    max_posts: int = Field(default=3, ge=1, le=10)
    include_email: StrictBool = Field(default=False, description="Search for work emails on person enrichment calls.")

    @field_validator("api_key")
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("A non-empty Mart API key is required.")
        return value

    def _params(self, operation: str, url: str, limit: int | None) -> dict[str, Any]:
        params: dict[str, Any] = {"type": operation, "url": url}
        if operation == "profile" and self.include_email:
            params["email"] = "true"
        if operation == "posts":
            if limit is None:
                limit = min(3, self.max_posts)
            if not 1 <= limit <= self.max_posts:
                raise ToolException("Requested posts exceed the configured max_posts budget.")
            params["limit"] = limit
        return params

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        if not 200 <= response.status_code < 300:
            delay = response.headers.get("Retry-After", "")
            suffix = f" Retry after {delay} seconds." if delay.isdigit() else ""
            raise ToolException(f"Mart returned HTTP {response.status_code}.{suffix}")
        try:
            payload = response.json()
        except ValueError:
            raise ToolException("Mart returned an invalid JSON response.") from None
        if not isinstance(payload, dict):
            raise ToolException("Mart returned an unexpected response shape.")
        return payload

    def _request(self, operation: str, url: str, limit: int | None = None) -> dict[str, Any]:
        params = self._params(operation, url, limit)
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
                response = client.get(ENDPOINT, params=params, headers={
                    "x-api-key": self.api_key.get_secret_value(), "Accept": "application/json",
                })
            return self._decode(response)
        except httpx.TimeoutException:
            raise ToolException("Mart request timed out. No automatic retry was made.") from None
        except httpx.RequestError:
            raise ToolException("Could not reach Mart. No automatic retry was made.") from None

    async def _arequest(self, operation: str, url: str, limit: int | None = None) -> dict[str, Any]:
        params = self._params(operation, url, limit)
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
                response = await client.get(ENDPOINT, params=params, headers={
                    "x-api-key": self.api_key.get_secret_value(), "Accept": "application/json",
                })
            return self._decode(response)
        except httpx.TimeoutException:
            raise ToolException("Mart request timed out. No automatic retry was made.") from None
        except httpx.RequestError:
            raise ToolException("Could not reach Mart. No automatic retry was made.") from None

    def _tool(self, name: str, operation: str, description: str, schema: type[BaseModel]) -> BaseTool:
        def run(url: str, limit: int | None = None) -> dict[str, Any]:
            return self._request(operation, url, limit)

        async def arun(url: str, limit: int | None = None) -> dict[str, Any]:
            return await self._arequest(operation, url, limit)

        return StructuredTool.from_function(
            func=run, coroutine=arun, name=name, description=description, args_schema=schema,
        )

    def get_tools(self) -> list[BaseTool]:
        posts_schema = create_model(
            "MartPostsInput", __base__=PostsInput,
            limit=(int, Field(default=min(3, self.max_posts), ge=1, le=self.max_posts,
                              description="Maximum returned posts within this toolkit's budget.")),
        )
        return [
            self._tool("mart_enrich_person", "profile",
                       "Enrich a person from a public LinkedIn profile URL. Returns the complete profile response and availability. Preserve titleSource and missing fields. "
                       + ("Work-email lookup is enabled. Preserve workEmail, emailStatus, emailMatch and emailConfidence; an inferred address is not proof of identity."
                          if self.include_email else "Work-email lookup is disabled."), PersonInput),
            self._tool("mart_enrich_company", "company",
                       "Enrich a company from a public LinkedIn company URL. Returns available company details and availability.", CompanyInput),
            self._tool("mart_refresh_contact", "refresh",
                       "Refresh a contact's available public fields. Returns a snapshot; does not update a CRM or prove job changes. Preserve prior values when fields are missing.", PersonInput),
            self._tool("mart_get_posts", "posts",
                       f"Get up to {self.max_posts} available public posts for a person or company. Partial public feed, not a complete archive. One credit per returned post.", posts_schema),
        ]
