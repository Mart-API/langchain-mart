"""LangChain's standard unit suite for each exposed tool and email mode."""

from langchain_tests.unit_tests.tools import ToolsUnitTests
from langchain_mart import MartToolkit


class TestPersonStandard(ToolsUnitTests):
    tool_name = "mart_enrich_person"
    email_enabled = False
    url = "https://www.linkedin.com/in/example-person"

    @property
    def tool_constructor(self):
        return next(tool for tool in MartToolkit(api_key="fixture-key", include_email=self.email_enabled).get_tools()
                    if tool.name == self.tool_name)

    @property
    def tool_invoke_params_example(self):
        return {"url": self.url}


class TestPersonEmailStandard(TestPersonStandard):
    email_enabled = True


class TestCompanyStandard(TestPersonStandard):
    tool_name = "mart_enrich_company"
    url = "https://www.linkedin.com/company/example-company"


class TestRefreshStandard(TestPersonStandard):
    tool_name = "mart_refresh_contact"


class TestPostsStandard(TestPersonStandard):
    tool_name = "mart_get_posts"
