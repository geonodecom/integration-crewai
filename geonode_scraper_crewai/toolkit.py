from __future__ import annotations

import asyncio
from typing import Any, Sequence

from geonode_scraper_tools_core import ScraperToolService, ScraperToolSettings, get_operations
from geonode_scraper_tools_core.registry import OperationSpec


class ScraperCrewAIToolkit:
    def __init__(self, service: ScraperToolService) -> None:
        self.service = service

    @classmethod
    def from_settings(cls, settings: ScraperToolSettings) -> ScraperCrewAIToolkit:
        return cls(ScraperToolService(settings, client=_client()))

    def get_tools(self, operations: Sequence[str] | None = None) -> list[Any]:
        from crewai.tools import BaseTool

        return [self._build_tool(BaseTool=BaseTool, operation=operation) for operation in get_operations(operations)]

    def _build_tool(self, *, BaseTool: type[Any], operation: OperationSpec) -> Any:
        service = self.service

        class _CrewAIScraperTool(BaseTool):
            name: str
            description: str
            args_schema: type[Any]

            def _run(self, **kwargs: Any) -> dict[str, Any]:
                return operation.invoke(service, **kwargs)

            async def _arun(self, **kwargs: Any) -> dict[str, Any]:
                return await asyncio.to_thread(operation.invoke, service, **kwargs)

            @staticmethod
            def cache_function(_arguments: dict[str, Any], _result: Any) -> bool:
                return False

        return _CrewAIScraperTool(
            name=operation.tool_name,
            description=operation.description,
            args_schema=operation.args_schema,
        )


def build_crewai_tools(
    *,
    settings: ScraperToolSettings | None = None,
    service: ScraperToolService | None = None,
    operations: Sequence[str] | None = None,
) -> list[Any]:
    resolved_service = _resolve_service(settings=settings, service=service)
    return ScraperCrewAIToolkit(resolved_service).get_tools(operations=operations)


def _resolve_service(
    *,
    settings: ScraperToolSettings | None,
    service: ScraperToolService | None,
) -> ScraperToolService:
    if service is not None:
        return service
    if settings is None:
        raise ValueError("Provide either settings or service to build CrewAI tools.")
    return ScraperToolService(settings, client=_client())


def _client() -> str:
    # The API labels requests from this package `crewai` (scraper-api#674).
    from . import __version__

    return f"geonode-crewai/{__version__}"
