"""Per-run tool-use rules that protect local dependencies without answering tasks."""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_DISCOVERY_REQUIRED_MESSAGE = (
    "Database schema discovery is required before querying. "
    "Call describe_database_schema, then write a read-only SELECT query using that schema."
)


@dataclass
class DatabaseSchemaProtocol:
    """Require interface discovery before a query, while leaving SQL reasoning to the agent."""

    schema_discovered: bool = False

    def blocks(self, tool_name: str) -> bool:
        return tool_name == "query_database" and not self.schema_discovered

    def record_success(self, tool_name: str) -> None:
        if tool_name == "describe_database_schema":
            self.schema_discovered = True
