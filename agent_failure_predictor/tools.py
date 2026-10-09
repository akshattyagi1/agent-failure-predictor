"""Deterministic local tools used by the first test agent."""

from __future__ import annotations

import ast
import operator
import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENT_PATH = DATA_DIR / "knowledge_base.txt"
DATABASE_PATH = DATA_DIR / "demo.sqlite3"
ENVIRONMENT_ROOT = DATA_DIR / "environments"

_active_document_path = DOCUMENT_PATH
_active_database_path = DATABASE_PATH

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


ENVIRONMENTS = {
    "default": {
        "document": "Agent Hub is a local observability project.\nSydney is the planned initial deployment city.\nTelemetry events record latency, retries, tool calls, and errors.\n",
        "schema": "CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, name TEXT, balance REAL)",
        "insert": "INSERT INTO customers (name, balance) VALUES (?, ?)",
        "rows": [("Asha", 1450.0), ("Mina", 900.0), ("Jordan", 2100.0)],
    },
    "accounts": {
        "document": "Account Monitor tracks customer account health.\nPerth is the planned support hub.\nAudit events record account checks, retry attempts, and validation errors.\n",
        "schema": "CREATE TABLE IF NOT EXISTS accounts (account_id INTEGER PRIMARY KEY, holder_name TEXT, current_balance REAL)",
        "insert": "INSERT INTO accounts (holder_name, current_balance) VALUES (?, ?)",
        "rows": [("Lina", 3200.0), ("Omar", 750.0), ("Priya", 1850.0)],
    },
    "orders": {
        "document": "Order Lens is a local fulfillment observability project.\nBrisbane is the planned operations city.\nRun records capture order lookups, delays, and validation errors.\n",
        "schema": "CREATE TABLE IF NOT EXISTS orders (order_id INTEGER PRIMARY KEY, customer TEXT, total REAL)",
        "insert": "INSERT INTO orders (customer, total) VALUES (?, ?)",
        "rows": [("Nia", 125.0), ("Evan", 640.0), ("Sora", 280.0)],
    },
}


def initialise_demo_data(environment_id: str = "default") -> None:
    """Select and seed a deterministic local environment for one sequential agent run."""
    global _active_document_path, _active_database_path
    if environment_id not in ENVIRONMENTS:
        raise ValueError(f"Unknown tool environment: {environment_id}")
    definition = ENVIRONMENTS[environment_id]
    DATA_DIR.mkdir(exist_ok=True)
    if environment_id == "default":
        _active_document_path, _active_database_path = DOCUMENT_PATH, DATABASE_PATH
    else:
        directory = ENVIRONMENT_ROOT / environment_id
        directory.mkdir(parents=True, exist_ok=True)
        _active_document_path, _active_database_path = directory / "knowledge_base.txt", directory / "demo.sqlite3"
    if not _active_document_path.exists():
        _active_document_path.write_text(definition["document"], encoding="utf-8")
    with sqlite3.connect(_active_database_path) as connection:
        connection.execute(definition["schema"])
        table = definition["schema"].split()[5]
        count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count == 0:
            connection.executemany(definition["insert"], definition["rows"])


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Use only numbers and arithmetic operators."""
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("Only basic arithmetic is allowed.")

    return str(evaluate(tree.body))


def search_documents(query: str) -> str:
    """Search the local knowledge base for text relevant to a query."""
    terms = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
    matches = [line for line in _active_document_path.read_text(encoding="utf-8").splitlines() if terms & set(line.lower().split())]
    return "\n".join(matches) if matches else "No matching documents found."


def query_database(sql: str) -> str:
    """Run a read-only SELECT query against the local customers database."""
    normalized = sql.strip().lower()
    if not normalized.startswith("select") or ";" in normalized:
        raise ValueError("Only one read-only SELECT statement is allowed.")
    with sqlite3.connect(_active_database_path) as connection:
        cursor = connection.execute(sql)
        columns = [column[0] for column in cursor.description or []]
        rows = cursor.fetchmany(25)
    return str([dict(zip(columns, row)) for row in rows])


def describe_database_schema() -> str:
    """Return the names and types of tables and columns available to read-only database queries."""
    with sqlite3.connect(_active_database_path) as connection:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        descriptions = []
        for (table_name,) in tables:
            columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
            details = ", ".join(f"{column[1]} ({column[2]})" for column in columns)
            descriptions.append(f"Table {table_name}: {details}.")
    return " ".join(descriptions)


AVAILABLE_TOOLS = {
    "calculate": calculate,
    "search_documents": search_documents,
    "query_database": query_database,
    "describe_database_schema": describe_database_schema,
}
