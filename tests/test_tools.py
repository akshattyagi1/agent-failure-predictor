from agent_failure_predictor.tools import (
    calculate,
    describe_database_schema,
    initialise_demo_data,
    query_database,
    search_documents,
)


def test_local_tools(tmp_path, monkeypatch):
    # The integration agent uses project data; these tests prove the safe primitives.
    initialise_demo_data()
    assert calculate("25 * 17") == "425"
    assert "Sydney" in search_documents("Sydney")
    assert "Asha" in query_database("SELECT name FROM customers WHERE balance > 1000")
    assert "id (INTEGER)" in describe_database_schema()
    assert "name (TEXT)" in describe_database_schema()
    initialise_demo_data("accounts")
    assert "holder_name (TEXT)" in describe_database_schema()
    assert "Lina" in query_database("SELECT holder_name FROM accounts WHERE current_balance > 1000")
    initialise_demo_data()
