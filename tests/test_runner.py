from agent_failure_predictor.runner import classify_task, resolve_task_type, text_tool_call


def test_task_classifier_prioritizes_explicit_tool_context_over_generic_question_words():
    assert classify_task("What is Agent Hub described as in the knowledge base?") == "document_lookup"
    assert classify_task("What is the highest customer balance in the database?") == "database_lookup"
    assert classify_task("What is 18 multiplied by 7?") == "calculation"
    assert resolve_task_type("What is the database status?", "document_lookup") == "document_lookup"


def test_text_tool_call_accepts_only_known_tool_json():
    assert text_tool_call('Running tool. {"name":"describe_database_schema","parameters":{}}') == (
        "describe_database_schema", {}
    )
    assert text_tool_call(r'{"name":"describe_database_schema","parameters":{}}') == (
        "describe_database_schema", {}
    )
    assert text_tool_call("Please inspect the schema first.\n\ndescribe_database_schema") == (
        "describe_database_schema", {}
    )
    assert text_tool_call('{"name":"unknown","parameters":{}}') is None
