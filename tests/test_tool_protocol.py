from agent_failure_predictor.tool_protocol import DatabaseSchemaProtocol


def test_database_schema_protocol_blocks_query_until_schema_is_discovered():
    protocol = DatabaseSchemaProtocol()

    assert protocol.blocks("query_database")
    assert not protocol.blocks("describe_database_schema")
    protocol.record_success("describe_database_schema")
    assert not protocol.blocks("query_database")
