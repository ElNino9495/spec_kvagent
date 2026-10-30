import inspect


def test_connector_is_concrete_and_takes_kv_cache_config():
    from kvm.connector import AgentKVConnector, AgentKVMetadata

    assert not inspect.isabstract(AgentKVConnector)
    assert "kv_cache_config" in inspect.signature(AgentKVConnector.__init__).parameters
    assert AgentKVMetadata().loads == []
