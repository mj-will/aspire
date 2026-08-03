def test_version():
    from aspire import __version__

    assert __version__ != "unknown"
