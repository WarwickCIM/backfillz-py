def pytest_addoption(parser):
    """Whether to compare generated images with stored expected images."""
    parser.addoption("--compare-images", action="store", default="True")
