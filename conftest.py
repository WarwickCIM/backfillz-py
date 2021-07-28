def pytest_addoption(parser):
    parser.addoption("--compare-images", action="store", default="True")
