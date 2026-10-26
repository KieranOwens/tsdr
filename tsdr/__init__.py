from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tsdr")
except PackageNotFoundError:
    # tsdr is not installed (e.g., it is imported from a source checkout)
    __version__ = "unknown"
