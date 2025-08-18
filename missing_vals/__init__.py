# Package initialization for missing_vals
# Do not configure logging here; expose a NullHandler to avoid "No handler" warnings
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
