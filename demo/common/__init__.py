"""Code shared by the demos: path resolution and the wire skeleton driver."""

import logging

# Isaac Sim sets Python's root logger to DEBUG at startup, and numba then dumps compiler IR
# for every jitted engine function. Nothing in the demos needs that.
logging.getLogger("numba").setLevel(logging.WARNING)
