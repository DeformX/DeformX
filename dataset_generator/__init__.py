"""DeformX dataset generation: drop/hang and datacenter wire renders."""

import logging

# Isaac Sim sets Python's root logger to DEBUG at startup, and numba then dumps compiler IR
# for every jitted engine function. Nothing in the generators needs that.
logging.getLogger("numba").setLevel(logging.WARNING)
