import logging

from qtaim_embed import _pymatgen_compat  # noqa: F401  install pickle shim

logging.getLogger(__name__).addHandler(logging.NullHandler())
