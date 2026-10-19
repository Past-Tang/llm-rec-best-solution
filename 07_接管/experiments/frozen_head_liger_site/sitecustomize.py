"""Install the locked frozen-head Liger fix in CLI and torchrun children."""
from liger_frozen_head_patch import install
try:
    install()
except BaseException as error:
    raise SystemExit(f"frozen_head_liger_site failed closed: {type(error).__name__}: {error}") from error
