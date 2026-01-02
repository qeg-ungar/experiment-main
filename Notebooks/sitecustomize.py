"""Workspace-local startup hooks for Jupyter/IPython.

Python automatically imports `sitecustomize` (if importable on sys.path) during
startup. In VS Code Jupyter, the kernel CWD is commonly the notebook folder,
so placing this file next to notebooks makes the behavior workspace-local.

Purpose:
- Avoid noisy IPython formatter tracebacks when displaying RPyC netref objects.
  IPython's formatter probes `obj.__class__`, which may trigger remote lookups
  on RPyC proxies. We force IPython to use `type(obj)` (local) instead.
"""


def _install_rpyc_ipython_formatter_workaround() -> None:
    try:
        import IPython.core.formatters as _ip_formatters
    except Exception:
        return

    if getattr(_ip_formatters, "_rpyc_safe_get_type_installed", False):
        return

    _ip_formatters._rpyc_safe_get_type_installed = True
    _orig_get_type = _ip_formatters._get_type

    def _safe_get_type(obj):  # noqa: ANN001
        try:
            return type(obj)
        except Exception:
            return _orig_get_type(obj)

    _ip_formatters._get_type = _safe_get_type


_install_rpyc_ipython_formatter_workaround()
