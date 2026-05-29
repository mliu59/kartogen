"""Module entry point so ``python -m worldgen`` runs the export CLI.

Putting the CLI here (instead of in ``export.py``) avoids the
``RuntimeWarning: '...' found in sys.modules ...`` that ``-m worldgen.export``
triggered: ``worldgen/__init__.py`` re-exports from ``worldgen.export``, so
that module is already loaded by the time the ``-m`` machinery would try to
load it as ``__main__``. ``__main__.py`` is not re-exported anywhere, so no
double-load.
"""

from worldgen.export import main

if __name__ == "__main__":
    main()
