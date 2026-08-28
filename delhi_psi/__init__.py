"""Delhi Public Services Index (PSI) pipeline.

Public entry points live in `delhi_psi.pipeline` (`preprocess`, `compute`,
`compute_frames`) and `delhi_psi.config` (`load_config`). The math modules
(`geometry`, `neighbors`, `index`) are pure functions with keyword knobs and
never import `config`.
"""

__version__ = "0.3.0"
