"""eqhunt - symbolic regression by genetic programming.

Two entry points:

* Ultra-simple:  ``eqhunt.fit(X, y)`` returns a trained Model.
* Fully configurable:  build a ``Config``, instantiate ``Model(config)``,
  call ``.fit(X, y)`` (or ``.fit_csv(path)``).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

from ._core import Config, Formula as _Formula, OpWeights, Sample

__version__ = "0.0.5"
__all__ = ["fit", "Model", "Config", "OpWeights", "Sample"]


Number = Union[int, float]


def _is_2d(X: Sequence) -> bool:
    if len(X) == 0:
        return False
    first = X[0]
    return hasattr(first, "__iter__") and not isinstance(first, (str, bytes))


def _load_csv(path: str) -> List[Sample]:
    """Mirror of the C++ CSV loader (same comment rules, comma-separated)."""
    samples: List[Sample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cols = [float(c) for c in line.split(",")]
            except ValueError:
                continue
            if len(cols) < 2:
                continue
            s = Sample()
            s.inputs = cols[:-1]
            s.target = cols[-1]
            samples.append(s)
    return samples


def _to_samples(X: Sequence, y: Sequence[Number]) -> List[Sample]:
    samples: List[Sample] = []
    for xi, yi in zip(X, y):
        s = Sample()
        if hasattr(xi, "__iter__") and not isinstance(xi, (str, bytes)):
            s.inputs = [float(v) for v in xi]
        else:
            s.inputs = [float(xi)]
        s.target = float(yi)
        samples.append(s)
    return samples


class Model:
    """Friendly Python wrapper around the C++ Formula engine.

    Examples
    --------
    Ultra simple::

        import eqhunt
        m = eqhunt.fit([[1, 1], [2, 3], [4, 5]], [2, 5, 9])
        print(m.formula)
        m.predict([6, 7])

    Fully configurable::

        cfg = eqhunt.Config()
        cfg.pop = 800
        cfg.gen = 50000
        cfg.op_weights.sin = 1.0          # boost sine
        cfg.trig_penalty   = 1.5
        m = eqhunt.Model(cfg).fit(X, y)
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config if config is not None else Config()
        self._impl = _Formula(self.config)
        self.error: Optional[float] = None     # total error over the last training set
        self.n_samples: int = 0                # size of the last training set

    # ---- training ----------------------------------------------------------

    def fit(self, X: Sequence, y: Sequence[Number]) -> "Model":
        samples = _to_samples(X, y)
        self._impl.train(samples)
        self.error = self._impl.total_error(samples)
        self.n_samples = len(samples)
        return self

    def fit_csv(self, path: str) -> "Model":
        self._impl.train_csv(path)
        samples = _load_csv(path)
        self.error = self._impl.total_error(samples) if samples else None
        self.n_samples = len(samples)
        return self

    @property
    def mean_error(self) -> Optional[float]:
        """Mean absolute error on the last training set (None if not trained)."""
        if self.error is None or self.n_samples == 0:
            return None
        return self.error / self.n_samples

    def get_error(self) -> Optional[float]:
        """Total error on the last training set (None if not trained)."""
        return self.error

    def stop(self) -> None:
        """Ask the training loop to stop after the current generation."""
        self._impl.request_stop()

    # ---- inspection / inference -------------------------------------------

    @property
    def formula(self) -> str:
        return self._impl.get_formula()

    @property
    def num_vars(self) -> int:
        return self._impl.num_vars

    def predict(self, X: Sequence) -> Union[Optional[float], List[Optional[float]]]:
        """Evaluate the trained formula.

        Accepts either a single input row (returns scalar / None) or a
        batch of rows (returns list of scalars / None).
        """
        if _is_2d(X):
            return [self._impl.compute([float(v) for v in row]) for row in X]
        if hasattr(X, "__iter__") and not isinstance(X, (str, bytes)):
            return self._impl.compute([float(v) for v in X])
        return self._impl.compute([float(X)])

    def total_error(self, X: Sequence, y: Sequence[Number]) -> float:
        return self._impl.total_error(_to_samples(X, y))

    # ---- save / load --------------------------------------------------------

    def load_formula(self, formula_str: str) -> "Model":
        """Replace the current trained tree with one parsed from a string.

        Accepts both bare expressions ("(x*x - y*y)") and the full
        get_formula() output ("f(x,y) = (x*x - y*y)"). Returns self so it can
        be chained. No training is performed.
        """
        self._impl.load(formula_str)
        self.error = None
        self.n_samples = 0
        return self

    def save(self, path: str) -> None:
        """Write the current formula to a text file."""
        from pathlib import Path
        Path(path).write_text(self.formula)

    @classmethod
    def from_formula(cls, formula_str: str, config: Optional[Config] = None) -> "Model":
        """Create a Model directly from a formula string, no training needed."""
        m = cls(config)
        m.load_formula(formula_str)
        return m

    @classmethod
    def load_file(cls, path: str, config: Optional[Config] = None) -> "Model":
        """Create a Model by reading a formula from a text file."""
        from pathlib import Path
        return cls.from_formula(Path(path).read_text(), config)

    # ---- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<eqhunt.Model {self.formula}>"

    def __str__(self) -> str:
        return self.formula


def fit(
    X: Sequence,
    y: Sequence[Number],
    *,
    generations: int = 15000,
    pop: int = 400,
    accepted_error: float = 0.5,
    verbose: bool = False,
    config: Optional[Config] = None,
    **kwargs,
) -> Model:
    """Train a Model with reasonable defaults.

    Any extra keyword argument is forwarded to ``Config``. Pass a pre-built
    ``config=`` to use a fully customized configuration; the keyword arguments
    above still override individual fields.
    """
    cfg = config if config is not None else Config()
    cfg.gen = generations
    cfg.pop = pop
    cfg.accepted_error = accepted_error
    cfg.verbose = verbose
    for k, v in kwargs.items():
        if not hasattr(cfg, k):
            raise TypeError(f"Unknown config option: {k!r}")
        setattr(cfg, k, v)
    return Model(cfg).fit(X, y)
