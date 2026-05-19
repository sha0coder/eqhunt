# eqhunt

pip install eqhunt

Symbolic regression by genetic programming. C++ engine, Python bindings via [nanobind](https://github.com/wjakob/nanobind).

Give it a table of `(inputs, target)` pairs; it returns a human-readable formula
that approximates the relationship. No neural network, no black box — just an
algebraic expression you can read, paste into a calculator, or hand-tune.

```python
import eqhunt

X = [[1, 1], [2, 3], [4, 5], [7, 2], [9, 9]]
y = [2, 5, 9, 9, 18]

model = eqhunt.fit(X, y)
print(model.formula)        # e.g.  f(x,y) = (x+y)
print(model.error)          # e.g   0.0
print(model.predict([6, 7])) # -> 13.0
```

## Install

```bash
pip install eqhunt
```

Prebuilt wheels are published for Linux, macOS and Windows on common Python
versions. If pip falls back to building from source you'll need a C++17 compiler.

## Two ways to use it

### Ultra-simple

```python
import eqhunt

model = eqhunt.fit(X, y, generations=5000)
print(model.formula)
model.predict([1, 2])         # single row
model.predict([[1, 2], [3, 4]])  # batch
```

`fit()` accepts any `Config` field as a keyword argument:

```python
eqhunt.fit(X, y, pop=800, trig_penalty=2.0, bloat_penalty=0.3)
```

### Fully configurable

```python
import eqhunt

cfg = eqhunt.Config()
cfg.pop               = 800
cfg.gen               = 50000
cfg.tournament_size   = 5
cfg.initial_depth     = 5
cfg.bloat_penalty     = 0.3
cfg.trig_penalty      = 1.5
cfg.accepted_error    = 0.01

# Re-weight individual operators (higher = more likely to appear)
cfg.op_weights.sin = 1.0      # boost sine
cfg.op_weights.cos = 1.0
cfg.op_weights.exp = 0.0      # disable exp entirely
cfg.pi_prob = 0.10            # 'pi' more frequent in terminals

model = eqhunt.Model(cfg).fit(X, y)
print(model.formula)
```

You can also train from a CSV file (one row per sample, last column = target,
lines starting with `#` are comments):

```python
eqhunt.Model().fit_csv("nivel_embase.csv")
```

## Operators available

| Category   | Operators                        |
|------------|----------------------------------|
| Arithmetic | `+  -  *  /  -x`                 |
| Powers     | `sqrt  **`                       |
| Conditional| `if(cond, then, else)` (cond > 0)|
| Trig       | `sin  cos  tan`                  |
| Exp / log  | `exp  log`                       |
| Constants  | numeric literals, `pi`           |

Trigonometric, log and exp nodes have **low default weights** so they only
appear after enough mutation pressure — useful for cyclic / physical data,
ignored otherwise. Adjust via `Config.op_weights`.

## How error and validity are handled

* Per-sample error is `|prediction - target|`; total error is the sum.
* Invalid evaluations (`/0`, `sqrt(<0)`, `log(<=0)`, `exp(huge)`) get a soft
  per-sample penalty rather than killing the whole formula — a single
  out-of-domain sample no longer disqualifies an otherwise good candidate.
  If more than 25% of samples fail, the formula is rejected.

## Stopping early

`Config.accepted_error` stops the search as soon as total error drops below
the threshold. You can also call `model.stop()` from another thread (or a
signal handler) to ask the loop to wrap up after the current generation.

## Config reference

| Field                | Default | Meaning                                              |
|----------------------|---------|------------------------------------------------------|
| `pop`                | 400     | Population size                                      |
| `gen`                | 15000   | Max generations                                      |
| `tournament_size`    | 4       | Tournament selection pool                            |
| `crossover_prob`     | 0.7     | Crossover probability per pair                       |
| `mutation_prob`      | 0.25    | Mutation probability per offspring                   |
| `initial_depth`      | 4       | Depth used to seed the initial population            |
| `mutation_depth`     | 3       | Depth for mutation-generated subtrees                |
| `const_min/max`      | -9, 9   | Range for random numeric terminals                   |
| `pi_prob`            | 0.01    | Probability a terminal is `pi`                       |
| `bloat_penalty`      | 0.1     | Per-node penalty (favours smaller trees)             |
| `trig_penalty`       | 0.5     | Extra penalty per `sin/cos/tan/log/exp` node         |
| `immigrant_rate`     | 0.05    | Fraction of population replaced by random each gen   |
| `weak_parent_rate`   | 0.2     | Prob. 2nd parent is random (not tournament)          |
| `accepted_error`     | 0.5     | Stop training once total error < this value          |
| `verbose`            | False*  | Print best-so-far per improvement                    |
| `simplify`           | True    | Run algebraic simplification on the final tree       |
| `simplify_interval`  | 500     | Periodically simplify top-N members during training  |
| `simplify_top_n`     | 10      | How many to simplify periodically                    |

*C++ default is `True`; the Python `fit()` helper defaults to `False`.

## Building from source

```bash
git clone https://github.com/sha0coder/eqhunt
cd eqhunt
pip install -e .
pytest
```

Requires Python 3.8+, a C++17 compiler, CMake 3.15+.

## License

MIT.
