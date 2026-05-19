"""Smoke tests — verify the bindings import and a tiny problem trains."""

import eqhunt


def test_simple_sum():
    X = [[i, j] for i in range(5) for j in range(5)]
    y = [i + j for i, j in X]
    m = eqhunt.fit(X, y, generations=2000, pop=200, verbose=False)
    assert m.num_vars == 2
    pred = m.predict([7, 8])
    assert pred is not None
    assert abs(pred - 15) < 1.0


def test_full_config():
    cfg = eqhunt.Config()
    cfg.pop = 200
    cfg.gen = 1000
    cfg.verbose = False
    cfg.op_weights.sin = 0.5
    model = eqhunt.Model(cfg)
    X = [[i] for i in range(10)]
    y = [2 * i + 1 for i in X[0:][0]]  # placeholder
    y = [2 * row[0] + 1 for row in X]
    model.fit(X, y)
    assert model.num_vars == 1
    assert isinstance(model.formula, str)


if __name__ == "__main__":
    test_simple_sum()
    test_full_config()
    print("ok")
