import eqhunt
import sys

cfg = eqhunt.Config()
cfg.pop = 800
cfg.gen = 50000
cfg.tournament_size = 5
cfg.initial_depth = 5
cfg.bloat_penalty = 0.3
cfg.trig_penalty = 1.5
cfg.accepted_error = 0.01


model = eqhunt.Model().fit_csv(sys.argv[1])
print("errorr level reached: ", model.error)

print(model.formula)
while True:
    inputs = [int(x) for x in input("values: ").strip().split(",")]
    print(model.predict(inputs))
