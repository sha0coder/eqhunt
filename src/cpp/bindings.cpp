#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include "formula3.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace formula3;

NB_MODULE(_core, m) {
    m.doc() = "eqhunt: symbolic regression engine (C++ backend)";

    nb::class_<OpWeights>(m, "OpWeights",
        "Relative selection weights per operator. Higher = more likely to appear.")
        .def(nb::init<>())
        .def_rw("add",  &OpWeights::add)
        .def_rw("sub",  &OpWeights::sub)
        .def_rw("mul",  &OpWeights::mul)
        .def_rw("div",  &OpWeights::div_)
        .def_rw("neg",  &OpWeights::neg)
        .def_rw("sqrt", &OpWeights::sqrt_)
        .def_rw("pow",  &OpWeights::pow_)
        .def_rw("if_",  &OpWeights::if_)
        .def_rw("sin",  &OpWeights::sin_)
        .def_rw("cos",  &OpWeights::cos_)
        .def_rw("tan",  &OpWeights::tan_)
        .def_rw("log",  &OpWeights::log_)
        .def_rw("exp",  &OpWeights::exp_);

    nb::class_<Config>(m, "Config",
        "Full configuration for the symbolic regression search.")
        .def(nb::init<>())
        .def_rw("pop",               &Config::pop)
        .def_rw("gen",               &Config::gen)
        .def_rw("tournament_size",   &Config::tournament_size)
        .def_rw("crossover_prob",    &Config::crossover_prob)
        .def_rw("mutation_prob",     &Config::mutation_prob)
        .def_rw("initial_depth",     &Config::initial_depth)
        .def_rw("mutation_depth",    &Config::mutation_depth)
        .def_rw("const_min",         &Config::const_min)
        .def_rw("const_max",         &Config::const_max)
        .def_rw("pi_prob",           &Config::pi_prob)
        .def_rw("bloat_penalty",     &Config::bloat_penalty)
        .def_rw("trig_penalty",      &Config::trig_penalty)
        .def_rw("immigrant_rate",    &Config::immigrant_rate)
        .def_rw("weak_parent_rate",  &Config::weak_parent_rate)
        .def_rw("accepted_error",    &Config::accepted_error)
        .def_rw("verbose",           &Config::verbose)
        .def_rw("simplify",          &Config::simplify)
        .def_rw("simplify_interval", &Config::simplify_interval)
        .def_rw("simplify_top_n",    &Config::simplify_top_n)
        .def_rw("op_weights",        &Config::op_weights);

    nb::class_<Sample>(m, "Sample", "Single training sample: vector of inputs + scalar target.")
        .def(nb::init<>())
        .def_rw("inputs", &Sample::inputs)
        .def_rw("target", &Sample::target);

    nb::class_<Formula>(m, "Formula", "The trained model itself (C++ owned).")
        .def(nb::init<Config>(), "config"_a = Config{})
        .def("train",        &Formula::train,        "data"_a,
             "Train on a list of Sample objects.")
        .def("train_csv",    &Formula::train_csv,    "path"_a,
             "Train from a CSV file (last column = target).")
        .def("request_stop", &Formula::request_stop,
             "Ask the training loop to stop after the current generation.")
        .def("compute",      &Formula::compute,      "inputs"_a,
             "Evaluate the trained formula. Returns None if invalid.")
        .def("get_formula",  &Formula::get_formula,
             "Return the formula as a human-readable string.")
        .def("total_error",  &Formula::total_error,  "data"_a,
             "Return the sum of |pred - target| over the dataset.")
        .def("load",         &Formula::load,         "formula"_a,
             "Reload the model from a textual formula (same syntax as get_formula()). "
             "Replaces the current trained tree, no training needed.")
        .def_prop_ro("num_vars", &Formula::num_vars)
        .def_prop_ro("formula",  &Formula::get_formula);
}
