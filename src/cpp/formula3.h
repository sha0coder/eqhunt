#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace formula3 {

struct Sample {
    std::vector<double> inputs;
    double              target;
};

// Relative weights for operator selection. Higher weight = more likely.
// Defaults: arithmetic >> sqrt/pow/if >> trig/log/exp.
struct OpWeights {
    double add = 10,   sub = 10,    mul = 10,   div_ = 6;
    double neg = 4,    sqrt_ = 3,   pow_ = 3,   if_ = 3;
    // trig/log/exp: very low weights -> only appear after many generations (mutation)
    double sin_ = 0.05, cos_  = 0.05, tan_ = 0.05;
    double log_ = 0.05, exp_  = 0.05;
};

struct Config {
    int    pop               = 400;
    int    gen               = 15000;
    int    tournament_size   = 4;
    double crossover_prob    = 0.7;
    double mutation_prob     = 0.25;
    int    initial_depth     = 4;
    int    mutation_depth    = 3;
    double const_min         = -9.0;
    double const_max         =  9.0;
    double pi_prob           = 0.01;   // probability a terminal is 'pi' (rare)
    double bloat_penalty     = 0.1;
    double trig_penalty      = 0.5;    // extra penalty per sin/cos/tan/log/exp node in fitness
    double immigrant_rate    = 0.05;
    double weak_parent_rate  = 0.2;
    double accepted_error    = 0.5;
    bool   verbose           = true;
    bool   simplify          = true;
    int    simplify_interval = 500;
    int    simplify_top_n    = 10;
    OpWeights op_weights;
};

class Formula {
public:
    explicit Formula(Config cfg = {});
    Formula(int pop, int gen, double accepted_error);
    ~Formula();

    Formula(const Formula&)            = delete;
    Formula& operator=(const Formula&) = delete;
    Formula(Formula&&) noexcept;
    Formula& operator=(Formula&&) noexcept;

    void train(const std::vector<Sample>& data);
    void train_csv(const std::string& path);
    void request_stop();

    std::optional<double> compute(const std::vector<double>& inputs) const;
    std::string           get_formula() const;
    int                   num_vars() const;
    double                total_error(const std::vector<Sample>& data) const;

    // Parse a textual formula (in the same syntax as get_formula()) and store
    // it as the current model. Accepts both bare expressions ("(x+y)") and
    // prefixed forms ("f(x,y) = (x+y)"). Throws std::runtime_error on parse
    // failure. numVars is set from the highest variable index seen.
    void load(const std::string& formula);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace formula3
