#include "formula3.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace formula3 {

namespace {

struct Node {
    enum Kind {
        Num, Var, Pi,
        Add, Sub, Mul, Div,
        Neg, Sqrt, Pow,
        If,
        Sin, Cos, Tan,
        Log, Exp,
    };
    Kind   kind;
    double value = 0.0;   // Num: constant. Var: variable index (integer stored as double).
    std::unique_ptr<Node> lhs, rhs, third;  // 'third' is only used in If (else branch)
};

using NodePtr = std::unique_ptr<Node>;

constexpr double kPi       = 3.141592653589793;
constexpr double kBigBound = 1e15;
constexpr double kTinyDiv  = 1e-12;
constexpr double kExpLim   = 100.0;

inline bool isOk(double v) { return std::isfinite(v) && std::abs(v) < kBigBound; }

std::string varName(int i) {
    static const char* names[] = {"x", "y", "z", "w", "v", "u"};
    if (i < 6) return names[i];
    return "x" + std::to_string(i);
}

std::string fmtNum(double v) {
    std::ostringstream os;
    if (std::floor(v) == v && std::abs(v) < 1e15) {
        os << (long long)v;
    } else {
        os.precision(4);
        os << v;
    }
    return os.str();
}

NodePtr clone(const Node& n) {
    auto c = std::make_unique<Node>();
    c->kind  = n.kind;
    c->value = n.value;
    if (n.lhs)   c->lhs   = clone(*n.lhs);
    if (n.rhs)   c->rhs   = clone(*n.rhs);
    if (n.third) c->third = clone(*n.third);
    return c;
}

bool evalNode(const Node& n, const std::vector<double>& vars, double& out) {
    switch (n.kind) {
        case Node::Num: out = n.value;            return true;
        case Node::Var: out = vars[(int)n.value]; return true;
        case Node::Pi:  out = kPi;                return true;
        case Node::Neg: {
            double v;
            if (!evalNode(*n.lhs, vars, v)) return false;
            out = -v; return isOk(out);
        }
        case Node::Sqrt: {
            double v;
            if (!evalNode(*n.lhs, vars, v)) return false;
            if (v < 0) return false;
            out = std::sqrt(v); return isOk(out);
        }
        case Node::Sin: {
            double v;
            if (!evalNode(*n.lhs, vars, v)) return false;
            out = std::sin(v); return isOk(out);
        }
        case Node::Cos: {
            double v;
            if (!evalNode(*n.lhs, vars, v)) return false;
            out = std::cos(v); return isOk(out);
        }
        case Node::Tan: {
            double v;
            if (!evalNode(*n.lhs, vars, v)) return false;
            out = std::tan(v); return isOk(out);
        }
        case Node::Log: {
            double v;
            if (!evalNode(*n.lhs, vars, v)) return false;
            if (v <= 0.0) return false;
            out = std::log(v); return isOk(out);
        }
        case Node::Exp: {
            double v;
            if (!evalNode(*n.lhs, vars, v)) return false;
            if (v > kExpLim || v < -kExpLim) return false;
            out = std::exp(v); return isOk(out);
        }
        case Node::If: {
            double c;
            if (!evalNode(*n.lhs, vars, c)) return false;
            return c > 0.0 ? evalNode(*n.rhs,   vars, out)
                           : evalNode(*n.third, vars, out);
        }
        default: {
            double a, b;
            if (!evalNode(*n.lhs, vars, a)) return false;
            if (!evalNode(*n.rhs, vars, b)) return false;
            switch (n.kind) {
                case Node::Add: out = a + b; return isOk(out);
                case Node::Sub: out = a - b; return isOk(out);
                case Node::Mul: out = a * b; return isOk(out);
                case Node::Div:
                    if (std::abs(b) < kTinyDiv) return false;
                    out = a / b; return isOk(out);
                case Node::Pow: {
                    if (!std::isfinite(b)) return false;
                    if (b > 12.0 || b < -12.0) return false;
                    if (a < 0.0 && std::floor(b) != b) return false;
                    if (a == 0.0 && b <= 0.0) return false;
                    out = std::pow(a, b);
                    return isOk(out);
                }
                default: return false;
            }
        }
    }
}

std::string toString(const Node& n) {
    switch (n.kind) {
        case Node::Num:  return fmtNum(n.value);
        case Node::Var:  return varName((int)n.value);
        case Node::Pi:   return "pi";
        case Node::Neg:  return "(-" + toString(*n.lhs) + ")";
        case Node::Sqrt: return "sqrt(" + toString(*n.lhs) + ")";
        case Node::Sin:  return "sin("  + toString(*n.lhs) + ")";
        case Node::Cos:  return "cos("  + toString(*n.lhs) + ")";
        case Node::Tan:  return "tan("  + toString(*n.lhs) + ")";
        case Node::Log:  return "log("  + toString(*n.lhs) + ")";
        case Node::Exp:  return "exp("  + toString(*n.lhs) + ")";
        case Node::Add:  return "(" + toString(*n.lhs) + "+"  + toString(*n.rhs) + ")";
        case Node::Sub:  return "(" + toString(*n.lhs) + "-"  + toString(*n.rhs) + ")";
        case Node::Mul:  return "(" + toString(*n.lhs) + "*"  + toString(*n.rhs) + ")";
        case Node::Div:  return "(" + toString(*n.lhs) + "/"  + toString(*n.rhs) + ")";
        case Node::Pow:  return "(" + toString(*n.lhs) + "**" + toString(*n.rhs) + ")";
        case Node::If:   return "if(" + toString(*n.lhs) + ","
                              + toString(*n.rhs) + "," + toString(*n.third) + ")";
    }
    return "?";
}

int treeSize(const Node& n) {
    int s = 1;
    if (n.lhs)   s += treeSize(*n.lhs);
    if (n.rhs)   s += treeSize(*n.rhs);
    if (n.third) s += treeSize(*n.third);
    return s;
}

// counts "costly" nodes (trig + log + exp) so the fitness can penalize them
int costlyCount(const Node& n) {
    int s = 0;
    switch (n.kind) {
        case Node::Sin: case Node::Cos: case Node::Tan:
        case Node::Log: case Node::Exp:
            s = 1; break;
        default: break;
    }
    if (n.lhs)   s += costlyCount(*n.lhs);
    if (n.rhs)   s += costlyCount(*n.rhs);
    if (n.third) s += costlyCount(*n.third);
    return s;
}

// catalog of non-terminal operators with their arity
struct OpEntry { Node::Kind kind; int arity; };
static const std::array<OpEntry, 13> kOps = {{
    {Node::Add, 2}, {Node::Sub, 2}, {Node::Mul, 2}, {Node::Div, 2},
    {Node::Neg, 1}, {Node::Sqrt,1}, {Node::Pow, 2}, {Node::If,  3},
    {Node::Sin, 1}, {Node::Cos, 1}, {Node::Tan, 1},
    {Node::Log, 1}, {Node::Exp, 1},
}};

std::array<double, 13> weightArray(const OpWeights& w) {
    return { w.add, w.sub, w.mul, w.div_,
             w.neg, w.sqrt_, w.pow_, w.if_,
             w.sin_, w.cos_, w.tan_,
             w.log_, w.exp_ };
}

int arityOf(Node::Kind k) {
    for (const auto& op : kOps) if (op.kind == k) return op.arity;
    return 0;  // terminals: Num, Var, Pi
}

NodePtr randomTerminal(std::mt19937& rng, int numVars, double cmin, double cmax, double piProb) {
    auto n = std::make_unique<Node>();
    double r = std::uniform_real_distribution<>(0.0, 1.0)(rng);
    if (r < piProb) {
        n->kind = Node::Pi;
    } else if (std::uniform_int_distribution<>(0, 2)(rng) != 0) {
        n->kind  = Node::Var;
        n->value = (double)std::uniform_int_distribution<>(0, numVars - 1)(rng);
    } else {
        n->kind  = Node::Num;
        n->value = std::uniform_real_distribution<>(cmin, cmax)(rng);
    }
    return n;
}

NodePtr randomTree(std::mt19937& rng, int depth, int numVars,
                   double cmin, double cmax, double piProb,
                   std::discrete_distribution<>& opDist) {
    if (depth <= 0 || std::uniform_int_distribution<>(0, 3)(rng) == 0)
        return randomTerminal(rng, numVars, cmin, cmax, piProb);
    auto n = std::make_unique<Node>();
    const auto& op = kOps[opDist(rng)];
    n->kind = op.kind;
    n->lhs  = randomTree(rng, depth - 1, numVars, cmin, cmax, piProb, opDist);
    if (op.arity >= 2)
        n->rhs = randomTree(rng, depth - 1, numVars, cmin, cmax, piProb, opDist);
    if (op.arity == 3)
        n->third = randomTree(rng, depth - 1, numVars, cmin, cmax, piProb, opDist);
    return n;
}

void collect(NodePtr& root, std::vector<NodePtr*>& out) {
    out.push_back(&root);
    if (root->lhs)   collect(root->lhs,   out);
    if (root->rhs)   collect(root->rhs,   out);
    if (root->third) collect(root->third, out);
}

void mutateTree(NodePtr& root, std::mt19937& rng, int numVars, const Config& cfg,
                std::discrete_distribution<>& opDist) {
    std::vector<NodePtr*> nodes;
    collect(root, nodes);
    auto& pick = *nodes[std::uniform_int_distribution<>(0, (int)nodes.size() - 1)(rng)];
    pick = randomTree(rng, cfg.mutation_depth, numVars,
                      cfg.const_min, cfg.const_max, cfg.pi_prob, opDist);
}

void crossoverTrees(NodePtr& a, NodePtr& b, std::mt19937& rng) {
    std::vector<NodePtr*> na, nb;
    collect(a, na);
    collect(b, nb);
    auto& pa = *na[std::uniform_int_distribution<>(0, (int)na.size() - 1)(rng)];
    auto& pb = *nb[std::uniform_int_distribution<>(0, (int)nb.size() - 1)(rng)];
    pa.swap(pb);
}

// Soft per-sample failure: a single invalid eval adds a fixed penalty but does not
// kill the whole formula. If more than 25% of samples fail, the formula is rejected.
double rawError(const Node& n, const std::vector<Sample>& data) {
    constexpr double kBadPenalty = 1e3;
    double err = 0;
    int bad = 0;
    for (const auto& s : data) {
        double v;
        if (!evalNode(n, s.inputs, v)) { err += kBadPenalty; ++bad; }
        else                            err += std::abs(v - s.target);
    }
    if (bad * 4 > (int)data.size()) return 1e18;
    return err;
}

bool sameTree(const Node& a, const Node& b) {
    if (a.kind != b.kind) return false;
    if (a.kind == Node::Num) return a.value == b.value;
    if (a.kind == Node::Var) return a.value == b.value;
    if (a.kind == Node::Pi)  return true;
    if (!sameTree(*a.lhs, *b.lhs)) return false;
    int arity = arityOf(a.kind);
    if (arity == 1) return true;
    if (!sameTree(*a.rhs, *b.rhs)) return false;
    if (arity == 3) return sameTree(*a.third, *b.third);
    return true;
}

NodePtr simplifyOnce(NodePtr n) {
    if (!n) return nullptr;
    if (n->lhs)   n->lhs   = simplifyOnce(std::move(n->lhs));
    if (n->rhs)   n->rhs   = simplifyOnce(std::move(n->rhs));
    if (n->third) n->third = simplifyOnce(std::move(n->third));

    auto makeNum = [&](double v) -> NodePtr {
        n->kind = Node::Num; n->value = v;
        n->lhs.reset(); n->rhs.reset(); n->third.reset();
        return std::move(n);
    };

    bool lNum   = n->lhs && n->lhs->kind == Node::Num;
    bool rNum   = n->rhs && n->rhs->kind == Node::Num;
    double lV   = lNum ? n->lhs->value : 0;
    double rV   = rNum ? n->rhs->value : 0;

    switch (n->kind) {
        case Node::Neg:
            if (lNum) return makeNum(-lV);
            if (n->lhs->kind == Node::Neg) return std::move(n->lhs->lhs);
            break;
        case Node::Sqrt:
            if (lNum && lV >= 0) {
                double v = std::sqrt(lV);
                if (isOk(v)) return makeNum(v);
            }
            break;
        case Node::Sin:
            if (lNum) { double v = std::sin(lV); if (isOk(v)) return makeNum(v); }
            break;
        case Node::Cos:
            if (lNum) { double v = std::cos(lV); if (isOk(v)) return makeNum(v); }
            break;
        case Node::Tan:
            if (lNum) { double v = std::tan(lV); if (isOk(v)) return makeNum(v); }
            break;
        case Node::Log:
            if (lNum && lV > 0) { double v = std::log(lV); if (isOk(v)) return makeNum(v); }
            break;
        case Node::Exp:
            if (lNum && lV <= kExpLim && lV >= -kExpLim) {
                double v = std::exp(lV); if (isOk(v)) return makeNum(v);
            }
            break;
        case Node::If:
            if (lNum) return lV > 0 ? std::move(n->rhs) : std::move(n->third);
            if (sameTree(*n->rhs, *n->third)) return std::move(n->rhs);
            break;
        case Node::Add:
            if (lNum && rNum) return makeNum(lV + rV);
            if (rNum && rV == 0) return std::move(n->lhs);
            if (lNum && lV == 0) return std::move(n->rhs);
            break;
        case Node::Sub:
            if (lNum && rNum) return makeNum(lV - rV);
            if (rNum && rV == 0) return std::move(n->lhs);
            if (lNum && lV == 0) { n->kind = Node::Neg; n->lhs = std::move(n->rhs); return n; }
            if (sameTree(*n->lhs, *n->rhs)) return makeNum(0);
            break;
        case Node::Mul:
            if (lNum && rNum) return makeNum(lV * rV);
            if ((lNum && lV == 0) || (rNum && rV == 0)) return makeNum(0);
            if (rNum && rV == 1) return std::move(n->lhs);
            if (lNum && lV == 1) return std::move(n->rhs);
            if (rNum && rV == -1) { n->kind = Node::Neg; n->rhs.reset(); return n; }
            if (lNum && lV == -1) { n->kind = Node::Neg; n->lhs = std::move(n->rhs); n->rhs.reset(); return n; }
            break;
        case Node::Div:
            if (lNum && rNum && rV != 0) return makeNum(lV / rV);
            if (rNum && rV == 1)  return std::move(n->lhs);
            if (rNum && rV == -1) { n->kind = Node::Neg; n->rhs.reset(); return n; }
            if (lNum && lV == 0 && rNum && rV != 0) return makeNum(0);
            break;
        case Node::Pow:
            if (lNum && rNum) {
                double r = std::pow(lV, rV);
                if (isOk(r)) return makeNum(r);
            }
            if (rNum && rV == 0) return makeNum(1);
            if (rNum && rV == 1) return std::move(n->lhs);
            if (lNum && lV == 1) return makeNum(1);
            if (lNum && lV == 0 && rNum && rV > 0) return makeNum(0);
            break;
        default: break;
    }
    return n;
}

NodePtr simplifyTree(NodePtr n) {
    for (int i = 0; i < 50; ++i) {
        std::string before = toString(*n);
        n = simplifyOnce(std::move(n));
        if (toString(*n) == before) break;
    }
    return n;
}

std::vector<Sample> loadCsv(const std::string& path, int& numVars) {
    std::vector<Sample> data;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open: " + path);
    std::string line;
    numVars = -1;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        if (line[first] == '#') continue;
        std::vector<double> cols;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) cols.push_back(std::stod(tok));
        if (cols.size() < 2)
            throw std::runtime_error("row needs >= 2 columns (input(s) + target)");
        if (numVars == -1) numVars = (int)cols.size() - 1;
        else if ((int)cols.size() - 1 != numVars)
            throw std::runtime_error("inconsistent column count");
        Sample s;
        s.target = cols.back();
        cols.pop_back();
        s.inputs = std::move(cols);
        data.push_back(std::move(s));
    }
    if (numVars <= 0) throw std::runtime_error("empty or invalid CSV");
    return data;
}

}  // namespace

// ====== PIMPL ======

struct Formula::Impl {
    Config            cfg;
    NodePtr           best;
    int               numVars = 0;
    std::mt19937      rng;
    std::atomic<bool> stop_requested{false};
    std::discrete_distribution<> opDist;

    explicit Impl(Config c) : cfg(std::move(c)), rng(std::random_device{}()) {
        auto w = weightArray(cfg.op_weights);
        opDist = std::discrete_distribution<>(w.begin(), w.end());
    }

    std::string fnHeader() const {
        std::string s = "f(";
        for (int i = 0; i < numVars; ++i) s += (i ? "," : "") + varName(i);
        return s + ")";
    }

    void train(const std::vector<Sample>& data) {
        if (data.empty()) throw std::runtime_error("empty dataset");
        numVars = (int)data[0].inputs.size();
        if (numVars <= 0) throw std::runtime_error("samples without inputs");

        stop_requested.store(false);

        std::vector<NodePtr> pop;
        pop.reserve(cfg.pop);
        for (int i = 0; i < cfg.pop; ++i)
            pop.push_back(randomTree(rng, cfg.initial_depth, numVars,
                                     cfg.const_min, cfg.const_max, cfg.pi_prob, opDist));

        NodePtr bestEver;
        double  bestFit = -1e18;

        for (int g = 0; g < cfg.gen; ++g) {
            if (stop_requested.load()) {
                if (cfg.verbose)
                    std::cout << "\ntraining interrupted by user at gen " << g << "\n";
                break;
            }
            std::vector<double> fits(cfg.pop);
            std::vector<double> raws(cfg.pop);
            for (int i = 0; i < cfg.pop; ++i) {
                raws[i] = rawError(*pop[i], data);
                if (raws[i] >= 1e17) {
                    fits[i] = -1e18;
                } else {
                    fits[i] = -raws[i]
                              - cfg.bloat_penalty * treeSize(*pop[i])
                              - cfg.trig_penalty  * costlyCount(*pop[i]);
                }
            }

            if (cfg.simplify_interval > 0 && g > 0 && g % cfg.simplify_interval == 0) {
                int N = std::min(cfg.simplify_top_n, cfg.pop);
                std::vector<int> idx(cfg.pop);
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + N, idx.end(),
                                  [&](int a, int b) { return fits[a] > fits[b]; });
                for (int j = 0; j < N; ++j) {
                    int i = idx[j];
                    NodePtr s = simplifyTree(clone(*pop[i]));
                    double err_new = rawError(*s, data);
                    if (err_new <= raws[i]) {
                        pop[i]  = std::move(s);
                        raws[i] = err_new;
                        fits[i] = (err_new >= 1e17) ? -1e18
                                                    : -err_new
                                                      - cfg.bloat_penalty * treeSize(*pop[i])
                                                      - cfg.trig_penalty  * costlyCount(*pop[i]);
                    }
                }
            }

            auto it = std::max_element(fits.begin(), fits.end());
            int bestIdx = (int)(it - fits.begin());

            if (*it > bestFit) {
                bestFit = *it;
                bestEver = clone(*pop[bestIdx]);
                if (cfg.verbose) {
                    std::cout << "\033[2J\033[H"
                              << "gen " << g << "  fit=" << bestFit
                              << "  err=" << raws[bestIdx]
                              << "  " << fnHeader() << " = " << toString(*bestEver) << "\n";
                }
            }
            if (raws[bestIdx] < cfg.accepted_error) {
                if (cfg.verbose)
                    std::cout << "error " << raws[bestIdx]
                              << " < " << cfg.accepted_error
                              << " reached at gen " << g << ", stopping\n";
                break;
            }

            auto tournament = [&]() {
                int b = std::uniform_int_distribution<>(0, cfg.pop - 1)(rng);
                for (int i = 1; i < cfg.tournament_size; ++i) {
                    int c = std::uniform_int_distribution<>(0, cfg.pop - 1)(rng);
                    if (fits[c] > fits[b]) b = c;
                }
                return clone(*pop[b]);
            };

            std::vector<NodePtr> next;
            next.reserve(cfg.pop);
            next.push_back(clone(*pop[bestIdx]));   // elitism

            std::uniform_real_distribution<> u01(0.0, 1.0);

            const int imm = (int)(cfg.pop * cfg.immigrant_rate);
            for (int i = 0; i < imm && (int)next.size() < cfg.pop; ++i)
                next.push_back(randomTree(rng, cfg.initial_depth, numVars,
                                          cfg.const_min, cfg.const_max, cfg.pi_prob, opDist));

            while ((int)next.size() < cfg.pop) {
                auto a = tournament();
                auto b = (u01(rng) < cfg.weak_parent_rate)
                    ? clone(*pop[std::uniform_int_distribution<>(0, cfg.pop - 1)(rng)])
                    : tournament();
                if (u01(rng) < cfg.crossover_prob) crossoverTrees(a, b, rng);
                if (u01(rng) < cfg.mutation_prob)  mutateTree(a, rng, numVars, cfg, opDist);
                if (u01(rng) < cfg.mutation_prob)  mutateTree(b, rng, numVars, cfg, opDist);
                next.push_back(std::move(a));
                if ((int)next.size() < cfg.pop) next.push_back(std::move(b));
            }
            pop = std::move(next);
        }

        if (cfg.simplify && bestEver) {
            NodePtr s = simplifyTree(clone(*bestEver));
            if (rawError(*s, data) <= rawError(*bestEver, data))
                bestEver = std::move(s);
        }

        best = std::move(bestEver);
    }
};

// ====== API publica ======

Formula::Formula(Config cfg) : pimpl_(std::make_unique<Impl>(std::move(cfg))) {}

Formula::Formula(int pop, int gen, double accepted_error) {
    Config c;
    c.pop = pop;
    c.gen = gen;
    c.accepted_error = accepted_error;
    pimpl_ = std::make_unique<Impl>(std::move(c));
}

Formula::~Formula() = default;
Formula::Formula(Formula&&) noexcept = default;
Formula& Formula::operator=(Formula&&) noexcept = default;

void Formula::train(const std::vector<Sample>& data) {
    pimpl_->train(data);
}

void Formula::request_stop() {
    if (pimpl_) pimpl_->stop_requested.store(true);
}

void Formula::train_csv(const std::string& path) {
    int nv = 0;
    auto data = loadCsv(path, nv);
    if (pimpl_->cfg.verbose) {
        std::cout << "dataset: " << data.size() << " samples, "
                  << nv << " variable(s) from " << path << "\n";
    }
    pimpl_->train(data);
}

std::optional<double> Formula::compute(const std::vector<double>& inputs) const {
    if (!pimpl_->best) return std::nullopt;
    double v;
    if (!evalNode(*pimpl_->best, inputs, v)) return std::nullopt;
    return v;
}

std::string Formula::get_formula() const {
    if (!pimpl_->best) return "<not trained>";
    return pimpl_->fnHeader() + " = " + toString(*pimpl_->best);
}

int Formula::num_vars() const {
    return pimpl_->numVars;
}

double Formula::total_error(const std::vector<Sample>& data) const {
    if (!pimpl_->best) return 1e18;
    return rawError(*pimpl_->best, data);
}

}  // namespace formula3
