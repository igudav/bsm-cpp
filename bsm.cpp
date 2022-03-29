#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>

#include "bsm.h"


namespace {
    double normal_cdf(double x) {
        constexpr double SQRT_2 = 1.4142135623730950488016887242;
        return 0.5 + 0.5 * erf(x / SQRT_2);
    }

    void check_params_for_calibration(bsm::MarketData md,
                                      bsm::CallStockOption option,
                                      bsm::MarketState ms) {

        if (option.time_to_expiration() < std::numeric_limits<double>::epsilon()) {
            throw std::invalid_argument("Time to expiration must be positive for model calibration");
        }
        if (md.call_price() >= ms.stock_price()) {
            throw std::invalid_argument("Market call price must be less than stock price");
        }
        if (md.call_price() <= ms.stock_price()
                - option.strike_price() * exp(-ms.interest_rate() * option.time_to_expiration())) {
            throw std::invalid_argument("Call price implies its time value is nonpositive");
        }
    }
}


bsm::CallStockOption::CallStockOption(double strike_price, double time_to_expiration) {
    if (strike_price < std::numeric_limits<double>::epsilon()) {
        throw std::invalid_argument("Strike price must be positive");
    }
    if (time_to_expiration < 0.0) {
        throw std::invalid_argument("Time to expiration must be nonnegative");
    }
    this->strike = strike_price;
    this->time = time_to_expiration;
}

double bsm::CallStockOption::strike_price() const { return this->strike; }

double bsm::CallStockOption::time_to_expiration() const { return this->time; }


bsm::MarketState::MarketState(double stock_price, double interest_rate) : rate{interest_rate} {
    if (stock_price < 0.0) {
        throw std::invalid_argument("Stock price must be nonnegative");
    }
    this->stock_p = stock_price;
}

double bsm::MarketState::stock_price() const { return this->stock_p; }

double bsm::MarketState::interest_rate() const { return this->rate; }


bsm::MarketData::MarketData(double call_price) {
    if (call_price < 0.0) {
        throw std::invalid_argument("Call price must be nonnegative");
    }
    this->call_p = call_price;
}

double bsm::MarketData::call_price() const { return this->call_p; }


bsm::ModelParams::ModelParams(double volatility) {
    if (volatility < std::numeric_limits<double>::epsilon()) {
        throw std::invalid_argument("Volatility must be positive");
    }
    this->vola = volatility;
}

double bsm::ModelParams::volatility() const { return this->vola; }


bsm::ModelParamsGrad::ModelParamsGrad(double volatility) : vola{volatility} {}

double bsm::ModelParamsGrad::volatility() const { return vola; }


double bsm::price(bsm::CallStockOption option,
                  bsm::MarketState ms,
                  bsm::ModelParams model) {

    if (option.time_to_expiration() < std::numeric_limits<double>::epsilon()) {
        return std::max(0.0, ms.stock_price() - option.strike_price());
    }

    double sigma_sq_t = model.volatility() * sqrt(option.time_to_expiration());
    double d1 = 1.0 / sigma_sq_t * (log(ms.stock_price() / option.strike_price())
        + (ms.interest_rate() + model.volatility() * model.volatility() / 2.0) * option.time_to_expiration());
    double d2 = d1 - sigma_sq_t;
    double discount_factor = exp(-ms.interest_rate() * option.time_to_expiration());

    return ms.stock_price() * normal_cdf(d1) - discount_factor * option.strike_price() * normal_cdf(d2);
}

bsm::ModelParamsGrad bsm::model_grad(bsm::CallStockOption option,
                                     bsm::MarketState ms,
                                     bsm::ModelParams model) {

    constexpr double SQRT_2_PI = 2.5066282746310005;
    double sqrt_t = sqrt(option.time_to_expiration());
    double d1 = 1.0 / (model.volatility() * sqrt_t) * (log(ms.stock_price() / option.strike_price()) 
        + (ms.interest_rate() + model.volatility() * model.volatility() / 2.0) * option.time_to_expiration());

    return { ms.stock_price() * sqrt_t / SQRT_2_PI * exp(-d1 * d1 / 2.0) };
}

bsm::ModelParams bsm::calibrate(bsm::MarketData md,
                                bsm::CallStockOption option,
                                bsm::MarketState ms,
                                const bsm::AccuracyParams &params) {

    constexpr double VOL_MIN = std::numeric_limits<double>::epsilon();
    constexpr double VOL_MAX = 5.0;
    constexpr double SQRT_2_PI = 2.5066282746310005;

    check_params_for_calibration(md, option, ms);

    double init_vol = md.call_price() / ms.stock_price() * SQRT_2_PI / sqrt(option.time_to_expiration());
    init_vol = std::max(init_vol, VOL_MIN);
    init_vol = std::min(init_vol, VOL_MAX);
    bsm::ModelParams model{init_vol};

    double p = bsm::price(option, ms, model);
    auto grad = bsm::model_grad(option, ms, model);

    double lb = VOL_MIN;
    double ub = VOL_MAX;
    
    unsigned iters = 0;
    for (; iters < params.max_iter; ++iters) {

        double abs_error = fabs(p - md.call_price());

        // accuracy requirements met
        if (abs_error < params.atol and abs_error / md.call_price() < params.rtol) {
            break;
        }

        double new_vola;
        bool newton_ok = false;
        if (fabs(grad.volatility()) > std::numeric_limits<double>::epsilon()) {  // not dividing by zero

            new_vola = model.volatility() - (p - md.call_price()) / grad.volatility();  // Newton step

            if (new_vola >= lb and new_vola <= ub) {  // limits are not exceeded
                model = { new_vola };
                p = bsm::price(option, ms, model);
                grad = bsm::model_grad(option, ms, model);
                newton_ok = true;
            }
        } 

        if (not newton_ok) {  // fall to bisect
            new_vola = 0.5 * (lb + ub);
            model = { new_vola };
            p = bsm::price(option, ms, model);
            grad = bsm::model_grad(option, ms, model);
        }

        if (p > md.call_price()) {
            ub = new_vola;
        } else {
            lb = new_vola;
        }
    }

    if (iters == params.max_iter) {
        std::cerr << "WARNING: Maximum iterations threshold exceeded, IV may be inaccurate. Inputs: "
            << "Option price: " << md.call_price() << ", "
            << "Strike: " << option.strike_price() << ", "
            << "Time to expiration: " << option.time_to_expiration() << ", "
            << "Stock price: " << ms.stock_price() << ", " 
            << "Interest rate: " << ms.interest_rate() << "\n";
    }

    return model;
}

