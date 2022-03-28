#include <iostream>
#include <stdexcept>

#include "bsm.h"


int main() {

    bsm::CallStockOption option = {
        90.0,  // strike_price
        0.5,  // time_to_expiration 
    };
    bsm::MarketState ms = {
        100.0,  // stock_price
        0.05,  // interest_rate
    };
    bsm::MarketData md = {
        13.49851,  // call_price
    };
    bsm::AccuracyParams params = {
        1e-7,  // atol
        1e-7,  // rtol
        2,  // max_iter
    };

    try{
        bsm::ModelParams model = bsm::calibrate(md, option, ms, params);
        std::cout << "IV: " << model.volatility() << std::endl;
    } catch (const std::invalid_argument & e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
