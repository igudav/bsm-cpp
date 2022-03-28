#ifndef BSM_H
#define BSM_H

namespace bsm {
    struct AccuracyParams {
        double atol;
        double rtol;
        unsigned max_iter;
    };

    class CallStockOption {
        double strike;
        double time;
        
    public:
        CallStockOption(double strike_price, double time_to_expiration);

        double strike_price() const;

        double time_to_expiration() const;
    };

    class MarketState {
        double stock_p;
        double rate;

    public:
        MarketState(double stock_price, double interest_rate);

        double stock_price() const;

        double interest_rate() const;
    };

    class MarketData {
        double call_p;

    public:
        MarketData(double call_price);

        double call_price() const;
    };

    class ModelParams {
        double vola;

    public:
        ModelParams(double volatility);

        double volatility() const;
    };

    class ModelParamsGrad {
        double vola;

    public:
        ModelParamsGrad(double volatility);

        double volatility() const;
    };


    double price(CallStockOption option,
                 MarketState ms,
                 ModelParams model);

    ModelParamsGrad model_grad(CallStockOption option,
                               MarketState ms,
                               ModelParams model);
    
    ModelParams calibrate(MarketData md,
                          CallStockOption option,
                          MarketState ms,
                          const AccuracyParams &params);
}

#endif

