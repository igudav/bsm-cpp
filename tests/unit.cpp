#include "../bsm.h"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <tuple>

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::internal::CaptureStderr;
using ::testing::internal::GetCapturedStderr;


TEST(Ctors, CallStockOption) {
    EXPECT_THROW(bsm::CallStockOption(0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(bsm::CallStockOption(100.0, -1.0), std::invalid_argument);
    EXPECT_NO_THROW(bsm::CallStockOption(100.0, 0.0));
}

TEST(Ctors, MarketState) {
    EXPECT_THROW(bsm::MarketState(-1.0, 0.0), std::invalid_argument);
    EXPECT_NO_THROW(bsm::MarketState(0.0, -0.2));
}

TEST(Ctors, MarketData) {
    EXPECT_THROW(bsm::MarketData(-1.0), std::invalid_argument);
}

TEST(Ctors, ModelParams) {
    EXPECT_THROW(bsm::ModelParams(0.0), std::invalid_argument);
    EXPECT_NO_THROW(bsm::ModelParams{std::numeric_limits<double>::epsilon()});
}


// 3 inputs to price and grad, correct price and right answers for price and grad
using EvalTestParams = std::tuple<bsm::CallStockOption, bsm::MarketState, bsm::ModelParams, double, double>;

class EvalParametrized : public testing::TestWithParam<EvalTestParams> {};


TEST_P(EvalParametrized, Price) {
    auto test_params = GetParam();
    EXPECT_NEAR(bsm::price(std::get<0>(test_params),
                           std::get<1>(test_params),
                           std::get<2>(test_params)),
                std::get<3>(test_params),
                1e-7);
};


TEST_P(EvalParametrized, Grad) {
    auto test_params = GetParam();
    EXPECT_NEAR(bsm::model_grad(std::get<0>(test_params),
                                std::get<1>(test_params),
                                std::get<2>(test_params)).volatility(),
                std::get<4>(test_params),
                1e-7);
};


INSTANTIATE_TEST_SUITE_P(
    EvalAnswers,
    EvalParametrized,
    testing::Values(
        // CallStockOption(strike, time), MarketState(stock price, interest rate),
        // ModelParams(volatility), correct price and correct grad
        EvalTestParams({ 100.0, 0.004 }, { 100.0, 0.0 }, { 0.2 }, 0.50462314, 2.52308205987434),
        EvalTestParams({ 130.0, 0.004 }, { 100.0, 0.0 }, { 0.2 }, 0.0, 0.0),
        EvalTestParams({ 70.0, 0.004 }, { 100.0, 0.0 }, { 0.2 }, 30.0, 0.0),
        EvalTestParams({ 100.0, 1.0 }, { 100.0, 0.0 }, { 0.2 }, 7.96556746, 39.6952547477012),
        EvalTestParams({ 100.0, 10.0 }, { 100.0, 0.0 }, { 0.2 }, 24.81703660, 120.003894843014),
        EvalTestParams({ 100.0, 1.0 }, { 100.0, -0.2 }, { 0.2 }, 1.83572243, 26.6085249898755),
        EvalTestParams({ 50.0, 0.0 }, { 100.0, 0.1 }, { 0.2 }, 50.0, 0.0),
        EvalTestParams({ 150.0, 0.0 }, { 100.0, 0.1 }, { 0.2 }, 0.0, 0.0)
    )
);


// 3 inputs to calibration
using CalibrateTestParams = std::tuple<bsm::MarketData, bsm::CallStockOption, bsm::MarketState>;

class CalibParametrized : public testing::TestWithParam<CalibrateTestParams> {};

TEST(Calibration, ArgumentChecks) {
    bsm::MarketState ms{100.0, 0.05};
    bsm::AccuracyParams params{1e-7, 1e-7, 100};
    EXPECT_THROW(bsm::calibrate({ 5.0 }, { 100.0, 0.0 }, ms, params), std::invalid_argument);

    bsm::CallStockOption option{100.0, 1.0};
    EXPECT_THROW(bsm::calibrate({ 100.0 }, option, ms, params), std::invalid_argument);
    EXPECT_THROW(bsm::calibrate({ 4.87 }, option, ms, params), std::invalid_argument);
}


TEST_P(CalibParametrized, Volatility) {

    auto test_params = GetParam();

    auto md = std::get<0>(test_params);
    auto option = std::get<1>(test_params);
    auto ms = std::get<2>(test_params);
    bsm::AccuracyParams params{1e-7, 1e-7, 100 };

    CaptureStderr();
    auto model = bsm::calibrate(md, option, ms, params);
    auto out = GetCapturedStderr();
    EXPECT_TRUE(out.empty());  // no warnings produced

    double price = bsm::price(option, ms, model);
    
    // check if requirements met
    EXPECT_NEAR(price, md.call_price(), params.atol);
    EXPECT_NEAR(price / md.call_price(), 1.0, params.rtol);
};

INSTANTIATE_TEST_SUITE_P(
    CalibAnswers,
    CalibParametrized,
    testing::Values(
        // MarketData(call price), CallStockOption(strike, time), MarketState(stock price, interest rate)
        CalibrateTestParams({ 30.02}, { 70.0, 0.004 }, { 100.0, 0.05 }), // ITM, short time to expiration
        CalibrateTestParams({ 0.51 }, { 100.0, 0.004 }, { 100.0, 0.05 }), // ATM, short time to expiration
        CalibrateTestParams({ 0.01 }, { 130.0, 0.004 }, { 100.0, 0.05 }), // OTM, short time to expiration
        CalibrateTestParams({ 52.48 }, { 50.0, 1.0 }, { 100.0, 0.05 }), // ITM, mid time to expiration
        CalibrateTestParams({ 10.45 }, { 100.0, 1.0 }, { 100.0, 0.05 }), // ATM, mid time to expiration
        CalibrateTestParams({ 2.06 }, { 150.0, 1.0 }, { 100.0, 0.05 }), // OTM, mid time to expiration
        CalibrateTestParams({ 72.08 }, { 50.0, 10.0 }, { 100.0, 0.05 }), // ITM, long time to expiration
        CalibrateTestParams({ 45.19 }, { 100.0, 10.0 }, { 100.0, 0.05 }), // ATM, long time to expiration
        CalibrateTestParams({ 39.5 }, { 150.0, 10.0 }, { 100.0, 0.05 }), // OTM, long time to expiration
        CalibrateTestParams({ 4.15 }, { 100.0, 1.0 }, { 100.0, -0.1 }) // negative rates
    )
);

