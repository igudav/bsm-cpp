#include "../bsm.h"

#include <benchmark/benchmark.h>

static void BM_price(benchmark::State &state) {
    for (auto _ : state) {
        bsm::price({ 90.0, 0.5 }, { 100.0, 0.05 }, { 0.2 });
    }
}
BENCHMARK(BM_price);

static void BM_grad(benchmark::State &state) {
    for (auto _ : state) {
        bsm::model_grad({ 100.0, 0.5 }, { 100.0, 0.05 }, { 0.2 });
    }
}
BENCHMARK(BM_grad);


constexpr bsm::AccuracyParams params{1e-7, 1e-7, 100};
constexpr size_t N_STRIKES = 3;
constexpr size_t N_EXPIRATIONS = 3;
constexpr double vol_surface[N_EXPIRATIONS][N_STRIKES] = {
    { 1.47, 0.18, 1.47 },
    { 0.58, 0.23, 0.57 },
    { 0.41, 0.31, 0.41 }
};
constexpr double expirations[N_EXPIRATIONS] = { 0.004, 1.0, 10.0 };
constexpr double strikes[N_STRIKES] = { 50.0, 100.0, 150.0 };

static void BM_calibrate(benchmark::State &state) {

    bsm::CallStockOption option{strikes[state.range(1)], expirations[state.range(0)]};
    bsm::MarketState ms{100.0, 0.05};
    bsm::ModelParams model{vol_surface[state.range(0)][state.range(1)]};

    state.counters["T"] = expirations[state.range(0)];
    state.counters["K"] = strikes[state.range(1)];
    state.counters["sigma"] = vol_surface[state.range(0)][state.range(1)];

    // calc call price
    double p = bsm::price(option, ms, model);

    for (auto _ : state) {
        bsm::calibrate({ p }, option, ms, params);
    }
}
BENCHMARK(BM_calibrate)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, N_EXPIRATIONS - 1, 1),
        benchmark::CreateDenseRange(0, N_STRIKES - 1, 1)
    });

BENCHMARK_MAIN();

