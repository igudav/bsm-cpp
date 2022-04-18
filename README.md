# Модель Блэка–Шоулза

Тестовое задание в один HFT фонд.
Задача написать программу на C++ для рассчета IV, которую можно сразу на прод пустить.
Входные данные задаются в исходном коде, ошибка должны быть обработаны, 
код должен быть быстрый, без лишних вычислений и откровенных потерь времени.
Рассчет IV в коде смотрится как калибровка модели Блэка–Шоулза.
Структура программы и названия функций выбирались из рассчета,
что если захочется использовать другую модель, например, Хестона,
то достаточно будт поменять в коде название пространства имен, а логика останется прежней.
В папке ``benchmark_results`` находятся результаты для 3-х разных алгоритмов.
Из них видно, что можно включать метод разбиения отрезка пополам, 
когда экспирация скоро и опцион вне денег, а в остальных случаях использовать безопасный метод Ньютона.

## Сборка

### Вариант 1
Просто чтобы запустить.
```bash
g++ main.cpp bsm.cpp
./a.out
```

### Вариант 2
Сборка через CMake, простой вариант, только main.
```bash
mkdir build
cd build
cmake -S .. -B .
cmake --build .
./main
```
### Вариант 3
В этом варианте CMake подтягивает GTest и Google Benchmark для юнит-тестов и тестов производительности.
Тесты производитльности включают расчет цены, производной цены по параметрам модели (вега)
и калибровку для набора страйков и времен до экспирации.
```bash
mkdir build
cd build
cmake -S .. -B . -DWITH_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
# IV computation
./main
# unit tests
./unit_tests
# google benchmark
./performance_tests
```
