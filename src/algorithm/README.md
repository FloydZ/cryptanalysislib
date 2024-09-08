Historgram:
```
2024-09-08T21:57:36+02:00
Running ./bench/algorithm/bench_algorithm_histogram
Run on (16 X 1700 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 1280 KiB (x8)
  L3 Unified 12288 KiB (x1)
Load Average: 2.44, 2.03, 1.39
----------------------------------------------------------------------------
Benchmark                                  Time             CPU   Iterations
----------------------------------------------------------------------------
BM_stupid_histogram<uint8_t>/32         21.0 ns         20.9 ns     34044067
BM_stupid_histogram<uint8_t>/64         39.7 ns         39.5 ns     17640014
BM_stupid_histogram<uint8_t>/128        90.6 ns         90.0 ns      8269706
BM_stupid_histogram<uint8_t>/256         161 ns          160 ns      4322937
BM_stupid_histogram<uint8_t>/512         299 ns          298 ns      2088915
BM_stupid_histogram<uint8_t>/1024        580 ns          579 ns      1171529
BM_stupid_histogram<uint8_t>/2048       1155 ns         1152 ns       611537
BM_stupid_histogram<uint8_t>/4096       2293 ns         2288 ns       306648
BM_histogram<uint8_t>/32                27.8 ns         27.8 ns     24208819
BM_histogram<uint8_t>/64                43.8 ns         43.7 ns     16029984
BM_histogram<uint8_t>/128               81.4 ns         81.2 ns      8528715
BM_histogram<uint8_t>/256                161 ns          160 ns      4553701
BM_histogram<uint8_t>/512                297 ns          296 ns      2312418
BM_histogram<uint8_t>/1024               594 ns          589 ns      1187817
BM_histogram<uint8_t>/2048              1164 ns         1161 ns       600661
BM_histogram<uint8_t>/4096              2310 ns         2303 ns       301251
BM_histogram_4x<uint8_t>/32              127 ns          126 ns      5565398
BM_histogram_4x<uint8_t>/64              143 ns          143 ns      4992559
BM_histogram_4x<uint8_t>/128             178 ns          177 ns      3973236
BM_histogram_4x<uint8_t>/256             252 ns          251 ns      2741621
BM_histogram_4x<uint8_t>/512             410 ns          409 ns      1745065
BM_histogram_4x<uint8_t>/1024            707 ns          705 ns       994079
BM_histogram_4x<uint8_t>/2048           1314 ns         1310 ns       531911
BM_histogram_4x<uint8_t>/4096           2535 ns         2528 ns       277725
```
