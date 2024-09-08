Historgram:
```
2024-09-08T22:34:54+02:00
Running ./bench/algorithm/bench_algorithm_histogram
Run on (16 X 1700 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 1280 KiB (x8)
  L3 Unified 12288 KiB (x1)
Load Average: 2.45, 2.76, 2.41
----------------------------------------------------------------------------
Benchmark                                  Time             CPU   Iterations
----------------------------------------------------------------------------
BM_histogram<uint8_t>/32                28.4 ns         28.3 ns     25778073
BM_histogram<uint8_t>/64                46.3 ns         46.2 ns     15403129
BM_histogram<uint8_t>/128               82.2 ns         81.9 ns      8628664
BM_histogram<uint8_t>/256                159 ns          159 ns      4505131
BM_histogram<uint8_t>/512                311 ns          309 ns      2096917
BM_histogram<uint8_t>/1024               581 ns          579 ns      1191687
BM_histogram<uint8_t>/2048              1159 ns         1156 ns       608258
BM_histogram<uint8_t>/4096              2314 ns         2306 ns       301515
BM_histogram_4x<uint8_t>/32              126 ns          125 ns      5541088
BM_histogram_4x<uint8_t>/64              141 ns          140 ns      4899202
BM_histogram_4x<uint8_t>/128             178 ns          177 ns      3994807
BM_histogram_4x<uint8_t>/256             249 ns          248 ns      2735126
BM_histogram_4x<uint8_t>/512             400 ns          399 ns      1752857
BM_histogram_4x<uint8_t>/1024            705 ns          703 ns       987254
BM_histogram_4x<uint8_t>/2048           1317 ns         1313 ns       533448
BM_histogram_4x<uint8_t>/4096           2960 ns         2950 ns       170376
BM_histogram_8x<uint8_t>/32              222 ns          221 ns      3188463
BM_histogram_8x<uint8_t>/64              234 ns          234 ns      3023846
BM_histogram_8x<uint8_t>/128             284 ns          282 ns      2610931
BM_histogram_8x<uint8_t>/256             390 ns          387 ns      1981166
BM_histogram_8x<uint8_t>/512             548 ns          543 ns      1389815
BM_histogram_8x<uint8_t>/1024            909 ns          904 ns       773254
BM_histogram_8x<uint8_t>/2048           1464 ns         1460 ns       446914
BM_histogram_8x<uint8_t>/4096           2711 ns         2702 ns       262543
BM_stupid_histogram<uint8_t>/32         24.2 ns         24.1 ns     29333135
BM_stupid_histogram<uint8_t>/64         40.4 ns         40.3 ns     16681535
BM_stupid_histogram<uint8_t>/128        78.2 ns         78.0 ns      8900894
BM_stupid_histogram<uint8_t>/256         154 ns          154 ns      4551170
BM_stupid_histogram<uint8_t>/512         316 ns          315 ns      2211707
BM_stupid_histogram<uint8_t>/1024        616 ns          615 ns      1139191
BM_stupid_histogram<uint8_t>/2048       1215 ns         1211 ns       568199
BM_stupid_histogram<uint8_t>/4096       2383 ns         2376 ns       293445
./bench/algorithm/bench_algorithm_histogram  29,68s user 0,00s system 99% cpu 29,803 total
```

Prefixsum:
============

```
2024-09-08T22:32:14+02:00
Running ./bench/algorithm/bench_algorithm_prefixsum
Run on (16 X 1700 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 1280 KiB (x8)
  L3 Unified 12288 KiB (x1)
Load Average: 2.15, 2.99, 2.41
------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations
------------------------------------------------------------------------------
BM_stupid_prefixsum<uint32_t>/32          12.6 ns         12.6 ns     55505754
BM_stupid_prefixsum<uint32_t>/64          26.6 ns         26.5 ns     26348936
BM_stupid_prefixsum<uint32_t>/128         62.7 ns         62.5 ns     11207178
BM_stupid_prefixsum<uint32_t>/256          156 ns          154 ns      4811004
BM_stupid_prefixsum<uint32_t>/512          301 ns          300 ns      2226511
BM_stupid_prefixsum<uint32_t>/1024         603 ns          601 ns      1163377
BM_stupid_prefixsum<uint32_t>/2048        1211 ns         1207 ns       580285
BM_stupid_prefixsum<uint32_t>/4096        2441 ns         2424 ns       289824
BM_stupid_prefixsum<uint32_t>/8192        4890 ns         4870 ns       144767
BM_stupid_prefixsum<uint32_t>/16384       9745 ns         9715 ns        72154
BM_stupid_prefixsum<uint32_t>/32768      19490 ns        19432 ns        35951
BM_stupid_prefixsum<uint32_t>/65536      39082 ns        38953 ns        18080
BM_stupid_prefixsum<uint64_t>/128         63.6 ns         63.3 ns     11006165
BM_stupid_prefixsum<uint64_t>/256          140 ns          139 ns      4861711
BM_stupid_prefixsum<uint64_t>/512          301 ns          299 ns      2358776
BM_stupid_prefixsum<uint64_t>/1024         608 ns          604 ns      1159291
BM_stupid_prefixsum<uint64_t>/2048        1218 ns         1215 ns       576253
BM_stupid_prefixsum<uint64_t>/4096        2430 ns         2422 ns       289299
BM_stupid_prefixsum<uint64_t>/8192        4884 ns         4870 ns       144050
BM_stupid_prefixsum<uint64_t>/16384       9736 ns         9707 ns        72189
BM_stupid_prefixsum<uint64_t>/32768      19466 ns        19408 ns        36118
BM_stupid_prefixsum<uint64_t>/65536      40957 ns        40853 ns        17146
BM_prefixsum<uint32_t>/32                 21.3 ns         21.2 ns     32971336
BM_prefixsum<uint32_t>/64                 24.3 ns         24.2 ns     28875175
BM_prefixsum<uint32_t>/128                38.0 ns         37.9 ns     18476850
BM_prefixsum<uint32_t>/256                75.7 ns         75.5 ns      9290278
BM_prefixsum<uint32_t>/512                 152 ns          152 ns      4608305
BM_prefixsum<uint32_t>/1024                306 ns          305 ns      2296740
BM_prefixsum<uint32_t>/2048                615 ns          613 ns      1140549
BM_prefixsum<uint32_t>/4096               1212 ns         1209 ns       579130
BM_prefixsum<uint32_t>/8192               2440 ns         2433 ns       287896
BM_prefixsum<uint32_t>/16384              4855 ns         4844 ns       144199
BM_prefixsum<uint32_t>/32768              9766 ns         9738 ns        71868
BM_prefixsum<uint32_t>/65536             19385 ns        19328 ns        36216
BM_prefixsum<uint64_t>/128                62.9 ns         62.7 ns     11228705
BM_prefixsum<uint64_t>/256                 141 ns          141 ns      5000682
BM_prefixsum<uint64_t>/512                 293 ns          292 ns      2312814
BM_prefixsum<uint64_t>/1024                604 ns          602 ns      1158377
BM_prefixsum<uint64_t>/2048               1211 ns         1207 ns       579484
BM_prefixsum<uint64_t>/4096               2424 ns         2417 ns       289525
BM_prefixsum<uint64_t>/8192               4871 ns         4857 ns       144107
BM_prefixsum<uint64_t>/16384              9731 ns         9702 ns        72090
BM_prefixsum<uint64_t>/32768             19475 ns        19382 ns        36111
BM_prefixsum<uint64_t>/65536             38825 ns        38707 ns        18078
./bench/algorithm/bench_algorithm_prefixsum  41,10s user 0,01s system 99% cpu 41,249 total
```
