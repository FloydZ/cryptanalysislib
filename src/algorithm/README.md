Histogram:
=========

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
```


Copy:
==== 

```
2024-10-07T15:16:15+02:00
Running ./bench/algorithm/bench_algorithm_copy
Run on (4 X 3600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x4)
  L1 Instruction 64 KiB (x4)
  L2 Unified 2048 KiB (x1)
  L3 Unified 6144 KiB (x1)
Load Average: 1.08, 3.20, 3.29
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
BM_stdcopy<uint8_t>/32                       3.14 ns         3.14 ns    222818164
BM_stdcopy<uint8_t>/64                       3.17 ns         3.17 ns    222352930
BM_stdcopy<uint8_t>/128                      4.18 ns         4.18 ns    170748956
BM_stdcopy<uint8_t>/256                      4.51 ns         4.50 ns    147728352
BM_stdcopy<uint8_t>/512                      9.95 ns         9.94 ns     68513083
BM_stdcopy<uint8_t>/1024                     13.7 ns         13.7 ns     50905532
BM_stdcopy<uint8_t>/2048                     23.7 ns         23.7 ns     29810896
BM_stdcopy<uint8_t>/4096                     99.2 ns         99.1 ns      6832491
BM_stdcopy<uint8_t>/8192                      177 ns          177 ns      3471392
BM_stdcopy<uint8_t>/16384                     359 ns          359 ns      2011546
BM_stdcopy<uint8_t>/32768                    1105 ns         1104 ns       630457
BM_stdcopy<uint8_t>/65536                    2256 ns         2254 ns       316916
BM_stdcopy<uint8_t>/131072                   8384 ns         8375 ns        81619
BM_stdcopy<uint8_t>/262144                   9361 ns         9350 ns        76204
BM_stdcopy<uint8_t>/524288                  18530 ns        18509 ns        38457
BM_stdcopy<uint8_t>/1048576                 46518 ns        46411 ns        15332
BM_copy<uint8_t>/32                          2.50 ns         2.50 ns    276045495
BM_copy<uint8_t>/64                          4.39 ns         4.39 ns    160637279
BM_copy<uint8_t>/128                         7.36 ns         7.35 ns     84538405
BM_copy<uint8_t>/256                         12.4 ns         12.4 ns     56632149
BM_copy<uint8_t>/512                         8.88 ns         8.87 ns     76167937
BM_copy<uint8_t>/1024                        17.9 ns         17.8 ns     39803303
BM_copy<uint8_t>/2048                        38.5 ns         38.4 ns     18022780
BM_copy<uint8_t>/4096                        86.3 ns         86.2 ns      8159703
BM_copy<uint8_t>/8192                         163 ns          163 ns      4437057
BM_copy<uint8_t>/16384                        331 ns          330 ns      2145389
BM_copy<uint8_t>/32768                       1086 ns         1085 ns       644641
BM_copy<uint8_t>/65536                       2215 ns         2213 ns       312483
BM_copy<uint8_t>/131072                      4582 ns         4577 ns       157125
BM_copy<uint8_t>/262144                      9704 ns         9693 ns        72132
BM_copy<uint8_t>/524288                     18108 ns        18087 ns        38751
BM_copy<uint8_t>/1048576                    43328 ns        43230 ns        15867
BM_copy_multithreaded<uint8_t>/32            3.25 ns         3.25 ns    223546315
BM_copy_multithreaded<uint8_t>/64            6.68 ns         6.68 ns    111935951
BM_copy_multithreaded<uint8_t>/128           5.44 ns         5.44 ns    129778211
BM_copy_multithreaded<uint8_t>/256           7.64 ns         7.63 ns     97601820
BM_copy_multithreaded<uint8_t>/512           10.0 ns         9.99 ns     76463348
BM_copy_multithreaded<uint8_t>/1024          8021 ns         4098 ns       172463
BM_copy_multithreaded<uint8_t>/2048          7034 ns         4781 ns       149994
BM_copy_multithreaded<uint8_t>/4096          9597 ns         7131 ns        98192
BM_copy_multithreaded<uint8_t>/8192          9650 ns         7198 ns        96532
BM_copy_multithreaded<uint8_t>/16384        10139 ns         7285 ns        95959
BM_copy_multithreaded<uint8_t>/32768        10660 ns         7585 ns        92278
BM_copy_multithreaded<uint8_t>/65536        11295 ns         7806 ns        89285
BM_copy_multithreaded<uint8_t>/131072       12749 ns         8116 ns        86052
BM_copy_multithreaded<uint8_t>/262144       19362 ns        10184 ns        68839
BM_copy_multithreaded<uint8_t>/524288       31069 ns        10671 ns        66048
BM_copy_multithreaded<uint8_t>/1048576      77557 ns        13185 ns        54131
./bench/algorithm/bench_algorithm_copy  68,87s user 16,50s system 155% cpu 54,723 total
```
