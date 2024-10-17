General:
========
The `algorithm` subdirectory tries to mimic the `std::algorithm` and `std::numeric` interfaces, thus the following 
algorithms are implemented (* means that this algorithm is not in the stl or a slightly different API. [simd] 
indicates that an optimized version for `T=(unsigned} integer` exists. [thread] indicates that a multithreaded
implementation exists. [config] indicates that the algorithm/class is configurable):
    - accumulate() [simd, thread, config]
    - all_of() [thread, config]
    - any_of() [thread, config]
    - none_of() [thread, config]
    - apply() [thread, config]
    - argmin()/argmax() [simd, thread, config]
    - copy() [simd, thread, config]
    - count() [simd, thread, config]
    - eea()
    - equal() [simd, thread, config]
    - exclusive_scan() [thread, config]
    - fill() [thread, config]
    - find() [simd, thread, config]
    - for_each() [thread, config]
    - gcd() []
    - histogram() [simd, config]
    - inclusive_scan() [thread, config, simd]
    - int2scan() []
    - pcs() []
    - prefixsum() [simd, thread, config]
    - random_index() []
    - reduce() [thread, config]
    - regex() []
    - rotate() []
    - subsetsum() []
    - transform() [thread, config]

NOTE: The api changes are mostly done, because we needed more flexibility.

Multithreading:
=============
`cryptanalysislib` supports multithreading for most of its algorithms and data structures. The following models are 
supported:
    - seq: sequential implementation
    - par: parallel implementation using a scheduler. Dynamically using threads.
    - pure: pure `std::thread` implementation. Just start `${nproc}` threads and lets go.

and the following are in development:
    - cuda: deploy on graphic cards (NVIDIA only)
    - dis: distributed



accumulate:
===========
Usage:
```c++
std::vector<T> in; in.resize(s);
std::fill(in.begin(), in.end(), 1);
const auto d = cryptanalysislib::accumulate(in.begin(), in.end(), 0);
```

all_of:
======
Usage: 
```c++
const auto d = cryptanalysislib::all_of(in.begin(), in.end(), [](const T &a) {
	return a == 1;
});
```

any_of:
======

Usage:
```c++
const auto d1 = cryptanalysislib::any_of(in.begin(), in.end(), [](const T &a) {
    return a == 1;
});
```

none_of:
======

Usage:
```c++
const auto d1 = cryptanalysislib::none_of(in.begin(), in.end(), [](const T &a) {
	return a == 1;
});
```		

apply:
======

Usage:
```c++
```

argmin:
======
Usage:
```c++
const auto d = cryptanalysislib::argmin(in.begin(), in.end());
```

argmax:
======
Usage:
```c++
const auto d = cryptanalysislib::argmax(in.begin(), in.end());
```

Benchmarks:

```
2024-10-17T09:59:28+02:00
Run on (12 X 5453 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 1024 KiB (x6)
  L3 Unified 32768 KiB (x1)
Load Average: 0.27, 0.61, 0.61
--------------------------------------------------------------------------------------------------
Benchmark                                        Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------------
BM_argmax_u32_simd/32                         14.2 ns         14.2 ns     53729408 cycles=35.3458
BM_argmax_u32_simd/64                         13.3 ns         13.2 ns     51512174 cycles=31.1725
BM_argmax_u32_simd/128                        21.8 ns         21.8 ns     38117510 cycles=71.0169
BM_argmax_u32_simd/256                        29.0 ns         29.0 ns     23976717 cycles=104.904
BM_argmax_u32_simd/512                        42.7 ns         42.6 ns     13138884 cycles=168.554
BM_argmax_u32_simd/1024                       59.6 ns         59.5 ns     10552837 cycles=248.382
BM_argmax_u32_simd/2048                        128 ns          127 ns      6418009 cycles=567.335
BM_argmax_u32_simd/4096                        192 ns          192 ns      3382357 cycles=869.885
BM_argmax_u32_simd/8192                        387 ns          387 ns      1795148 cycles=1.78626k
BM_argmax_u32_simd/16384                       798 ns          796 ns       880193 cycles=3.71102k
BM_argmax_u32_simd/32768                      1384 ns         1381 ns       490143 cycles=6.46057k
BM_argmax_u32_simd/65536                      2987 ns         2981 ns       233410 cycles=13.9793k
BM_argmax_u32_simd_bl16/32                    16.1 ns         16.1 ns     53791376 cycles=44.4612
BM_argmax_u32_simd_bl16/64                    17.1 ns         17.1 ns     31277380 cycles=48.7969
BM_argmax_u32_simd_bl16/128                   18.0 ns         17.9 ns     39492716 cycles=52.9504
BM_argmax_u32_simd_bl16/256                   25.6 ns         25.6 ns     33738718 cycles=88.6483
BM_argmax_u32_simd_bl16/512                   32.8 ns         32.7 ns     24303800 cycles=121.872
BM_argmax_u32_simd_bl16/1024                  51.2 ns         51.1 ns     11446057 cycles=208.713
BM_argmax_u32_simd_bl16/2048                  85.8 ns         85.6 ns      9980413 cycles=370.536
BM_argmax_u32_simd_bl16/4096                   134 ns          133 ns      5191223 cycles=595.65
BM_argmax_u32_simd_bl16/8192                   278 ns          278 ns      2456394 cycles=1.27384k
BM_argmax_u32_simd_bl16/16384                  597 ns          596 ns      1147035 cycles=2.7696k
BM_argmax_u32_simd_bl16/32768                 1079 ns         1077 ns       665705 cycles=5.03023k
BM_argmax_u32_simd_bl16/65536                 2153 ns         2149 ns       329761 cycles=10.0691k
BM_argmax_u32_simd_bl32/32                    13.3 ns         13.3 ns     52476776 cycles=30.9989
BM_argmax_u32_simd_bl32/64                    14.9 ns         14.9 ns     47189762 cycles=38.7453
BM_argmax_u32_simd_bl32/128                   17.3 ns         17.2 ns     39921917 cycles=49.7975
BM_argmax_u32_simd_bl32/256                   19.3 ns         19.3 ns     36378325 cycles=57.7741
BM_argmax_u32_simd_bl32/512                   22.3 ns         22.3 ns     31341497 cycles=71.8618
BM_argmax_u32_simd_bl32/1024                  31.2 ns         31.1 ns     22416902 cycles=114.695
BM_argmax_u32_simd_bl32/2048                  51.2 ns         51.1 ns     13773184 cycles=208.085
BM_argmax_u32_simd_bl32/4096                  96.0 ns         95.8 ns      7458040 cycles=416.768
BM_argmax_u32_simd_bl32/8192                   151 ns          151 ns      4663336 cycles=675.854
BM_argmax_u32_simd_bl32/16384                  415 ns          414 ns      1691055 cycles=1.91246k
BM_argmax_u32_simd_bl32/32768                  804 ns          803 ns       872021 cycles=3.74137k
BM_argmax_u32_simd_bl32/65536                 1616 ns         1613 ns       434234 cycles=7.54749k
BM_argmax<uint8_t>/32                         13.4 ns         13.4 ns     52109912 cycles=31.5848
BM_argmax<uint8_t>/64                         16.4 ns         16.3 ns     43172489 cycles=43.8616
BM_argmax<uint8_t>/128                        33.2 ns         33.2 ns     21070372 cycles=123.143
BM_argmax<uint8_t>/256                        50.3 ns         50.2 ns     13568020 cycles=202.768
BM_argmax<uint8_t>/512                        87.2 ns         87.0 ns      8162085 cycles=376.619
BM_argmax<uint8_t>/1024                        171 ns          170 ns      4245603 cycles=768.458
BM_argmax<uint8_t>/2048                        304 ns          303 ns      2296832 cycles=1.39229k
BM_argmax<uint8_t>/4096                        598 ns          597 ns      1175300 cycles=2.77147k
BM_argmax<uint8_t>/8192                       1181 ns         1179 ns       590797 cycles=5.50863k
BM_argmax<uint8_t>/16384                      2378 ns         2374 ns       296977 cycles=11.124k
BM_argmax<uint8_t>/32768                      4839 ns         4830 ns       145053 cycles=22.6685k
BM_argmax<uint8_t>/65536                      9710 ns         9691 ns        72163 cycles=45.5137k
BM_argmax<uint32_t>/32                        13.4 ns         13.4 ns     52226359 cycles=31.6964
BM_argmax<uint32_t>/64                        16.3 ns         16.2 ns     42638647 cycles=42.9864
BM_argmax<uint32_t>/128                       27.5 ns         27.5 ns     21334564 cycles=95.5381
BM_argmax<uint32_t>/256                       51.1 ns         51.0 ns     13617293 cycles=205.988
BM_argmax<uint32_t>/512                       94.5 ns         94.4 ns      7799644 cycles=411.382
BM_argmax<uint32_t>/1024                       190 ns          189 ns      3997347 cycles=856.205
BM_argmax<uint32_t>/2048                       351 ns          350 ns      2071718 cycles=1.61512k
BM_argmax<uint32_t>/4096                       652 ns          651 ns      1088739 cycles=3.02397k
BM_argmax<uint32_t>/8192                      1264 ns         1262 ns       553291 cycles=5.89843k
BM_argmax<uint32_t>/16384                     2671 ns         2665 ns       262442 cycles=12.4954k
BM_argmax<uint32_t>/32768                     5276 ns         5264 ns       132576 cycles=24.7136k
BM_argmax<uint32_t>/65536                    10655 ns        10627 ns        66282 cycles=49.9497k
BM_argmax<uint64_t>/32                        13.4 ns         13.4 ns     52135911 cycles=31.7334
BM_argmax<uint64_t>/64                        16.4 ns         16.3 ns     42942625 cycles=43.3775
BM_argmax<uint64_t>/128                       32.8 ns         32.7 ns     20943332 cycles=120.697
BM_argmax<uint64_t>/256                       50.9 ns         50.8 ns     13604902 cycles=205.481
BM_argmax<uint64_t>/512                        104 ns          104 ns      6903522 cycles=453.536
BM_argmax<uint64_t>/1024                       172 ns          172 ns      4105747 cycles=774.742
BM_argmax<uint64_t>/2048                       337 ns          336 ns      2055729 cycles=1.54667k
BM_argmax<uint64_t>/4096                       648 ns          647 ns      1109755 cycles=3.00924k
BM_argmax<uint64_t>/8192                      1380 ns         1378 ns       500363 cycles=6.44127k
BM_argmax<uint64_t>/16384                     2772 ns         2767 ns       252020 cycles=12.9694k
BM_argmax<uint64_t>/32768                     5522 ns         5511 ns       127183 cycles=25.8693k
BM_argmax<uint64_t>/65536                    11212 ns        11159 ns        62514 cycles=52.5603k
BM_argmax_multithreaded<uint8_t>/32           13.8 ns         13.8 ns     51996397 cycles=32.9674
BM_argmax_multithreaded<uint8_t>/64           18.8 ns         18.7 ns     37352784 cycles=55.3688
BM_argmax_multithreaded<uint8_t>/128          27.2 ns         27.2 ns     25819874 cycles=95.3224
BM_argmax_multithreaded<uint8_t>/256          46.0 ns         45.9 ns     15218730 cycles=183.438
BM_argmax_multithreaded<uint8_t>/512          88.1 ns         87.9 ns      7957803 cycles=381.045
BM_argmax_multithreaded<uint8_t>/1024          163 ns          162 ns      4321411 cycles=730.877
BM_argmax_multithreaded<uint8_t>/2048          310 ns          310 ns      2261780 cycles=1.42377k
BM_argmax_multithreaded<uint8_t>/4096          613 ns          611 ns      1157267 cycles=2.84149k
BM_argmax_multithreaded<uint8_t>/8192         1203 ns         1201 ns       580087 cycles=5.61175k
BM_argmax_multithreaded<uint8_t>/16384        8307 ns         2501 ns       283614 cycles=38.9343k
BM_argmax_multithreaded<uint8_t>/32768        8914 ns         3292 ns       221750 cycles=41.7794k
BM_argmax_multithreaded<uint8_t>/65536       10928 ns         5337 ns       118454 cycles=51.2279k
BM_argmax_multithreaded<uint32_t>/32          13.8 ns         13.8 ns     50887089 cycles=32.1166
BM_argmax_multithreaded<uint32_t>/64          18.9 ns         18.9 ns     37247732 cycles=56.1235
BM_argmax_multithreaded<uint32_t>/128         27.5 ns         27.5 ns     24655481 cycles=96.2156
BM_argmax_multithreaded<uint32_t>/256         46.5 ns         46.5 ns     14925840 cycles=185.101
BM_argmax_multithreaded<uint32_t>/512         88.7 ns         88.5 ns      7919684 cycles=382.944
BM_argmax_multithreaded<uint32_t>/1024         167 ns          167 ns      4271888 cycles=750.777
BM_argmax_multithreaded<uint32_t>/2048         317 ns          317 ns      2174245 cycles=1.45534k
BM_argmax_multithreaded<uint32_t>/4096         619 ns          618 ns      1124681 cycles=2.86989k
BM_argmax_multithreaded<uint32_t>/8192        1255 ns         1252 ns       489224 cycles=5.85227k
BM_argmax_multithreaded<uint32_t>/16384       8397 ns         2521 ns       274748 cycles=39.3548k
BM_argmax_multithreaded<uint32_t>/32768       8896 ns         3199 ns       220279 cycles=41.698k
BM_argmax_multithreaded<uint32_t>/65536      10660 ns         5120 ns       130407 cycles=49.969k
BM_argmax_multithreaded<uint64_t>/32          14.0 ns         14.0 ns     50538244 cycles=32.8083
BM_argmax_multithreaded<uint64_t>/64          19.0 ns         19.0 ns     37123240 cycles=56.2606
BM_argmax_multithreaded<uint64_t>/128         27.5 ns         27.4 ns     25438695 cycles=95.8249
BM_argmax_multithreaded<uint64_t>/256         46.7 ns         46.6 ns     15063732 cycles=185.601
BM_argmax_multithreaded<uint64_t>/512         89.5 ns         89.3 ns      7758042 cycles=386.761
BM_argmax_multithreaded<uint64_t>/1024         165 ns          165 ns      4219183 cycles=743.065
BM_argmax_multithreaded<uint64_t>/2048         322 ns          321 ns      2194316 cycles=1.47849k
BM_argmax_multithreaded<uint64_t>/4096         624 ns          622 ns      1123181 cycles=2.89269k
BM_argmax_multithreaded<uint64_t>/8192        1362 ns         1360 ns       515273 cycles=6.3589k
BM_argmax_multithreaded<uint64_t>/16384       8390 ns         2488 ns       290021 cycles=39.3221k
BM_argmax_multithreaded<uint64_t>/32768       8485 ns         3053 ns       225600 cycles=39.7681k
BM_argmax_multithreaded<uint64_t>/65536      10769 ns         5020 ns       138131 cycles=50.4805k
```

copy:
======

Usage:
```c++
const auto end = cryptanalysislib::argmax(in.begin(), in.end(), out.begin());
```

Benchmark:
``` 
2024-10-17T10:12:59+02:00
Run on (12 X 5453 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 1024 KiB (x6)
  L3 Unified 32768 KiB (x1)
Load Average: 1.15, 1.20, 0.93
-------------------------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------
BM_stdcopy<uint8_t>/32                       13.1 ns         13.1 ns     54098269 cycles=30.6933
BM_stdcopy<uint8_t>/64                       12.9 ns         12.9 ns     53697904 cycles=30.2965
BM_stdcopy<uint8_t>/128                      13.0 ns         13.0 ns     53933117 cycles=30.7421
BM_stdcopy<uint8_t>/256                      13.0 ns         12.9 ns     54159488 cycles=30.387
BM_stdcopy<uint8_t>/512                      13.6 ns         13.6 ns     51665533 cycles=33.5146
BM_stdcopy<uint8_t>/1024                     13.7 ns         13.6 ns     51334945 cycles=32.1206
BM_stdcopy<uint8_t>/2048                     21.2 ns         21.2 ns     33158076 cycles=63.6758
BM_stdcopy<uint8_t>/4096                     35.1 ns         35.0 ns     19984897 cycles=126.159
BM_stdcopy<uint8_t>/8192                     57.3 ns         57.2 ns     12240985 cycles=235.935
BM_stdcopy<uint8_t>/16384                     120 ns          120 ns      5866023 cycles=525.865
BM_stdcopy<uint8_t>/32768                     393 ns          393 ns      1783168 cycles=1.81054k
BM_stdcopy<uint8_t>/65536                     777 ns          776 ns       902618 cycles=3.60986k
BM_stdcopy<uint8_t>/131072                   1563 ns         1560 ns       448589 cycles=7.30038k
BM_stdcopy<uint8_t>/262144                   3111 ns         3105 ns       225398 cycles=14.5599k
BM_stdcopy<uint8_t>/524288                   7374 ns         7355 ns        98740 cycles=34.5543k
BM_stdcopy<uint8_t>/1048576                 15263 ns        15222 ns        47148 cycles=71.5619k
BM_copy<uint8_t>/32                          13.0 ns         13.0 ns     54127807 cycles=30.5668
BM_copy<uint8_t>/64                          13.0 ns         13.0 ns     53849302 cycles=30.605
BM_copy<uint8_t>/128                         13.0 ns         12.9 ns     53804311 cycles=30.4409
BM_copy<uint8_t>/256                         13.1 ns         13.0 ns     54038503 cycles=30.8344
BM_copy<uint8_t>/512                         13.7 ns         13.7 ns     51372351 cycles=33.59
BM_copy<uint8_t>/1024                        13.8 ns         13.8 ns     52090971 cycles=32.4373
BM_copy<uint8_t>/2048                        20.6 ns         20.5 ns     34086027 cycles=61.4548
BM_copy<uint8_t>/4096                        35.1 ns         35.0 ns     19976041 cycles=126.157
BM_copy<uint8_t>/8192                        57.9 ns         57.8 ns     12106018 cycles=233.602
BM_copy<uint8_t>/16384                        120 ns          120 ns      5853096 cycles=524.816
BM_copy<uint8_t>/32768                        397 ns          396 ns      1756820 cycles=1.82601k
BM_copy<uint8_t>/65536                        788 ns          786 ns       890267 cycles=3.65918k
BM_copy<uint8_t>/131072                      1787 ns         1784 ns       392471 cycles=8.34788k
BM_copy<uint8_t>/262144                      3186 ns         3180 ns       220174 cycles=14.9132k
BM_copy<uint8_t>/524288                      8005 ns         7984 ns        87531 cycles=37.5159k
BM_copy<uint8_t>/1048576                    14817 ns        14777 ns        47370 cycles=69.4721k
BM_copy_multithreaded<uint8_t>/32            13.0 ns         13.0 ns     53850638 cycles=30.5352
BM_copy_multithreaded<uint8_t>/64            13.2 ns         13.2 ns     53849799 cycles=31.0011
BM_copy_multithreaded<uint8_t>/128           13.0 ns         13.0 ns     53578322 cycles=30.5502
BM_copy_multithreaded<uint8_t>/256           13.1 ns         13.1 ns     53331415 cycles=31.1125
BM_copy_multithreaded<uint8_t>/512           13.4 ns         13.4 ns     52535917 cycles=32.3837
BM_copy_multithreaded<uint8_t>/1024          14.3 ns         14.3 ns     49009024 cycles=32.9676
BM_copy_multithreaded<uint8_t>/2048          21.2 ns         21.2 ns     33165961 cycles=64.1029
BM_copy_multithreaded<uint8_t>/4096          35.4 ns         35.3 ns     19718838 cycles=128.281
BM_copy_multithreaded<uint8_t>/8192          58.1 ns         58.0 ns     12103137 cycles=234.551
BM_copy_multithreaded<uint8_t>/16384          122 ns          121 ns      5773397 cycles=532.693
BM_copy_multithreaded<uint8_t>/32768          397 ns          396 ns      1767483 cycles=1.82551k
BM_copy_multithreaded<uint8_t>/65536          788 ns          787 ns       889578 cycles=3.66265k
BM_copy_multithreaded<uint8_t>/131072        1856 ns         1852 ns       377574 cycles=8.67136k
BM_copy_multithreaded<uint8_t>/262144        9796 ns         2453 ns       273609 cycles=45.9217k
BM_copy_multithreaded<uint8_t>/524288       10174 ns         3198 ns       221780 cycles=47.6874k
BM_copy_multithreaded<uint8_t>/1048576      14035 ns         5311 ns       130283 cycles=65.8052k
```

count:
======
Usage:
```c++
std::vector<T> in; in.resize(s);
std::fill(in.begin(), in.end(), 1);
const auto d = cryptanalysislib::accumulate(in.begin(), in.end(), 0);
```

Benchmark::
```
2024-10-09T15:38:29+02:00
Run on (12 X 5453 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 1024 KiB (x6)
  L3 Unified 32768 KiB (x1)
Load Average: 1.48, 0.93, 0.54
-----------------------------------------------------------------------------------
Benchmark                                         Time             CPU   Iterations
-----------------------------------------------------------------------------------
BM_stdcount<uint8_t>/32                        2.06 ns         2.06 ns    339045481
BM_stdcount<uint8_t>/64                        3.01 ns         3.01 ns    233579633
BM_stdcount<uint8_t>/128                       5.08 ns         5.07 ns    138268284
BM_stdcount<uint8_t>/256                       9.30 ns         9.29 ns     75417241
BM_stdcount<uint8_t>/512                       17.3 ns         17.3 ns     40478970
BM_stdcount<uint8_t>/1024                      33.4 ns         33.3 ns     21016623
BM_stdcount<uint8_t>/2048                      65.8 ns         65.6 ns     10670666
BM_stdcount<uint8_t>/4096                       129 ns          129 ns      5426248
BM_stdcount<uint8_t>/8192                       258 ns          257 ns      2718758
BM_stdcount<uint8_t>/16384                      511 ns          510 ns      1373265
BM_stdcount<uint8_t>/32768                     1022 ns         1021 ns       686082
BM_stdcount<uint8_t>/65536                     2071 ns         2068 ns       338482
BM_stdcount<uint8_t>/131072                    4129 ns         4122 ns       170039
BM_stdcount<uint8_t>/262144                    8267 ns         8253 ns        84907
BM_stdcount<uint8_t>/524288                   16484 ns        16453 ns        42480
BM_stdcount<uint8_t>/1048576                  32906 ns        32837 ns        21456
BM_stdcount<uint32_t>/32                       2.10 ns         2.10 ns    333473165
BM_stdcount<uint32_t>/64                       3.05 ns         3.05 ns    229344605
BM_stdcount<uint32_t>/128                      5.20 ns         5.19 ns    135007553
BM_stdcount<uint32_t>/256                      9.60 ns         9.58 ns     73232384
BM_stdcount<uint32_t>/512                      18.9 ns         18.9 ns     37431007
BM_stdcount<uint32_t>/1024                     34.7 ns         34.6 ns     20166034
BM_stdcount<uint32_t>/2048                     66.4 ns         66.3 ns     10527331
BM_stdcount<uint32_t>/4096                      133 ns          133 ns      5261816
BM_stdcount<uint32_t>/8192                      262 ns          262 ns      2675922
BM_stdcount<uint32_t>/16384                     527 ns          526 ns      1330154
BM_stdcount<uint32_t>/32768                    1057 ns         1055 ns       663756
BM_stdcount<uint32_t>/65536                    2111 ns         2107 ns       334386
BM_stdcount<uint32_t>/131072                   4515 ns         4507 ns       153534
BM_stdcount<uint32_t>/262144                   9374 ns         9353 ns        74816
BM_stdcount<uint32_t>/524288                  17567 ns        17521 ns        39991
BM_stdcount<uint32_t>/1048576                 34613 ns        34521 ns        20268
BM_stdcount<uint64_t>/32                       2.47 ns         2.46 ns    284064518
BM_stdcount<uint64_t>/64                       3.81 ns         3.81 ns    184986757
BM_stdcount<uint64_t>/128                      6.43 ns         6.42 ns    108957670
BM_stdcount<uint64_t>/256                      11.7 ns         11.7 ns     60176295
BM_stdcount<uint64_t>/512                      22.5 ns         22.4 ns     31246376
BM_stdcount<uint64_t>/1024                     42.3 ns         42.2 ns     16564499
BM_stdcount<uint64_t>/2048                     83.2 ns         83.1 ns      8401699
BM_stdcount<uint64_t>/4096                      173 ns          172 ns      4058674
BM_stdcount<uint64_t>/8192                      383 ns          383 ns      1829403
BM_stdcount<uint64_t>/16384                     848 ns          846 ns       826770
BM_stdcount<uint64_t>/32768                    1560 ns         1558 ns       449442
BM_stdcount<uint64_t>/65536                    3758 ns         3751 ns       186648
BM_stdcount<uint64_t>/131072                   7442 ns         7425 ns        94422
BM_stdcount<uint64_t>/262144                  15844 ns        15802 ns        44353
BM_stdcount<uint64_t>/524288                  30950 ns        30868 ns        22679
BM_stdcount<uint64_t>/1048576                 61723 ns        61562 ns        11376
BM_count<uint8_t>/32                           2.15 ns         2.14 ns    328391316
BM_count<uint8_t>/64                          0.839 ns        0.838 ns    836019384
BM_count<uint8_t>/128                          1.28 ns         1.28 ns    547816886
BM_count<uint8_t>/256                          2.32 ns         2.32 ns    301895044
BM_count<uint8_t>/512                          4.51 ns         4.50 ns    154909734
BM_count<uint8_t>/1024                         9.05 ns         9.03 ns     77804036
BM_count<uint8_t>/2048                         17.9 ns         17.9 ns     39205633
BM_count<uint8_t>/4096                         35.6 ns         35.5 ns     19697309
BM_count<uint8_t>/8192                         72.3 ns         72.2 ns      9683989
BM_count<uint8_t>/16384                         141 ns          141 ns      4954664
BM_count<uint8_t>/32768                         282 ns          282 ns      2483129
BM_count<uint8_t>/65536                         570 ns          569 ns      1233838
BM_count<uint8_t>/131072                       1198 ns         1196 ns       576067
BM_count<uint8_t>/262144                       2273 ns         2269 ns       308420
BM_count<uint8_t>/524288                       4761 ns         4752 ns       147283
BM_count<uint8_t>/1048576                      9684 ns         9661 ns        72473
BM_count<uint32_t>/32                          1.27 ns         1.27 ns    551916690
BM_count<uint32_t>/64                          2.25 ns         2.25 ns    313604595
BM_count<uint32_t>/128                         4.38 ns         4.37 ns    161714341
BM_count<uint32_t>/256                         8.52 ns         8.51 ns     82595276
BM_count<uint32_t>/512                         16.6 ns         16.6 ns     42189414
BM_count<uint32_t>/1024                        33.0 ns         33.0 ns     21203519
BM_count<uint32_t>/2048                        69.1 ns         69.0 ns     10156826
BM_count<uint32_t>/4096                         133 ns          133 ns      5262239
BM_count<uint32_t>/8192                         272 ns          271 ns      2580873
BM_count<uint32_t>/16384                        538 ns          537 ns      1304308
BM_count<uint32_t>/32768                       1177 ns         1175 ns       595290
BM_count<uint32_t>/65536                       2212 ns         2208 ns       314378
BM_count<uint32_t>/131072                      4704 ns         4695 ns       149135
BM_count<uint32_t>/262144                     10205 ns        10183 ns        68766
BM_count<uint32_t>/524288                     18482 ns        18433 ns        38023
BM_count<uint32_t>/1048576                    38302 ns        38201 ns        18321
BM_count<uint64_t>/32                          2.22 ns         2.22 ns    315199377
BM_count<uint64_t>/64                          4.37 ns         4.36 ns    159975879
BM_count<uint64_t>/128                         8.61 ns         8.59 ns     81859805
BM_count<uint64_t>/256                         16.8 ns         16.7 ns     41834988
BM_count<uint64_t>/512                         33.3 ns         33.2 ns     21128097
BM_count<uint64_t>/1024                        69.5 ns         69.4 ns     10061601
BM_count<uint64_t>/2048                         133 ns          133 ns      5246192
BM_count<uint64_t>/4096                         272 ns          271 ns      2582102
BM_count<uint64_t>/8192                         538 ns          537 ns      1304138
BM_count<uint64_t>/16384                       1179 ns         1177 ns       594321
BM_count<uint64_t>/32768                       2215 ns         2211 ns       316727
BM_count<uint64_t>/65536                       4738 ns         4729 ns       148703
BM_count<uint64_t>/131072                     10245 ns        10221 ns        68414
BM_count<uint64_t>/262144                     18630 ns        18579 ns        37899
BM_count<uint64_t>/524288                     34993 ns        34899 ns        19844
BM_count<uint64_t>/1048576                    78623 ns        78414 ns         9056
BM_count_uXX_simd<uint8_t>/32                  2.15 ns         2.14 ns    328869361
BM_count_uXX_simd<uint8_t>/64                 0.952 ns        0.951 ns    736631661
BM_count_uXX_simd<uint8_t>/128                 1.29 ns         1.29 ns    548033993
BM_count_uXX_simd<uint8_t>/256                 2.07 ns         2.07 ns    338394124
BM_count_uXX_simd<uint8_t>/512                 4.26 ns         4.25 ns    166212467
BM_count_uXX_simd<uint8_t>/1024                8.03 ns         8.02 ns     87493579
BM_count_uXX_simd<uint8_t>/2048                16.1 ns         16.1 ns     43594648
BM_count_uXX_simd<uint8_t>/4096                32.3 ns         32.3 ns     21701806
BM_count_uXX_simd<uint8_t>/8192                64.4 ns         64.3 ns     10887811
BM_count_uXX_simd<uint8_t>/16384                126 ns          126 ns      5563070
BM_count_uXX_simd<uint8_t>/32768                261 ns          261 ns      2723587
BM_count_uXX_simd<uint8_t>/65536                530 ns          529 ns      1314678
BM_count_uXX_simd<uint8_t>/131072              1143 ns         1141 ns       620877
BM_count_uXX_simd<uint8_t>/262144              2175 ns         2171 ns       325318
BM_count_uXX_simd<uint8_t>/524288              4632 ns         4623 ns       152426
BM_count_uXX_simd<uint8_t>/1048576            10121 ns        10097 ns        69459
BM_count_uXX_simd<uint32_t>/32                 1.30 ns         1.30 ns    538582703
BM_count_uXX_simd<uint32_t>/64                 2.07 ns         2.07 ns    335260515
BM_count_uXX_simd<uint32_t>/128                4.14 ns         4.14 ns    170706540
BM_count_uXX_simd<uint32_t>/256                7.76 ns         7.74 ns     91034381
BM_count_uXX_simd<uint32_t>/512                14.9 ns         14.9 ns     46886214
BM_count_uXX_simd<uint32_t>/1024               30.0 ns         30.0 ns     23353796
BM_count_uXX_simd<uint32_t>/2048               57.9 ns         57.8 ns     12096459
BM_count_uXX_simd<uint32_t>/4096                121 ns          121 ns      5802889
BM_count_uXX_simd<uint32_t>/8192                253 ns          253 ns      2770604
BM_count_uXX_simd<uint32_t>/16384               497 ns          496 ns      1413222
BM_count_uXX_simd<uint32_t>/32768              1128 ns         1126 ns       621727
BM_count_uXX_simd<uint32_t>/65536              2089 ns         2085 ns       335705
BM_count_uXX_simd<uint32_t>/131072             4578 ns         4570 ns       152652
BM_count_uXX_simd<uint32_t>/262144             9520 ns         9498 ns        74619
BM_count_uXX_simd<uint32_t>/524288            16612 ns        16568 ns        42352
BM_count_uXX_simd<uint32_t>/1048576           38161 ns        38060 ns        18494
BM_count_uXX_simd<uint64_t>/32                 2.09 ns         2.09 ns    336453502
BM_count_uXX_simd<uint64_t>/64                 4.19 ns         4.18 ns    166135909
BM_count_uXX_simd<uint64_t>/128                7.69 ns         7.67 ns     91751418
BM_count_uXX_simd<uint64_t>/256                15.0 ns         15.0 ns     46522822
BM_count_uXX_simd<uint64_t>/512                30.0 ns         30.0 ns     23024537
BM_count_uXX_simd<uint64_t>/1024               57.8 ns         57.7 ns     12154931
BM_count_uXX_simd<uint64_t>/2048                121 ns          121 ns      5795974
BM_count_uXX_simd<uint64_t>/4096                254 ns          254 ns      2757875
BM_count_uXX_simd<uint64_t>/8192                497 ns          497 ns      1412216
BM_count_uXX_simd<uint64_t>/16384              1128 ns         1126 ns       620941
BM_count_uXX_simd<uint64_t>/32768              2094 ns         2090 ns       334893
BM_count_uXX_simd<uint64_t>/65536              4702 ns         4693 ns       148156
BM_count_uXX_simd<uint64_t>/131072             9355 ns         9335 ns        76924
BM_count_uXX_simd<uint64_t>/262144            18501 ns        18456 ns        38012
BM_count_uXX_simd<uint64_t>/524288            35098 ns        35003 ns        20678
BM_count_uXX_simd<uint64_t>/1048576           74467 ns        74270 ns         9400
BM_count_multithreaded<uint8_t>/32             2.24 ns         2.23 ns    312428439
BM_count_multithreaded<uint8_t>/64             3.25 ns         3.25 ns    215774094
BM_count_multithreaded<uint8_t>/128            5.37 ns         5.36 ns    130981303
BM_count_multithreaded<uint8_t>/256            9.47 ns         9.45 ns     74032622
BM_count_multithreaded<uint8_t>/512            17.8 ns         17.8 ns     39385459
BM_count_multithreaded<uint8_t>/1024           2506 ns         1433 ns       499584
BM_count_multithreaded<uint8_t>/2048           2400 ns         1847 ns       379774
BM_count_multithreaded<uint8_t>/4096           3720 ns         3145 ns       223712
BM_count_multithreaded<uint8_t>/8192           6122 ns         5638 ns       124590
BM_count_multithreaded<uint8_t>/16384          8759 ns         8304 ns        84344
BM_count_multithreaded<uint8_t>/32768          8865 ns         8376 ns        84547
BM_count_multithreaded<uint8_t>/65536          9173 ns         8621 ns        80762
BM_count_multithreaded<uint8_t>/131072         9936 ns         9138 ns        77341
BM_count_multithreaded<uint8_t>/262144        12305 ns        10478 ns        66590
BM_count_multithreaded<uint8_t>/524288        13443 ns        10974 ns        63714
BM_count_multithreaded<uint8_t>/1048576       16548 ns        12190 ns        57469
BM_count_multithreaded<uint32_t>/32            2.35 ns         2.35 ns    295514184
BM_count_multithreaded<uint32_t>/64            3.34 ns         3.34 ns    209438791
BM_count_multithreaded<uint32_t>/128           5.49 ns         5.48 ns    127674405
BM_count_multithreaded<uint32_t>/256           9.81 ns         9.79 ns     71677984
BM_count_multithreaded<uint32_t>/512           19.1 ns         19.0 ns     37326147
BM_count_multithreaded<uint32_t>/1024          2720 ns         1524 ns       470224
BM_count_multithreaded<uint32_t>/2048          2388 ns         1820 ns       387684
BM_count_multithreaded<uint32_t>/4096          3702 ns         3129 ns       224898
BM_count_multithreaded<uint32_t>/8192          6052 ns         5639 ns       125613
BM_count_multithreaded<uint32_t>/16384         8753 ns         8310 ns        84128
BM_count_multithreaded<uint32_t>/32768         8859 ns         8380 ns        84356
BM_count_multithreaded<uint32_t>/65536         9132 ns         8606 ns        81102
BM_count_multithreaded<uint32_t>/131072       10299 ns         9407 ns        74462
BM_count_multithreaded<uint32_t>/262144       13312 ns        11056 ns        64075
BM_count_multithreaded<uint32_t>/524288       13412 ns        11123 ns        60637
BM_count_multithreaded<uint32_t>/1048576      15815 ns        11864 ns        59069
BM_count_multithreaded<uint64_t>/32            2.89 ns         2.89 ns    249011020
BM_count_multithreaded<uint64_t>/64            3.93 ns         3.92 ns    180601076
BM_count_multithreaded<uint64_t>/128           6.38 ns         6.37 ns    109937633
BM_count_multithreaded<uint64_t>/256           11.8 ns         11.7 ns     59858181
BM_count_multithreaded<uint64_t>/512           22.9 ns         22.9 ns     30308500
BM_count_multithreaded<uint64_t>/1024          2685 ns         1505 ns       455753
BM_count_multithreaded<uint64_t>/2048          2442 ns         1856 ns       383931
BM_count_multithreaded<uint64_t>/4096          3749 ns         3179 ns       222791
BM_count_multithreaded<uint64_t>/8192          6129 ns         5704 ns       122982
BM_count_multithreaded<uint64_t>/16384         8880 ns         8420 ns        83134
BM_count_multithreaded<uint64_t>/32768         8963 ns         8493 ns        82482
BM_count_multithreaded<uint64_t>/65536         9663 ns         9012 ns        77484
BM_count_multithreaded<uint64_t>/131072       12422 ns        10692 ns        65630
BM_count_multithreaded<uint64_t>/262144       13750 ns        11398 ns        61653
BM_count_multithreaded<uint64_t>/524288       15456 ns        11977 ns        58267
BM_count_multithreaded<uint64_t>/1048576      22758 ns        14091 ns        49750
```

eea:
======

equal:
======

Usage:
```c++
const auto end = cryptanalysislib::argmax(in.begin(), in.end(), out.begin());
```

Benchmark:
``` 
2024-10-17T10:24:45+02:00
Run on (12 X 5453 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 1024 KiB (x6)
  L3 Unified 32768 KiB (x1)
Load Average: 3.15, 1.94, 1.32
--------------------------------------------------------------------------------------------------
Benchmark                                        Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------------
BM_stdequal<uint8_t>/32                       13.1 ns         13.1 ns     53122650 cycles=30.7324
BM_stdequal<uint8_t>/64                       13.0 ns         13.0 ns     53861974 cycles=30.5698
BM_stdequal<uint8_t>/128                      13.1 ns         13.1 ns     53633583 cycles=30.7849
BM_stdequal<uint8_t>/256                      13.3 ns         13.3 ns     53320500 cycles=31.3012
BM_stdequal<uint8_t>/512                      13.4 ns         13.4 ns     52409365 cycles=31.59
BM_stdequal<uint8_t>/1024                     13.8 ns         13.8 ns     50588788 cycles=32.4355
BM_stdequal<uint8_t>/2048                     22.5 ns         22.4 ns     31126983 cycles=73.1614
BM_stdequal<uint8_t>/4096                     39.6 ns         39.5 ns     17830541 cycles=152.735
BM_stdequal<uint8_t>/8192                     59.5 ns         59.4 ns     11822973 cycles=246.429
BM_stdequal<uint8_t>/16384                     144 ns          144 ns      4856652 cycles=645.316
BM_stdequal<uint8_t>/32768                     399 ns          398 ns      1755989 cycles=1.8406k
BM_stdequal<uint8_t>/65536                     790 ns          789 ns       888931 cycles=3.67404k
BM_stdequal<uint8_t>/131072                   1572 ns         1569 ns       446086 cycles=7.34087k
BM_stdequal<uint8_t>/262144                   3127 ns         3121 ns       224014 cycles=14.6385k
BM_stdequal<uint8_t>/524288                   7616 ns         7598 ns        92437 cycles=35.6969k
BM_stdequal<uint8_t>/1048576                 14840 ns        14800 ns        47425 cycles=69.5804k
BM_equal<uint8_t>/32                          13.0 ns         13.0 ns     53817440 cycles=30.6097
BM_equal<uint8_t>/64                          13.0 ns         13.0 ns     53736194 cycles=30.5427
BM_equal<uint8_t>/128                         13.0 ns         13.0 ns     53872636 cycles=30.5448
BM_equal<uint8_t>/256                         13.1 ns         13.1 ns     53615032 cycles=30.7764
BM_equal<uint8_t>/512                         13.6 ns         13.6 ns     51069581 cycles=31.5473
BM_equal<uint8_t>/1024                        16.2 ns         16.1 ns     43157716 cycles=43.5448
BM_equal<uint8_t>/2048                        32.7 ns         32.6 ns     21471862 cycles=121.268
BM_equal<uint8_t>/4096                        52.2 ns         52.1 ns     13300394 cycles=213.083
BM_equal<uint8_t>/8192                        93.0 ns         92.8 ns      7538439 cycles=403.229
BM_equal<uint8_t>/16384                        167 ns          166 ns      4189990 cycles=749.858
BM_equal<uint8_t>/32768                        407 ns          406 ns      1725548 cycles=1.87701k
BM_equal<uint8_t>/65536                        818 ns          816 ns       855429 cycles=3.80577k
BM_equal<uint8_t>/131072                      1594 ns         1591 ns       440126 cycles=7.44433k
BM_equal<uint8_t>/262144                      3381 ns         3374 ns       207568 cycles=15.8302k
BM_equal<uint8_t>/524288                      7989 ns         7968 ns        87746 cycles=37.4427k
BM_equal<uint8_t>/1048576                    15009 ns        14967 ns        46716 cycles=70.3736k
BM_equal_multithreaded<uint8_t>/32            12.9 ns         12.9 ns     54074724 cycles=30.3655
BM_equal_multithreaded<uint8_t>/64            13.0 ns         13.0 ns     53864689 cycles=30.6701
BM_equal_multithreaded<uint8_t>/128           13.0 ns         12.9 ns     54109847 cycles=30.41
BM_equal_multithreaded<uint8_t>/256           13.0 ns         13.0 ns     54076133 cycles=30.4811
BM_equal_multithreaded<uint8_t>/512           13.6 ns         13.6 ns     51898317 cycles=32.1364
BM_equal_multithreaded<uint8_t>/1024          16.1 ns         16.1 ns     43480090 cycles=43.5068
BM_equal_multithreaded<uint8_t>/2048          30.0 ns         29.9 ns     23443191 cycles=108.892
BM_equal_multithreaded<uint8_t>/4096          49.1 ns         49.0 ns     14278482 cycles=198.603
BM_equal_multithreaded<uint8_t>/8192          91.3 ns         91.1 ns      7760440 cycles=396.212
BM_equal_multithreaded<uint8_t>/16384          165 ns          165 ns      4179604 cycles=743.013
BM_equal_multithreaded<uint8_t>/32768          409 ns          408 ns      1720273 cycles=1.8844k
BM_equal_multithreaded<uint8_t>/65536          828 ns          827 ns       847935 cycles=3.85398k
BM_equal_multithreaded<uint8_t>/131072        1600 ns         1596 ns       439802 cycles=7.47173k
BM_equal_multithreaded<uint8_t>/262144        9537 ns         3062 ns       239417 cycles=44.7064k
BM_equal_multithreaded<uint8_t>/524288        9549 ns         3114 ns       222670 cycles=44.7621k
BM_equal_multithreaded<uint8_t>/1048576      12796 ns         5226 ns       134645 cycles=59.9944k
```

exclusive_scan:
======

fill:
======

find:
======

for_each:
======


Usage:
```c++
std::vector<T> in; in.resize(s);
std::fill(in.begin(), in.end(), 1);
const auto d = cryptanalysislib::accumulate(in.begin(), in.end(), 0);
```

Benchmark
```
Run on (12 X 5453 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 1024 KiB (x6)
  L3 Unified 32768 KiB (x1)
Load Average: 1.62, 0.75, 0.59
----------------------------------------------------------------------------------
Benchmark                                        Time             CPU   Iterations
----------------------------------------------------------------------------------
BM_stdfind<uint8_t>/32                        5.26 ns         5.25 ns    124585063
BM_stdfind<uint8_t>/64                        8.85 ns         8.83 ns     80524757
BM_stdfind<uint8_t>/128                       17.0 ns         16.9 ns     41182948
BM_stdfind<uint8_t>/256                       33.7 ns         33.6 ns     20803717
BM_stdfind<uint8_t>/512                       70.6 ns         70.5 ns      9930741
BM_stdfind<uint8_t>/1024                       135 ns          135 ns      5178919
BM_stdfind<uint8_t>/2048                       268 ns          267 ns      2628376
BM_stdfind<uint8_t>/4096                       519 ns          519 ns      1349630
BM_stdfind<uint8_t>/8192                      1015 ns         1013 ns       691077
BM_stdfind<uint8_t>/16384                     2006 ns         2002 ns       349585
BM_stdfind<uint8_t>/32768                     4188 ns         4180 ns       167379
BM_stdfind<uint8_t>/65536                     8356 ns         8341 ns        83958
BM_stdfind<uint8_t>/131072                   16713 ns        16684 ns        42051
BM_stdfind<uint8_t>/262144                   33373 ns        33316 ns        21059
BM_stdfind<uint8_t>/524288                   66432 ns        66303 ns        10572
BM_stdfind<uint8_t>/1048576                 132977 ns       132686 ns         5275
BM_stdfind<uint32_t>/32                       5.23 ns         5.22 ns    135614540
BM_stdfind<uint32_t>/64                       9.90 ns         9.89 ns     70986963
BM_stdfind<uint32_t>/128                      19.1 ns         19.1 ns     36625687
BM_stdfind<uint32_t>/256                      37.6 ns         37.5 ns     18655528
BM_stdfind<uint32_t>/512                      78.1 ns         77.9 ns      8982791
BM_stdfind<uint32_t>/1024                      150 ns          150 ns      4655212
BM_stdfind<uint32_t>/2048                      295 ns          294 ns      2379043
BM_stdfind<uint32_t>/4096                      584 ns          583 ns      1200781
BM_stdfind<uint32_t>/8192                     1167 ns         1165 ns       602993
BM_stdfind<uint32_t>/16384                    2344 ns         2340 ns       299647
BM_stdfind<uint32_t>/32768                    4667 ns         4659 ns       150170
BM_stdfind<uint32_t>/65536                    9287 ns         9271 ns        75599
BM_stdfind<uint32_t>/131072                  18529 ns        18496 ns        37809
BM_stdfind<uint32_t>/262144                  37471 ns        37392 ns        18737
BM_stdfind<uint32_t>/524288                  74598 ns        74406 ns         9410
BM_stdfind<uint32_t>/1048576                149120 ns       148734 ns         4702
BM_stdfind<uint64_t>/32                       5.24 ns         5.23 ns    134298891
BM_stdfind<uint64_t>/64                       9.90 ns         9.88 ns     71043835
BM_stdfind<uint64_t>/128                      19.2 ns         19.1 ns     36505722
BM_stdfind<uint64_t>/256                      37.7 ns         37.6 ns     18593244
BM_stdfind<uint64_t>/512                      77.5 ns         77.3 ns      9051988
BM_stdfind<uint64_t>/1024                      150 ns          149 ns      4688713
BM_stdfind<uint64_t>/2048                      294 ns          293 ns      2385929
BM_stdfind<uint64_t>/4096                      590 ns          589 ns      1189506
BM_stdfind<uint64_t>/8192                     1183 ns         1181 ns       595013
BM_stdfind<uint64_t>/16384                    2350 ns         2346 ns       298424
BM_stdfind<uint64_t>/32768                    4697 ns         4689 ns       149307
BM_stdfind<uint64_t>/65536                    9436 ns         9419 ns        74311
BM_stdfind<uint64_t>/131072                  19291 ns        19249 ns        36369
BM_stdfind<uint64_t>/262144                  37825 ns        37725 ns        18547
BM_stdfind<uint64_t>/524288                  75351 ns        75154 ns         9301
BM_stdfind<uint64_t>/1048576                150971 ns       150578 ns         4651
BM_find<uint8_t>/32                           6.77 ns         6.76 ns    103810596
BM_find<uint8_t>/64                          0.578 ns        0.577 ns   1000000000
BM_find<uint8_t>/128                          1.29 ns         1.28 ns    551196453
BM_find<uint8_t>/256                          2.41 ns         2.41 ns    294279173
BM_find<uint8_t>/512                          3.47 ns         3.47 ns    202720266
BM_find<uint8_t>/1024                         9.69 ns         9.67 ns     72409322
BM_find<uint8_t>/2048                         18.1 ns         18.1 ns     38678161
BM_find<uint8_t>/4096                         25.2 ns         25.2 ns     27590219
BM_find<uint8_t>/8192                         71.2 ns         71.1 ns      9766621
BM_find<uint8_t>/16384                         144 ns          143 ns      4888365
BM_find<uint8_t>/32768                         198 ns          197 ns      3548242
BM_find<uint8_t>/65536                         585 ns          584 ns      1199776
BM_find<uint8_t>/131072                        970 ns          968 ns       722938
BM_find<uint8_t>/262144                       2311 ns         2307 ns       303184
BM_find<uint8_t>/524288                       3713 ns         3706 ns       188971
BM_find<uint8_t>/1048576                     10301 ns        10279 ns        68156
BM_find<uint32_t>/32                          1.30 ns         1.30 ns    536620974
BM_find<uint32_t>/64                          2.45 ns         2.45 ns    288008093
BM_find<uint32_t>/128                         3.57 ns         3.56 ns    194797220
BM_find<uint32_t>/256                         9.81 ns         9.79 ns     72246324
BM_find<uint32_t>/512                         18.8 ns         18.8 ns     37272200
BM_find<uint32_t>/1024                        25.8 ns         25.8 ns     26968096
BM_find<uint32_t>/2048                        70.5 ns         70.4 ns      9930976
BM_find<uint32_t>/4096                         142 ns          142 ns      4941498
BM_find<uint32_t>/8192                         197 ns          197 ns      3557752
BM_find<uint32_t>/16384                        584 ns          583 ns      1200352
BM_find<uint32_t>/32768                        972 ns          971 ns       721133
BM_find<uint32_t>/65536                       2312 ns         2308 ns       303297
BM_find<uint32_t>/131072                      3708 ns         3701 ns       189215
BM_find<uint32_t>/262144                      9163 ns         9142 ns        76604
BM_find<uint32_t>/524288                     19557 ns        19504 ns        36418
BM_find<uint32_t>/1048576                    30048 ns        29965 ns        23453
BM_find<uint64_t>/32                          2.46 ns         2.45 ns    285721276
BM_find<uint64_t>/64                          3.58 ns         3.57 ns    198588645
BM_find<uint64_t>/128                         9.81 ns         9.79 ns     71168217
BM_find<uint64_t>/256                         18.9 ns         18.8 ns     37199323
BM_find<uint64_t>/512                         26.1 ns         26.1 ns     26924843
BM_find<uint64_t>/1024                        71.8 ns         71.7 ns      9764881
BM_find<uint64_t>/2048                         143 ns          142 ns      4914661
BM_find<uint64_t>/4096                         197 ns          197 ns      3543863
BM_find<uint64_t>/8192                         583 ns          582 ns      1207203
BM_find<uint64_t>/16384                        959 ns          958 ns       731662
BM_find<uint64_t>/32768                       2309 ns         2305 ns       303622
BM_find<uint64_t>/65536                       3356 ns         3350 ns       197793
BM_find<uint64_t>/131072                      9070 ns         9048 ns        82363
BM_find<uint64_t>/262144                     19323 ns        19272 ns        36283
BM_find<uint64_t>/524288                     38351 ns        38250 ns        18291
BM_find<uint64_t>/1048576                    76659 ns        76456 ns         9160
BM_simd_find<uint8_t>/32                      6.79 ns         6.78 ns    103239852
BM_simd_find<uint8_t>/64                     0.588 ns        0.587 ns   1000000000
BM_simd_find<uint8_t>/128                     1.32 ns         1.32 ns    524552410
BM_simd_find<uint8_t>/256                     2.39 ns         2.39 ns    293362705
BM_simd_find<uint8_t>/512                     3.47 ns         3.47 ns    201411792
BM_simd_find<uint8_t>/1024                    9.10 ns         9.08 ns     77635281
BM_simd_find<uint8_t>/2048                    17.6 ns         17.5 ns     39933197
BM_simd_find<uint8_t>/4096                    25.9 ns         25.8 ns     26915068
BM_simd_find<uint8_t>/8192                    71.6 ns         71.4 ns      9792013
BM_simd_find<uint8_t>/16384                    143 ns          143 ns      4902989
BM_simd_find<uint8_t>/32768                    199 ns          198 ns      3533440
BM_simd_find<uint8_t>/65536                    585 ns          584 ns      1198219
BM_simd_find<uint8_t>/131072                   973 ns          972 ns       719338
BM_simd_find<uint8_t>/262144                  2314 ns         2310 ns       302098
BM_simd_find<uint8_t>/524288                  3722 ns         3715 ns       188351
BM_simd_find<uint8_t>/1048576                 9119 ns         9099 ns        77107
BM_simd_find<uint32_t>/32                     1.30 ns         1.30 ns    538131978
BM_simd_find<uint32_t>/64                     2.43 ns         2.42 ns    288232838
BM_simd_find<uint32_t>/128                    3.48 ns         3.47 ns    198638896
BM_simd_find<uint32_t>/256                    9.15 ns         9.13 ns     76545049
BM_simd_find<uint32_t>/512                    17.8 ns         17.8 ns     39433149
BM_simd_find<uint32_t>/1024                   26.2 ns         26.1 ns     26857103
BM_simd_find<uint32_t>/2048                   71.8 ns         71.6 ns      9752488
BM_simd_find<uint32_t>/4096                    143 ns          142 ns      4914063
BM_simd_find<uint32_t>/8192                    200 ns          199 ns      3512914
BM_simd_find<uint32_t>/16384                   586 ns          585 ns      1196181
BM_simd_find<uint32_t>/32768                   977 ns          976 ns       718258
BM_simd_find<uint32_t>/65536                  2315 ns         2311 ns       302906
BM_simd_find<uint32_t>/131072                 3725 ns         3718 ns       188369
BM_simd_find<uint32_t>/262144                10247 ns        10224 ns        68580
BM_simd_find<uint32_t>/524288                19788 ns        19738 ns        36152
BM_simd_find<uint32_t>/1048576               33024 ns        32937 ns        21244
BM_simd_find<uint64_t>/32                     2.40 ns         2.40 ns    289918001
BM_simd_find<uint64_t>/64                     3.59 ns         3.59 ns    195646725
BM_simd_find<uint64_t>/128                    9.13 ns         9.11 ns     76783597
BM_simd_find<uint64_t>/256                    17.6 ns         17.6 ns     39735978
BM_simd_find<uint64_t>/512                    26.1 ns         26.0 ns     26869090
BM_simd_find<uint64_t>/1024                   71.8 ns         71.7 ns      9731160
BM_simd_find<uint64_t>/2048                    143 ns          142 ns      4915344
BM_simd_find<uint64_t>/4096                    200 ns          200 ns      3505236
BM_simd_find<uint64_t>/8192                    586 ns          585 ns      1196124
BM_simd_find<uint64_t>/16384                   975 ns          973 ns       718895
BM_simd_find<uint64_t>/32768                  2314 ns         2310 ns       302994
BM_simd_find<uint64_t>/65536                  3721 ns         3714 ns       188660
BM_simd_find<uint64_t>/131072                10270 ns        10248 ns        68332
BM_simd_find<uint64_t>/262144                19308 ns        19257 ns        36378
BM_simd_find<uint64_t>/524288                37977 ns        37876 ns        18478
BM_simd_find<uint64_t>/1048576               76961 ns        76754 ns         9128
BM_find_multithreaded<uint8_t>/32             8.00 ns         7.98 ns     87351112
BM_find_multithreaded<uint8_t>/64             2.35 ns         2.34 ns    298774641
BM_find_multithreaded<uint8_t>/128            2.74 ns         2.73 ns    259174482
BM_find_multithreaded<uint8_t>/256            3.47 ns         3.46 ns    226321210
BM_find_multithreaded<uint8_t>/512            4.66 ns         4.66 ns    158249047
BM_find_multithreaded<uint8_t>/1024         107181 ns       103816 ns         6819
BM_find_multithreaded<uint8_t>/2048         246961 ns       240166 ns         2905
BM_find_multithreaded<uint8_t>/4096         532978 ns       520236 ns         1351
BM_find_multithreaded<uint8_t>/8192        1100370 ns      1073483 ns          658
BM_find_multithreaded<uint8_t>/16384       1650821 ns      1613869 ns          434
BM_find_multithreaded<uint8_t>/32768       1654750 ns      1622630 ns          437
BM_find_multithreaded<uint8_t>/65536       1674116 ns      1637624 ns          428
BM_find_multithreaded<uint8_t>/131072      1661645 ns      1624360 ns          432
BM_find_multithreaded<uint8_t>/262144      1657424 ns      1620641 ns          432
BM_find_multithreaded<uint8_t>/524288      1656010 ns      1623236 ns          436
BM_find_multithreaded<uint8_t>/1048576     1674440 ns      1638875 ns          432
BM_find_multithreaded<uint32_t>/32            2.89 ns         2.88 ns    240633274
BM_find_multithreaded<uint32_t>/64            3.27 ns         3.27 ns    214349975
BM_find_multithreaded<uint32_t>/128           5.54 ns         5.53 ns    127675912
BM_find_multithreaded<uint32_t>/256           8.92 ns         8.90 ns     78682777
BM_find_multithreaded<uint32_t>/512           13.8 ns         13.8 ns     50840110
BM_find_multithreaded<uint32_t>/1024        104159 ns       100964 ns         7035
BM_find_multithreaded<uint32_t>/2048        247194 ns       240555 ns         2879
BM_find_multithreaded<uint32_t>/4096        535613 ns       522416 ns         1346
BM_find_multithreaded<uint32_t>/8192       1101454 ns      1078260 ns          652
BM_find_multithreaded<uint32_t>/16384      1685382 ns      1648720 ns          424
BM_find_multithreaded<uint32_t>/32768      1689057 ns      1655203 ns          429
BM_find_multithreaded<uint32_t>/65536      1692326 ns      1657227 ns          424
BM_find_multithreaded<uint32_t>/131072     1685478 ns      1651183 ns          421
BM_find_multithreaded<uint32_t>/262144     1690901 ns      1650984 ns          424
BM_find_multithreaded<uint32_t>/524288     1689459 ns      1653599 ns          423
BM_find_multithreaded<uint32_t>/1048576    1705534 ns      1669783 ns          422
BM_find_multithreaded<uint64_t>/32            3.66 ns         3.65 ns    199729719
BM_find_multithreaded<uint64_t>/64            5.47 ns         5.46 ns    128351609
BM_find_multithreaded<uint64_t>/128           8.95 ns         8.94 ns     78502153
BM_find_multithreaded<uint64_t>/256           14.4 ns         14.4 ns     48633541
BM_find_multithreaded<uint64_t>/512           35.3 ns         35.3 ns     19837389
BM_find_multithreaded<uint64_t>/1024        104563 ns       101209 ns         7010
BM_find_multithreaded<uint64_t>/2048        246386 ns       239502 ns         2879
BM_find_multithreaded<uint64_t>/4096        527229 ns       514953 ns         1370
BM_find_multithreaded<uint64_t>/8192       1103448 ns      1080461 ns          647
BM_find_multithreaded<uint64_t>/16384      1685813 ns      1652662 ns          421
BM_find_multithreaded<uint64_t>/32768      1688921 ns      1648180 ns          432
BM_find_multithreaded<uint64_t>/65536      1680866 ns      1646284 ns          426
BM_find_multithreaded<uint64_t>/131072     1702313 ns      1667432 ns          420
BM_find_multithreaded<uint64_t>/262144     1690544 ns      1657408 ns          425
BM_find_multithreaded<uint64_t>/524288     1692018 ns      1651342 ns          429
BM_find_multithreaded<uint64_t>/1048576    1701645 ns      1663799 ns          414
```
gcd:
======

histogram:
=========

Usage:
```c++
std::vector<T> in; in.resize(s);
std::fill(in.begin(), in.end(), 1);
const auto d = cryptanalysislib::accumulate(in.begin(), in.end(), 0);
```
```
2024-09-08T22:34:54+02:00
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

inclusive_scan:
======
int2weight:
======
pcs:
======
prefixsum:
======

```
2024-09-08T22:32:14+02:00
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


random_index:
======

reduce:
======

regex:
======

rotate:
======

subsetsum:
======

transform:
======