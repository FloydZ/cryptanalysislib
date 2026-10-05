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

Benchmark:
```
2024-10-17T10:30:22+02:00
Run on (12 X 5453 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 1024 KiB (x6)
  L3 Unified 32768 KiB (x1)
--------------------------------------------------------------------------------------------------------
Benchmark                                              Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------------------
BM_stdaccumulate<uint8_t>/32                        13.7 ns         13.6 ns     50743921 cycles=32.0641
BM_stdaccumulate<uint8_t>/64                        13.7 ns         13.7 ns     51447911 cycles=32.1715
BM_stdaccumulate<uint8_t>/128                       13.8 ns         13.7 ns     50954939 cycles=32.4102
BM_stdaccumulate<uint8_t>/256                       13.9 ns         13.8 ns     50667791 cycles=32.6881
BM_stdaccumulate<uint8_t>/512                       14.3 ns         14.3 ns     50025498 cycles=34.8046
BM_stdaccumulate<uint8_t>/1024                      16.7 ns         16.7 ns     41143681 cycles=46.1598
BM_stdaccumulate<uint8_t>/2048                      32.2 ns         32.2 ns     21878847 cycles=119.376
BM_stdaccumulate<uint8_t>/4096                      62.4 ns         62.3 ns     11216813 cycles=261.456
BM_stdaccumulate<uint8_t>/8192                       118 ns          118 ns      5928989 cycles=523.201
BM_stdaccumulate<uint8_t>/16384                      234 ns          233 ns      3002216 cycles=1.06462k
BM_stdaccumulate<uint8_t>/32768                      467 ns          466 ns      1501414 cycles=2.15951k
BM_stdaccumulate<uint8_t>/65536                      931 ns          929 ns       752717 cycles=4.33679k
BM_stdaccumulate<uint8_t>/131072                    1864 ns         1861 ns       376188 cycles=8.71297k
BM_stdaccumulate<uint8_t>/262144                    3741 ns         3734 ns       187463 cycles=17.5192k
BM_stdaccumulate<uint8_t>/524288                    7476 ns         7461 ns        94114 cycles=35.0371k
BM_stdaccumulate<uint8_t>/1048576                  14928 ns        14893 ns        46972 cycles=69.9941k
BM_stdaccumulate<uint32_t>/32                       13.0 ns         12.9 ns     53908804 cycles=30.3984
BM_stdaccumulate<uint32_t>/64                       13.1 ns         13.0 ns     53743757 cycles=30.8391
BM_stdaccumulate<uint32_t>/128                      13.0 ns         13.0 ns     53871054 cycles=30.618
BM_stdaccumulate<uint32_t>/256                      13.2 ns         13.2 ns     53103122 cycles=31.5849
BM_stdaccumulate<uint32_t>/512                      14.8 ns         14.8 ns     47307477 cycles=33.7531
BM_stdaccumulate<uint32_t>/1024                     19.2 ns         19.2 ns     36281216 cycles=56.4649
BM_stdaccumulate<uint32_t>/2048                     43.7 ns         43.6 ns     16059975 cycles=169.458
BM_stdaccumulate<uint32_t>/4096                     84.9 ns         84.7 ns      8252282 cycles=362.409
BM_stdaccumulate<uint32_t>/8192                      116 ns          116 ns      6056075 cycles=510.065
BM_stdaccumulate<uint32_t>/16384                     408 ns          408 ns      1717421 cycles=1.87927k
BM_stdaccumulate<uint32_t>/32768                     764 ns          763 ns       919476 cycles=3.55232k
BM_stdaccumulate<uint32_t>/65536                    1589 ns         1587 ns       441164 cycles=7.41978k
BM_stdaccumulate<uint32_t>/131072                   3093 ns         3087 ns       226753 cycles=14.4748k
BM_stdaccumulate<uint32_t>/262144                   7803 ns         7785 ns        89260 cycles=36.5709k
BM_stdaccumulate<uint32_t>/524288                  15131 ns        15091 ns        46375 cycles=70.9455k
BM_stdaccumulate<uint32_t>/1048576                 29797 ns        29716 ns        23511 cycles=139.744k
BM_stdaccumulate<uint64_t>/32                       13.0 ns         13.0 ns     53870521 cycles=30.5045
BM_stdaccumulate<uint64_t>/64                       13.1 ns         13.0 ns     53863895 cycles=30.7448
BM_stdaccumulate<uint64_t>/128                      13.2 ns         13.2 ns     52822137 cycles=31.6624
BM_stdaccumulate<uint64_t>/256                      14.9 ns         14.9 ns     46816183 cycles=34.5662
BM_stdaccumulate<uint64_t>/512                      21.3 ns         21.3 ns     32458676 cycles=67.7046
BM_stdaccumulate<uint64_t>/1024                     44.9 ns         44.8 ns     12946883 cycles=174.696
BM_stdaccumulate<uint64_t>/2048                     85.9 ns         85.8 ns      8150119 cycles=367.388
BM_stdaccumulate<uint64_t>/4096                      124 ns          124 ns      5675871 cycles=548.962
BM_stdaccumulate<uint64_t>/8192                      426 ns          425 ns      1646942 cycles=1.96306k
BM_stdaccumulate<uint64_t>/16384                     768 ns          767 ns       913253 cycles=3.57084k
BM_stdaccumulate<uint64_t>/32768                    1683 ns         1681 ns       416542 cycles=7.8622k
BM_stdaccumulate<uint64_t>/65536                    3122 ns         3116 ns       224861 cycles=14.6124k
BM_stdaccumulate<uint64_t>/131072                   8172 ns         8153 ns        86225 cycles=38.3005k
BM_stdaccumulate<uint64_t>/262144                  15017 ns        14976 ns        47548 cycles=70.4137k
BM_stdaccumulate<uint64_t>/524288                  31485 ns        31399 ns        22447 cycles=147.661k
BM_stdaccumulate<uint64_t>/1048576                 63014 ns        62842 ns        10973 cycles=295.565k
BM_accumulate<uint8_t>/32                           13.0 ns         13.0 ns     53824146 cycles=30.5364
BM_accumulate<uint8_t>/64                           13.0 ns         12.9 ns     53991153 cycles=30.4151
BM_accumulate<uint8_t>/128                          13.0 ns         13.0 ns     53926806 cycles=30.595
BM_accumulate<uint8_t>/256                          13.0 ns         13.0 ns     53859193 cycles=30.5246
BM_accumulate<uint8_t>/512                          13.1 ns         13.1 ns     53519833 cycles=30.8587
BM_accumulate<uint8_t>/1024                         13.2 ns         13.1 ns     53386826 cycles=31.1697
BM_accumulate<uint8_t>/2048                         15.1 ns         15.1 ns     46531583 cycles=34.4082
BM_accumulate<uint8_t>/4096                         19.3 ns         19.3 ns     36427881 cycles=56.8013
BM_accumulate<uint8_t>/8192                         43.8 ns         43.7 ns     15920622 cycles=169.851
BM_accumulate<uint8_t>/16384                        85.1 ns         84.9 ns      8267432 cycles=363.427
BM_accumulate<uint8_t>/32768                         116 ns          116 ns      6098698 cycles=510.714
BM_accumulate<uint8_t>/65536                         410 ns          409 ns      1716307 cycles=1.88686k
BM_accumulate<uint8_t>/131072                        764 ns          763 ns       900157 cycles=3.55235k
BM_accumulate<uint8_t>/262144                       1604 ns         1600 ns       440346 cycles=7.48885k
BM_accumulate<uint8_t>/524288                       3104 ns         3098 ns       223631 cycles=14.5287k
BM_accumulate<uint8_t>/1048576                      8133 ns         8113 ns        89578 cycles=38.1209k
BM_accumulate<uint32_t>/32                          13.1 ns         13.1 ns     54213820 cycles=30.8547
BM_accumulate<uint32_t>/64                          13.1 ns         13.1 ns     53028664 cycles=31.01
BM_accumulate<uint32_t>/128                         13.0 ns         13.0 ns     53629005 cycles=30.6248
BM_accumulate<uint32_t>/256                         13.3 ns         13.3 ns     53379729 cycles=31.9031
BM_accumulate<uint32_t>/512                         14.9 ns         14.8 ns     47102530 cycles=33.8809
BM_accumulate<uint32_t>/1024                        19.3 ns         19.2 ns     36550045 cycles=56.7349
BM_accumulate<uint32_t>/2048                        43.8 ns         43.7 ns     15985128 cycles=170.009
BM_accumulate<uint32_t>/4096                        84.9 ns         84.7 ns      8261130 cycles=362.498
BM_accumulate<uint32_t>/8192                         116 ns          116 ns      6039162 cycles=511.062
BM_accumulate<uint32_t>/16384                        408 ns          408 ns      1717114 cycles=1.8798k
BM_accumulate<uint32_t>/32768                        765 ns          764 ns       919292 cycles=3.55831k
BM_accumulate<uint32_t>/65536                       1590 ns         1587 ns       440818 cycles=7.42283k
BM_accumulate<uint32_t>/131072                      3095 ns         3089 ns       224588 cycles=14.4873k
BM_accumulate<uint32_t>/262144                      7272 ns         7254 ns        96667 cycles=34.0815k
BM_accumulate<uint32_t>/524288                     16513 ns        16470 ns        43822 cycles=77.4287k
BM_accumulate<uint32_t>/1048576                    30247 ns        30164 ns        22502 cycles=141.855k
BM_accumulate<uint64_t>/32                          13.1 ns         13.1 ns     53493950 cycles=30.8459
BM_accumulate<uint64_t>/64                          13.1 ns         13.1 ns     54155698 cycles=30.7634
BM_accumulate<uint64_t>/128                         13.6 ns         13.6 ns     53033745 cycles=32.4946
BM_accumulate<uint64_t>/256                         14.8 ns         14.8 ns     47497780 cycles=33.7785
BM_accumulate<uint64_t>/512                         19.0 ns         19.0 ns     36768378 cycles=55.8571
BM_accumulate<uint64_t>/1024                        43.3 ns         43.2 ns     16169476 cycles=167.873
BM_accumulate<uint64_t>/2048                        84.5 ns         84.4 ns      8291774 cycles=360.963
BM_accumulate<uint64_t>/4096                         115 ns          115 ns      6118429 cycles=505.368
BM_accumulate<uint64_t>/8192                         406 ns          406 ns      1717928 cycles=1.87104k
BM_accumulate<uint64_t>/16384                        760 ns          758 ns       923638 cycles=3.53122k
BM_accumulate<uint64_t>/32768                       1582 ns         1579 ns       442941 cycles=7.38628k
BM_accumulate<uint64_t>/65536                       3092 ns         3086 ns       227771 cycles=14.4723k
BM_accumulate<uint64_t>/131072                      7235 ns         7217 ns        96993 cycles=33.9088k
BM_accumulate<uint64_t>/262144                     15411 ns        15368 ns        45839 cycles=72.2617k
BM_accumulate<uint64_t>/524288                     29912 ns        29831 ns        23523 cycles=140.282k
BM_accumulate<uint64_t>/1048576                    60953 ns        60792 ns        11803 cycles=285.894k
BM_accumulate_uXX_simd<uint8_t>/32                  13.0 ns         13.0 ns     53800883 cycles=30.5569
BM_accumulate_uXX_simd<uint8_t>/64                  13.0 ns         13.0 ns     53864537 cycles=30.5489
BM_accumulate_uXX_simd<uint8_t>/128                 13.2 ns         13.1 ns     53865274 cycles=30.9038
BM_accumulate_uXX_simd<uint8_t>/256                 13.0 ns         13.0 ns     53871417 cycles=30.5583
BM_accumulate_uXX_simd<uint8_t>/512                 13.2 ns         13.1 ns     53830465 cycles=30.9033
BM_accumulate_uXX_simd<uint8_t>/1024                13.2 ns         13.2 ns     53308845 cycles=31.1268
BM_accumulate_uXX_simd<uint8_t>/2048                15.0 ns         15.0 ns     46714941 cycles=34.1038
BM_accumulate_uXX_simd<uint8_t>/4096                16.7 ns         16.6 ns     42070430 cycles=43.9126
BM_accumulate_uXX_simd<uint8_t>/8192                44.9 ns         44.8 ns     15542998 cycles=175.004
BM_accumulate_uXX_simd<uint8_t>/16384               86.4 ns         86.2 ns      8226560 cycles=369.127
BM_accumulate_uXX_simd<uint8_t>/32768                117 ns          117 ns      5980358 cycles=516.937
BM_accumulate_uXX_simd<uint8_t>/65536                411 ns          410 ns      1706839 cycles=1.89055k
BM_accumulate_uXX_simd<uint8_t>/131072               765 ns          764 ns       916918 cycles=3.5562k
BM_accumulate_uXX_simd<uint8_t>/262144              1597 ns         1595 ns       438755 cycles=7.45748k
BM_accumulate_uXX_simd<uint8_t>/524288              3094 ns         3088 ns       226730 cycles=14.4797k
BM_accumulate_uXX_simd<uint8_t>/1048576             7197 ns         7180 ns        97755 cycles=33.7291k
BM_accumulate_uXX_simd<uint32_t>/32                 13.0 ns         13.0 ns     53870898 cycles=30.5296
BM_accumulate_uXX_simd<uint32_t>/64                 13.0 ns         13.0 ns     53860015 cycles=30.55
BM_accumulate_uXX_simd<uint32_t>/128                13.0 ns         13.0 ns     53774168 cycles=30.6431
BM_accumulate_uXX_simd<uint32_t>/256                13.3 ns         13.3 ns     53674796 cycles=31.3275
BM_accumulate_uXX_simd<uint32_t>/512                14.6 ns         14.6 ns     48009232 cycles=32.6753
BM_accumulate_uXX_simd<uint32_t>/1024               16.5 ns         16.4 ns     42605437 cycles=43.369
BM_accumulate_uXX_simd<uint32_t>/2048               44.6 ns         44.5 ns     15683974 cycles=173.687
BM_accumulate_uXX_simd<uint32_t>/4096               85.7 ns         85.6 ns      8247211 cycles=366.099
BM_accumulate_uXX_simd<uint32_t>/8192                117 ns          117 ns      5917896 cycles=514.466
BM_accumulate_uXX_simd<uint32_t>/16384               411 ns          410 ns      1708965 cycles=1.89121k
BM_accumulate_uXX_simd<uint32_t>/32768               764 ns          763 ns       918311 cycles=3.55245k
BM_accumulate_uXX_simd<uint32_t>/65536              1596 ns         1593 ns       439332 cycles=7.44968k
BM_accumulate_uXX_simd<uint32_t>/131072             3095 ns         3090 ns       224806 cycles=14.4875k
BM_accumulate_uXX_simd<uint32_t>/262144             7737 ns         7719 ns        90787 cycles=36.261k
BM_accumulate_uXX_simd<uint32_t>/524288            15337 ns        15296 ns        45728 cycles=71.915k
BM_accumulate_uXX_simd<uint32_t>/1048576           29060 ns        28982 ns        24190 cycles=136.289k
BM_accumulate_uXX_simd<uint64_t>/32                 13.0 ns         13.0 ns     53851512 cycles=30.5394
BM_accumulate_uXX_simd<uint64_t>/64                 13.1 ns         13.1 ns     53612201 cycles=30.8185
BM_accumulate_uXX_simd<uint64_t>/128                13.3 ns         13.3 ns     53002890 cycles=31.7539
BM_accumulate_uXX_simd<uint64_t>/256                14.4 ns         14.3 ns     48786665 cycles=32.3166
BM_accumulate_uXX_simd<uint64_t>/512                16.4 ns         16.4 ns     42403786 cycles=43.3188
BM_accumulate_uXX_simd<uint64_t>/1024               45.0 ns         44.9 ns     15733161 cycles=175.244
BM_accumulate_uXX_simd<uint64_t>/2048               85.1 ns         85.0 ns      8231598 cycles=363.533
BM_accumulate_uXX_simd<uint64_t>/4096                117 ns          117 ns      5991106 cycles=515.933
BM_accumulate_uXX_simd<uint64_t>/8192                410 ns          409 ns      1709165 cycles=1.88834k
BM_accumulate_uXX_simd<uint64_t>/16384               767 ns          766 ns       918403 cycles=3.5652k
BM_accumulate_uXX_simd<uint64_t>/32768              1593 ns         1590 ns       440713 cycles=7.43875k
BM_accumulate_uXX_simd<uint64_t>/65536              3079 ns         3073 ns       227773 cycles=14.4105k
BM_accumulate_uXX_simd<uint64_t>/131072             7818 ns         7801 ns        89871 cycles=36.6437k
BM_accumulate_uXX_simd<uint64_t>/262144            15510 ns        15469 ns        45504 cycles=72.7245k
BM_accumulate_uXX_simd<uint64_t>/524288            30723 ns        30644 ns        23609 cycles=144.088k
BM_accumulate_uXX_simd<uint64_t>/1048576           59391 ns        59231 ns        11805 cycles=278.57k
BM_accumulate_multithreaded<uint8_t>/32             13.0 ns         13.0 ns     53869035 cycles=30.5453
BM_accumulate_multithreaded<uint8_t>/64             13.0 ns         13.0 ns     53893781 cycles=30.542
BM_accumulate_multithreaded<uint8_t>/128            13.0 ns         13.0 ns     53864951 cycles=30.5772
BM_accumulate_multithreaded<uint8_t>/256            13.2 ns         13.2 ns     53047238 cycles=31.3179
BM_accumulate_multithreaded<uint8_t>/512            13.1 ns         13.1 ns     53006366 cycles=31.1439
BM_accumulate_multithreaded<uint8_t>/1024           5219 ns         2386 ns       295470 cycles=24.4503k
BM_accumulate_multithreaded<uint8_t>/2048           5593 ns         2774 ns       258609 cycles=26.2048k
BM_accumulate_multithreaded<uint8_t>/4096           6105 ns         3866 ns       179994 cycles=28.6058k
BM_accumulate_multithreaded<uint8_t>/8192          14325 ns         9796 ns        71634 cycles=67.1636k
BM_accumulate_multithreaded<uint8_t>/16384         20409 ns        16073 ns        43406 cycles=95.7033k
BM_accumulate_multithreaded<uint8_t>/32768         20570 ns        16121 ns        43638 cycles=96.4592k
BM_accumulate_multithreaded<uint8_t>/65536         20449 ns        16089 ns        43544 cycles=95.8922k
BM_accumulate_multithreaded<uint8_t>/131072        20363 ns        16027 ns        43301 cycles=95.4841k
BM_accumulate_multithreaded<uint8_t>/262144        20603 ns        16223 ns        43221 cycles=96.6135k
BM_accumulate_multithreaded<uint8_t>/524288        20701 ns        16328 ns        42740 cycles=97.0746k
BM_accumulate_multithreaded<uint8_t>/1048576       21070 ns        16603 ns        42257 cycles=98.8003k
BM_accumulate_multithreaded<uint32_t>/32            13.3 ns         13.3 ns     52223141 cycles=31.1609
BM_accumulate_multithreaded<uint32_t>/64            13.4 ns         13.3 ns     50164442 cycles=31.3673
BM_accumulate_multithreaded<uint32_t>/128           13.2 ns         13.2 ns     52817218 cycles=31.0087
BM_accumulate_multithreaded<uint32_t>/256           13.3 ns         13.3 ns     52663126 cycles=31.1884
BM_accumulate_multithreaded<uint32_t>/512           16.6 ns         16.5 ns     42561121 cycles=42.4671
BM_accumulate_multithreaded<uint32_t>/1024          5300 ns         2527 ns       294188 cycles=24.8284k
BM_accumulate_multithreaded<uint32_t>/2048          5540 ns         2719 ns       247758 cycles=25.9576k
BM_accumulate_multithreaded<uint32_t>/4096          6074 ns         3867 ns       182519 cycles=28.4576k
BM_accumulate_multithreaded<uint32_t>/8192         14405 ns         9813 ns        72068 cycles=67.5374k
BM_accumulate_multithreaded<uint32_t>/16384        20441 ns        16128 ns        43537 cycles=95.8549k
BM_accumulate_multithreaded<uint32_t>/32768        20480 ns        16132 ns        42797 cycles=96.037k
BM_accumulate_multithreaded<uint32_t>/65536        20735 ns        16296 ns        43302 cycles=97.2293k
BM_accumulate_multithreaded<uint32_t>/131072       20837 ns        16397 ns        42431 cycles=97.7106k
BM_accumulate_multithreaded<uint32_t>/262144       21131 ns        16685 ns        41922 cycles=99.0853k
BM_accumulate_multithreaded<uint32_t>/524288       23400 ns        17773 ns        40539 cycles=109.732k
BM_accumulate_multithreaded<uint32_t>/1048576      24695 ns        18812 ns        37297 cycles=115.808k
BM_accumulate_multithreaded<uint64_t>/32            13.5 ns         13.4 ns     51190679 cycles=32.4249
BM_accumulate_multithreaded<uint64_t>/64            13.8 ns         13.7 ns     50769297 cycles=33.9095
BM_accumulate_multithreaded<uint64_t>/128           13.3 ns         13.3 ns     52827916 cycles=31.6028
BM_accumulate_multithreaded<uint64_t>/256           17.1 ns         17.0 ns     40935617 cycles=44.5032
BM_accumulate_multithreaded<uint64_t>/512           20.0 ns         19.9 ns     35095742 cycles=59.4961
BM_accumulate_multithreaded<uint64_t>/1024          5314 ns         2573 ns       291340 cycles=24.8974k
BM_accumulate_multithreaded<uint64_t>/2048          5692 ns         2806 ns       256688 cycles=26.667k
BM_accumulate_multithreaded<uint64_t>/4096          6117 ns         3880 ns       181633 cycles=28.6607k
BM_accumulate_multithreaded<uint64_t>/8192         14346 ns         9689 ns        71662 cycles=67.2616k
BM_accumulate_multithreaded<uint64_t>/16384        20890 ns        16266 ns        44030 cycles=97.9601k
BM_accumulate_multithreaded<uint64_t>/32768        20501 ns        16020 ns        43590 cycles=96.1346k
BM_accumulate_multithreaded<uint64_t>/65536        20650 ns        16230 ns        43007 cycles=96.831k
BM_accumulate_multithreaded<uint64_t>/131072       21078 ns        16509 ns        42221 cycles=98.8401k
BM_accumulate_multithreaded<uint64_t>/262144       21967 ns        17106 ns        41414 cycles=103.01k
BM_accumulate_multithreaded<uint64_t>/524288       24647 ns        18766 ns        37047 cycles=115.583k
BM_accumulate_multithreaded<uint64_t>/1048576      31456 ns        21927 ns        31978 cycles=147.52k
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
```c++
std::vector<T> in; in.resize(s);
const auto end = std::exclusive_scan(in.begin(), in.end(), out.begin());
```

Benchmark:
```
2024-10-17T11:23:53+02:00
Run on (12 X 5453 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 1024 KiB (x6)
  L3 Unified 32768 KiB (x1)
------------------------------------------------------------------------------------------------------------
Benchmark                                                  Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------------------------
BM_stdexclusive_scan<uint8_t>/32                        13.2 ns         13.1 ns     52097045 cycles=30.8972
BM_stdexclusive_scan<uint8_t>/64                        18.6 ns         18.6 ns     37555445 cycles=56.0624
BM_stdexclusive_scan<uint8_t>/128                       28.6 ns         28.6 ns     24582618 cycles=102.791
BM_stdexclusive_scan<uint8_t>/256                       51.8 ns         51.7 ns     13401393 cycles=211.574
BM_stdexclusive_scan<uint8_t>/512                       99.5 ns         99.4 ns      7043017 cycles=435.734
BM_stdexclusive_scan<uint8_t>/1024                       198 ns          198 ns      3538886 cycles=898.214
BM_stdexclusive_scan<uint8_t>/2048                       388 ns          387 ns      1806897 cycles=1.78951k
BM_stdexclusive_scan<uint8_t>/4096                       769 ns          767 ns       912177 cycles=3.57575k
BM_stdexclusive_scan<uint8_t>/8192                      1532 ns         1529 ns       457778 cycles=7.15475k
BM_stdexclusive_scan<uint8_t>/16384                     3057 ns         3051 ns       229385 cycles=14.3075k
BM_stdexclusive_scan<uint8_t>/32768                     6108 ns         6097 ns       114906 cycles=28.622k
BM_stdexclusive_scan<uint8_t>/65536                    12241 ns        12218 ns        57292 cycles=57.3887k
BM_stdexclusive_scan<uint8_t>/131072                   24504 ns        24459 ns        28619 cycles=114.914k
BM_stdexclusive_scan<uint8_t>/262144                   49311 ns        49214 ns        14311 cycles=231.282k
BM_stdexclusive_scan<uint8_t>/524288                   98244 ns        98004 ns         7051 cycles=460.825k
BM_stdexclusive_scan<uint8_t>/1048576                 196510 ns       195980 ns         3575 cycles=921.779k
BM_stdexclusive_scan<uint32_t>/32                       19.6 ns         19.6 ns     35250511 cycles=60.8507
BM_stdexclusive_scan<uint32_t>/64                       29.1 ns         29.1 ns     24079026 cycles=105.328
BM_stdexclusive_scan<uint32_t>/128                      54.4 ns         54.3 ns     12892796 cycles=224.024
BM_stdexclusive_scan<uint32_t>/256                      92.4 ns         92.2 ns      7581857 cycles=402.404
BM_stdexclusive_scan<uint32_t>/512                       168 ns          167 ns      4143517 cycles=754.952
BM_stdexclusive_scan<uint32_t>/1024                      324 ns          323 ns      2164205 cycles=1.48885k
BM_stdexclusive_scan<uint32_t>/2048                      702 ns          701 ns       998431 cycles=3.26183k
BM_stdexclusive_scan<uint32_t>/4096                     1237 ns         1235 ns       574018 cycles=5.77351k
BM_stdexclusive_scan<uint32_t>/8192                     2426 ns         2422 ns       288954 cycles=11.3487k
BM_stdexclusive_scan<uint32_t>/16384                    4797 ns         4788 ns       146069 cycles=22.4691k
BM_stdexclusive_scan<uint32_t>/32768                    9552 ns         9533 ns        72681 cycles=44.7745k
BM_stdexclusive_scan<uint32_t>/65536                   19061 ns        19025 ns        36759 cycles=89.3815k
BM_stdexclusive_scan<uint32_t>/131072                  38683 ns        38593 ns        18329 cycles=181.428k
BM_stdexclusive_scan<uint32_t>/262144                  78668 ns        78455 ns         8934 cycles=368.994k
BM_stdexclusive_scan<uint32_t>/524288                 156479 ns       156059 ns         4488 cycles=733.999k
BM_stdexclusive_scan<uint32_t>/1048576                314686 ns       313842 ns         2244 cycles=1.47613M
BM_stdexclusive_scan<uint64_t>/32                       17.9 ns         17.9 ns     39180250 cycles=52.5071
BM_stdexclusive_scan<uint64_t>/64                       28.8 ns         28.7 ns     24293379 cycles=103.411
BM_stdexclusive_scan<uint64_t>/128                      53.5 ns         53.4 ns     13105024 cycles=219.449
BM_stdexclusive_scan<uint64_t>/256                       102 ns          102 ns      6873404 cycles=446.843
BM_stdexclusive_scan<uint64_t>/512                       198 ns          198 ns      3536099 cycles=898.082
BM_stdexclusive_scan<uint64_t>/1024                      393 ns          392 ns      1784852 cycles=1.81104k
BM_stdexclusive_scan<uint64_t>/2048                      780 ns          779 ns       899399 cycles=3.62773k
BM_stdexclusive_scan<uint64_t>/4096                     1560 ns         1558 ns       449419 cycles=7.28797k
BM_stdexclusive_scan<uint64_t>/8192                     3114 ns         3109 ns       225093 cycles=14.5764k
BM_stdexclusive_scan<uint64_t>/16384                    6216 ns         6205 ns       112803 cycles=29.1288k
BM_stdexclusive_scan<uint64_t>/32768                   12439 ns        12416 ns        56384 cycles=58.3187k
BM_stdexclusive_scan<uint64_t>/65536                   24914 ns        24856 ns        28175 cycles=116.836k
BM_stdexclusive_scan<uint64_t>/131072                  50142 ns        50010 ns        10000 cycles=235.18k
BM_stdexclusive_scan<uint64_t>/262144                 100108 ns        99839 ns         7015 cycles=469.568k
BM_stdexclusive_scan<uint64_t>/524288                 200663 ns       200114 ns         3501 cycles=941.26k
BM_stdexclusive_scan<uint64_t>/1048576                402146 ns       401029 ns         1747 cycles=1.8864M
BM_exclusive_scan<uint8_t>/32                           13.1 ns         13.1 ns     53344472 cycles=30.8656
BM_exclusive_scan<uint8_t>/64                           18.7 ns         18.6 ns     37555146 cycles=56.2841
BM_exclusive_scan<uint8_t>/128                          29.3 ns         29.2 ns     23962941 cycles=105.852
BM_exclusive_scan<uint8_t>/256                          53.5 ns         53.4 ns     13123777 cycles=219.572
BM_exclusive_scan<uint8_t>/512                           102 ns          102 ns      6844491 cycles=448.987
BM_exclusive_scan<uint8_t>/1024                          201 ns          201 ns      3484140 cycles=912.59
BM_exclusive_scan<uint8_t>/2048                          394 ns          393 ns      1781592 cycles=1.81478k
BM_exclusive_scan<uint8_t>/4096                          780 ns          778 ns       899551 cycles=3.62578k
BM_exclusive_scan<uint8_t>/8192                         1552 ns         1549 ns       451775 cycles=7.2498k
BM_exclusive_scan<uint8_t>/16384                        3055 ns         3050 ns       227661 cycles=14.3016k
BM_exclusive_scan<uint8_t>/32768                        6112 ns         6101 ns       114916 cycles=28.6422k
BM_exclusive_scan<uint8_t>/65536                       12247 ns        12223 ns        56660 cycles=57.4163k
BM_exclusive_scan<uint8_t>/131072                      24383 ns        24338 ns        28675 cycles=114.346k
BM_exclusive_scan<uint8_t>/262144                      49502 ns        49406 ns        14207 cycles=232.177k
BM_exclusive_scan<uint8_t>/524288                      98982 ns        98745 ns         7089 cycles=464.288k
BM_exclusive_scan<uint8_t>/1048576                    199084 ns       198549 ns         3529 cycles=933.858k
BM_exclusive_scan<uint32_t>/32                          14.6 ns         14.6 ns     48394058 cycles=36.8546
BM_exclusive_scan<uint32_t>/64                          26.1 ns         26.1 ns     27593350 cycles=91.0441
BM_exclusive_scan<uint32_t>/128                         47.2 ns         47.1 ns     14851365 cycles=189.938
BM_exclusive_scan<uint32_t>/256                         83.1 ns         83.0 ns      8377012 cycles=358.545
BM_exclusive_scan<uint32_t>/512                          154 ns          154 ns      4540806 cycles=692.886
BM_exclusive_scan<uint32_t>/1024                         294 ns          294 ns      2381712 cycles=1.34987k
BM_exclusive_scan<uint32_t>/2048                         580 ns          579 ns      1208789 cycles=2.68915k
BM_exclusive_scan<uint32_t>/4096                        1131 ns         1129 ns       620186 cycles=5.27207k
BM_exclusive_scan<uint32_t>/8192                        2374 ns         2370 ns       295235 cycles=11.1048k
BM_exclusive_scan<uint32_t>/16384                       4771 ns         4762 ns       146897 cycles=22.348k
BM_exclusive_scan<uint32_t>/32768                       9532 ns         9516 ns        73558 cycles=44.6846k
BM_exclusive_scan<uint32_t>/65536                      19047 ns        19010 ns        36960 cycles=89.3156k
BM_exclusive_scan<uint32_t>/131072                     38240 ns        38146 ns        18340 cycles=179.348k
BM_exclusive_scan<uint32_t>/262144                     76338 ns        76135 ns         9192 cycles=358.062k
BM_exclusive_scan<uint32_t>/524288                    155789 ns       155376 ns         4528 cycles=730.762k
BM_exclusive_scan<uint32_t>/1048576                   306133 ns       305306 ns         2289 cycles=1.43601M
BM_exclusive_scan<uint64_t>/32                          14.5 ns         14.5 ns     48058487 cycles=36.7308
BM_exclusive_scan<uint64_t>/64                          25.4 ns         25.4 ns     27552780 cycles=87.869
BM_exclusive_scan<uint64_t>/128                         46.8 ns         46.7 ns     14971213 cycles=188.291
BM_exclusive_scan<uint64_t>/256                         82.3 ns         82.1 ns      8504942 cycles=354.453
BM_exclusive_scan<uint64_t>/512                          153 ns          152 ns      4597769 cycles=684.154
BM_exclusive_scan<uint64_t>/1024                         292 ns          291 ns      2403033 cycles=1.33691k
BM_exclusive_scan<uint64_t>/2048                         566 ns          565 ns      1236205 cycles=2.62395k
BM_exclusive_scan<uint64_t>/4096                        1231 ns         1229 ns       570701 cycles=5.74184k
BM_exclusive_scan<uint64_t>/8192                        2881 ns         2876 ns       243400 cycles=13.4847k
BM_exclusive_scan<uint64_t>/16384                       4939 ns         4930 ns       141994 cycles=23.1352k
BM_exclusive_scan<uint64_t>/32768                      10018 ns        10000 ns        70021 cycles=46.9611k
BM_exclusive_scan<uint64_t>/65536                      19894 ns        19845 ns        35288 cycles=93.288k
BM_exclusive_scan<uint64_t>/131072                     39624 ns        39519 ns        17712 cycles=185.842k
BM_exclusive_scan<uint64_t>/262144                     93674 ns        93419 ns         7515 cycles=439.383k
BM_exclusive_scan<uint64_t>/524288                    158793 ns       158355 ns         4427 cycles=744.853k
BM_exclusive_scan<uint64_t>/1048576                   352124 ns       351172 ns         1990 cycles=1.65175M
BM_exclusive_scan_multithreaded<uint8_t>/32             13.5 ns         13.5 ns     51758058 cycles=31.8839
BM_exclusive_scan_multithreaded<uint8_t>/64             20.4 ns         20.3 ns     34389773 cycles=63.9228
BM_exclusive_scan_multithreaded<uint8_t>/128            30.9 ns         30.8 ns     22610847 cycles=113.039
BM_exclusive_scan_multithreaded<uint8_t>/256            55.0 ns         54.9 ns     12743010 cycles=226.27
BM_exclusive_scan_multithreaded<uint8_t>/512             104 ns          104 ns      6724270 cycles=457.283
BM_exclusive_scan_multithreaded<uint8_t>/1024            203 ns          203 ns      3444178 cycles=922.501
BM_exclusive_scan_multithreaded<uint8_t>/2048            396 ns          396 ns      1769264 cycles=1.82782k
BM_exclusive_scan_multithreaded<uint8_t>/4096            784 ns          783 ns       893637 cycles=3.64768k
BM_exclusive_scan_multithreaded<uint8_t>/8192           1560 ns         1558 ns       449398 cycles=7.28823k
BM_exclusive_scan_multithreaded<uint8_t>/16384          3108 ns         3103 ns       225556 cycles=14.5489k
BM_exclusive_scan_multithreaded<uint8_t>/32768          6218 ns         6207 ns       112792 cycles=29.1375k
BM_exclusive_scan_multithreaded<uint8_t>/65536         12431 ns        12409 ns        56429 cycles=58.2831k
BM_exclusive_scan_multithreaded<uint8_t>/131072        41306 ns        31367 ns        22456 cycles=193.724k
BM_exclusive_scan_multithreaded<uint8_t>/262144        44440 ns        32871 ns        21420 cycles=208.422k
BM_exclusive_scan_multithreaded<uint8_t>/524288        52394 ns        35217 ns        19851 cycles=245.74k
BM_exclusive_scan_multithreaded<uint8_t>/1048576       71414 ns        36680 ns        18982 cycles=334.96k
BM_exclusive_scan_multithreaded<uint32_t>/32            16.3 ns         16.3 ns     43248190 cycles=44.6254
BM_exclusive_scan_multithreaded<uint32_t>/64            28.5 ns         28.4 ns     22702735 cycles=101.571
BM_exclusive_scan_multithreaded<uint32_t>/128           47.1 ns         47.0 ns     14537473 cycles=189.439
BM_exclusive_scan_multithreaded<uint32_t>/256           81.4 ns         81.3 ns      8624789 cycles=350.687
BM_exclusive_scan_multithreaded<uint32_t>/512            151 ns          151 ns      4637431 cycles=677.98
BM_exclusive_scan_multithreaded<uint32_t>/1024           288 ns          288 ns      2401346 cycles=1.32135k
BM_exclusive_scan_multithreaded<uint32_t>/2048           571 ns          570 ns      1228192 cycles=2.64862k
BM_exclusive_scan_multithreaded<uint32_t>/4096          1114 ns         1112 ns       629333 cycles=5.19637k
BM_exclusive_scan_multithreaded<uint32_t>/8192          2336 ns         2332 ns       300225 cycles=10.9282k
BM_exclusive_scan_multithreaded<uint32_t>/16384         4688 ns         4680 ns       149563 cycles=21.9613k
BM_exclusive_scan_multithreaded<uint32_t>/32768         9325 ns         9308 ns        75170 cycles=43.7117k
BM_exclusive_scan_multithreaded<uint32_t>/65536        18676 ns        18641 ns        37580 cycles=87.5782k
BM_exclusive_scan_multithreaded<uint32_t>/131072       43102 ns        32408 ns        21797 cycles=202.148k
BM_exclusive_scan_multithreaded<uint32_t>/262144       47762 ns        34757 ns        19614 cycles=224.012k
BM_exclusive_scan_multithreaded<uint32_t>/524288       63485 ns        38760 ns        18187 cycles=297.765k
BM_exclusive_scan_multithreaded<uint32_t>/1048576      89711 ns        40153 ns        17437 cycles=420.789k
BM_exclusive_scan_multithreaded<uint64_t>/32            15.8 ns         15.7 ns     44239833 cycles=41.9662
BM_exclusive_scan_multithreaded<uint64_t>/64            30.8 ns         30.7 ns     23105798 cycles=112.172
BM_exclusive_scan_multithreaded<uint64_t>/128           47.7 ns         47.6 ns     14706681 cycles=192.458
BM_exclusive_scan_multithreaded<uint64_t>/256           81.2 ns         81.1 ns      8643324 cycles=349.56
BM_exclusive_scan_multithreaded<uint64_t>/512            150 ns          149 ns      4689184 cycles=670.975
BM_exclusive_scan_multithreaded<uint64_t>/1024           361 ns          360 ns      1934847 cycles=1.66079k
BM_exclusive_scan_multithreaded<uint64_t>/2048           561 ns          560 ns      1252605 cycles=2.59815k
BM_exclusive_scan_multithreaded<uint64_t>/4096          1271 ns         1268 ns       553065 cycles=5.92877k
BM_exclusive_scan_multithreaded<uint64_t>/8192          2469 ns         2464 ns       283661 cycles=11.5486k
BM_exclusive_scan_multithreaded<uint64_t>/16384         4949 ns         4940 ns       141699 cycles=23.1857k
BM_exclusive_scan_multithreaded<uint64_t>/32768         9891 ns         9873 ns        70904 cycles=46.3648k
BM_exclusive_scan_multithreaded<uint64_t>/65536        19987 ns        19940 ns        35183 cycles=93.7235k
BM_exclusive_scan_multithreaded<uint64_t>/131072       43344 ns        32866 ns        21356 cycles=203.28k
BM_exclusive_scan_multithreaded<uint64_t>/262144       50601 ns        36293 ns        19330 cycles=237.332k
BM_exclusive_scan_multithreaded<uint64_t>/524288       67799 ns        40565 ns        17219 cycles=318.005k
BM_exclusive_scan_multithreaded<uint64_t>/1048576     108223 ns        44499 ns        15750 cycles=507.629k
```

fill:
======

Usage:
```c++
std::vector<T> in; in.resize(s);
const auto d = cryptanalysislib::fill(in.begin(), in.end(), 0);
```

Benchmark:
```

```

find:
======
```c++
const auto d = cryptanalysislib::find(in.begin(), in.end(), 0);
```

Benchmark:
```

```

for_each:
======


Usage:
```c++
std::vector<T> in; in.resize(s);
std::fill(in.begin(), in.end(), 1);
const auto d = cryptanalysislib::accumulate(in.begin(), in.end(), 0);
```

Benchmark:
```

```


gcd:
======
Usage:
```c++

```

Benchmark:
```

```

histogram:
=========

Usage:
```c++
std::vector<T> in; in.resize(s);
std::fill(in.begin(), in.end(), 1);
const auto d = cryptanalysislib::accumulate(in.begin(), in.end(), 0);
```

Benchmark:
```
2024-09-08T22:34:54+02:00
Run on (16 X 1700 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 1280 KiB (x8)
  L3 Unified 12288 KiB (x1)
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

Usage:
```c++

```

Benchmark:
```

```


int2weight:
======

Usage:
```c++

```

Benchmark:
```

```

max:
======

Usage:
```c++

```

Benchmark:
```

```

min:
======

Usage:
```c++

```

Benchmark:
```

```

pcs:
======

Usage:
```c++

```

Benchmark:
```

```

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

Usage:
```c++

```

Benchmark:
```

```
reduce:
======

Usage:
```c++

```

Benchmark:
```

```

regex:
======

:
```c++

```

Benchmark:
```

```

rotate:
======

Usage:
```c++

```

Benchmark:
```

```


subsetsum:
======

Usage:
```c++

```

Benchmark:
```

```

transform:
======
Usage:
```c++

```

Benchmark:
```

```