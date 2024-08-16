Benchmark on which the selection of the sorting algorithm is based on

```
Running ./bench/search/bench_b63_search_search
Run on (12 X 5453 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 1024 KiB (x6)
  L3 Unified 32768 KiB (x1)
Load Average: 0.74, 0.61, 0.36
----------------------------------------------------------------------------------------------
Benchmark                                                    Time             CPU   Iterations
----------------------------------------------------------------------------------------------
stdlowerbound_bench/128                                   24.7 ns         24.6 ns     27109249
stdlowerbound_bench/256                                   25.9 ns         25.9 ns     26987102
stdlowerbound_bench/512                                   28.6 ns         28.6 ns     24484215
stdlowerbound_bench/1024                                  31.3 ns         31.2 ns     22345643
stdlowerbound_bench/2048                                  35.4 ns         35.4 ns     20320529
stdlowerbound_bench/4096                                  38.1 ns         38.1 ns     18361310
stdlowerbound_bench/8192                                  45.3 ns         45.3 ns     15447648
stdlowerbound_bench/16384                                 51.2 ns         51.2 ns     13690049
stdlowerbound_bench/32768                                 56.1 ns         56.1 ns     12480629
stdlowerbound_bench/65536                                 62.2 ns         62.1 ns     11266986
stdlowerbound_bench/131072                                71.0 ns         71.0 ns      9866611
stdlowerbound_bench/262144                                93.8 ns         93.7 ns      7468117
stdlowerbound_bench/524288                                 113 ns          113 ns      6339430
stdlowerbound_bench/1048576                                123 ns          123 ns      5711016
stdlowerbound_bench/2097152                                134 ns          134 ns      5305124
stdlowerbound_bench/4194304                                151 ns          151 ns      4731610
stdlowerbound_bench/8388608                                287 ns          287 ns      2465110
stdlowerbound_bench/16777216                               381 ns          381 ns      1842399
stdlowerbound_bench/33554432                               457 ns          456 ns      1534395
stdlowerbound_bench/67108864                               534 ns          533 ns      1290650
stdlowerbound_bench/134217728                              614 ns          613 ns      1125423
stdlowerbound_bench/268435456                              688 ns          687 ns      1001721
stdlowerbound_bench_BigO                                 12.53 lgN       12.51 lgN  
stdlowerbound_bench_RMS                                     80 %            80 %    
branchless_lower_bound_bench/128                          24.8 ns         24.8 ns     28236137
branchless_lower_bound_bench/256                          27.5 ns         27.5 ns     25469439
branchless_lower_bound_bench/512                          30.0 ns         29.9 ns     23402365
branchless_lower_bound_bench/1024                         32.7 ns         32.6 ns     21501776
branchless_lower_bound_bench/2048                         35.1 ns         35.1 ns     19923673
branchless_lower_bound_bench/4096                         38.4 ns         38.4 ns     18278480
branchless_lower_bound_bench/8192                         46.0 ns         46.0 ns     15251540
branchless_lower_bound_bench/16384                        52.0 ns         51.9 ns     13490763
branchless_lower_bound_bench/32768                        56.8 ns         56.7 ns     12306510
branchless_lower_bound_bench/65536                        61.3 ns         61.2 ns     11426671
branchless_lower_bound_bench/131072                       66.4 ns         66.3 ns     10562845
branchless_lower_bound_bench/262144                       99.9 ns         99.8 ns      7024113
branchless_lower_bound_bench/524288                        112 ns          112 ns      6244793
branchless_lower_bound_bench/1048576                       127 ns          127 ns      5544009
branchless_lower_bound_bench/2097152                       138 ns          138 ns      5104267
branchless_lower_bound_bench/4194304                       152 ns          152 ns      4602161
branchless_lower_bound_bench/8388608                       289 ns          289 ns      2409320
branchless_lower_bound_bench/16777216                      387 ns          387 ns      1809004
branchless_lower_bound_bench/33554432                      467 ns          466 ns      1480985
branchless_lower_bound_bench/67108864                      544 ns          543 ns      1259933
branchless_lower_bound_bench/134217728                     626 ns          626 ns      1100671
branchless_lower_bound_bench/268435456                     706 ns          705 ns       980040
branchless_lower_bound_bench_BigO                        12.76 lgN       12.74 lgN  
branchless_lower_bound_bench_RMS                            80 %            80 %    
lower_bound_interpolation_search_3p_bench/128             22.4 ns         22.4 ns     24957228
lower_bound_interpolation_search_3p_bench/256             28.0 ns         27.9 ns     29658914
lower_bound_interpolation_search_3p_bench/512             27.8 ns         27.8 ns     28842987
lower_bound_interpolation_search_3p_bench/1024            29.2 ns         29.2 ns     25705594
lower_bound_interpolation_search_3p_bench/2048            33.0 ns         33.0 ns     21833632
lower_bound_interpolation_search_3p_bench/4096            33.5 ns         33.5 ns     22616553
lower_bound_interpolation_search_3p_bench/8192            35.4 ns         35.3 ns     20782272
lower_bound_interpolation_search_3p_bench/16384           35.6 ns         35.6 ns     17929776
lower_bound_interpolation_search_3p_bench/32768           37.5 ns         37.5 ns     18619756
lower_bound_interpolation_search_3p_bench/65536           39.1 ns         39.1 ns     18778024
lower_bound_interpolation_search_3p_bench/131072          41.1 ns         41.0 ns     17393432
lower_bound_interpolation_search_3p_bench/262144          44.1 ns         44.1 ns     15346493
lower_bound_interpolation_search_3p_bench/524288          50.6 ns         50.5 ns     10000000
lower_bound_interpolation_search_3p_bench/1048576         53.9 ns         53.8 ns     13071627
lower_bound_interpolation_search_3p_bench/2097152         53.9 ns         53.8 ns     13337542
lower_bound_interpolation_search_3p_bench/4194304         61.3 ns         61.2 ns     11247491
lower_bound_interpolation_search_3p_bench/8388608          132 ns          132 ns      4999881
lower_bound_interpolation_search_3p_bench/16777216         173 ns          172 ns      3899032
lower_bound_interpolation_search_3p_bench/33554432         202 ns          202 ns      3333860
lower_bound_interpolation_search_3p_bench/67108864         226 ns          226 ns      3159013
lower_bound_interpolation_search_3p_bench/134217728        242 ns          242 ns      2899125
lower_bound_interpolation_search_3p_bench/268435456        232 ns          232 ns      2817045
lower_bound_interpolation_search_3p_bench_BigO            5.37 lgN        5.37 lgN  
lower_bound_interpolation_search_3p_bench_RMS               61 %            61 %    
```
