



2024-09-05T11:29:29+02:00
Running ./bench/sort/bench_b63_sort_sortingnetwork
Run on (16 X 1700 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 1280 KiB (x8)
  L3 Unified 12288 KiB (x1)
Load Average: 0.45, 0.41, 0.22
-----------------------------------------------------------------------------------
Benchmark                                         Time             CPU   Iterations
-----------------------------------------------------------------------------------
bench_sortingnetwork_sort_u32x64/128            293 ns          292 ns      2395762
bench_sortingnetwork_sort_u32x128/128           696 ns          694 ns       988054
bench_sortingnetwork_sort_u32x128_v2/128        202 ns          201 ns      3484627
bench_djb_sort/16                              38.9 ns         38.8 ns     18486121
bench_djb_sort/32                              75.3 ns         75.1 ns      9325456
bench_djb_sort/48                               244 ns          243 ns      2874629
bench_djb_sort/64                               135 ns          135 ns      5200177
bench_djb_sort/80                               363 ns          362 ns      1934936
bench_djb_sort/96                               368 ns          367 ns      1906096
bench_djb_sort/112                              379 ns          378 ns      1856396
bench_djb_sort/128                              304 ns          303 ns      2311001
bench_sortingnetwork_small_avx2/16             25.2 ns         25.1 ns     27876420
bench_sortingnetwork_small_avx2/32             32.0 ns         31.9 ns     21973723
bench_sortingnetwork_small_avx2/48             50.7 ns         50.5 ns     13821914
bench_sortingnetwork_small_avx2/64             60.5 ns         60.3 ns     11617126
bench_sortingnetwork_small_avx2/80             83.6 ns         83.3 ns      8404958
bench_sortingnetwork_small_avx2/96              203 ns          202 ns      3464803
bench_sortingnetwork_small_avx2/112             238 ns          237 ns      2948142
bench_sortingnetwork_small_avx2/128             286 ns          285 ns      2462094
bench_sortingnetwork_small_avx2/144             332 ns          331 ns      2117066
bench_sortingnetwork_small_avx2/160             342 ns          341 ns      2052196
bench_sortingnetwork_small_avx2/176             366 ns          365 ns      1917019
bench_sortingnetwork_small_avx2/192             379 ns          378 ns      1850167
bench_sortingnetwork_small_avx2/208             433 ns          432 ns      1619784
bench_sortingnetwork_small_avx2/224             586 ns          584 ns      1216656
bench_sortingnetwork_small_avx2/240             613 ns          611 ns      1145464
bench_sortingnetwork_small_avx2/256             737 ns          735 ns       952712
