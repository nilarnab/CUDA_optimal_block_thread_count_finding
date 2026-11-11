# LS-CAT pilot collection summary

Status: **COMPLETE**

COMPLETE means the recorded plan has exactly one successful, correct, finite-positive timing for every planned trial. It does not establish statistical sufficiency or model generalization.

Planned trials: 2320; recorded trials: 2320; valid timings: 2320; failed records: 0.
Missing trials: 0; extra trials: 0.
Action groups: 232 full, 0 failed, 0 incomplete.

Only action groups with every planned round present exactly once and valid enter action_table.csv. The default is the fully valid action with the largest block_x * block_y; ties prefer larger block_x, then block_y. The tested oracle is the minimum median over fully valid actions. reward_log_vs_default = log(default_median_ms / action_median_ms); regret = action_median_ms / best_median_ms - 1. These are this pilot's definitions, not a reproduction of the original paper's score.

relative_mad = median(abs(trial_ms - median_ms)) / median_ms; it is a ratio, not a percentage.

These comparisons describe the recorded kernels and sizes. The same measurements choose the tested best action and estimate its speedup.

- "gemm__k16384" size 8192: 9 fully valid actions; default (32, 32), tested best (32, 16); default / tested-best median = 1.00737x.

- "gemm__k2048" size 2048: 9 fully valid actions; default (32, 32), tested best (32, 16); default / tested-best median = 1.0391x.

- "gemm__k4096" size 4096: 9 fully valid actions; default (32, 32), tested best (32, 32); default / tested-best median = 1x.

- "gemm__k8192" size 8192: 9 fully valid actions; default (32, 32), tested best (32, 32); default / tested-best median = 1x.

- "mean__k262144" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 11.4054x.

- "mean__k65536" size 4096: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 8.21398x.

- "nbody" size 65536: 16 fully valid actions; default (1024, 1), tested best (1024, 1); default / tested-best median = 1x.

- "nbody" size 262144: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.02448x.

- "nbody" size 524288: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.04339x.

- "nbody" size 1048576: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.05017x.

- "segmented_sum__k262144" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 17.0851x.

- "segmented_sum__k65536" size 4096: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 7.53357x.

- "syrk__k1024" size 1024: 9 fully valid actions; default (32, 32), tested best (8, 16); default / tested-best median = 2.14896x.

- "syrk__k2048" size 2048: 9 fully valid actions; default (32, 32), tested best (8, 16); default / tested-best median = 2.10631x.

- "syrk__k4096" size 4096: 9 fully valid actions; default (32, 32), tested best (8, 32); default / tested-best median = 2.07693x.

- "syrk__k8192" size 4096: 9 fully valid actions; default (32, 32), tested best (8, 32); default / tested-best median = 2.14675x.

- "variance__k1024" size 256: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 3.3149x.

- "variance__k16384" size 2048: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 13.0688x.
