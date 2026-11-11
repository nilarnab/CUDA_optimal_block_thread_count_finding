# LS-CAT pilot collection summary

Status: **COMPLETE**

COMPLETE means the recorded plan has exactly one successful, correct, finite-positive timing for every planned trial. It does not establish statistical sufficiency or model generalization.

Planned trials: 2070; recorded trials: 2070; valid timings: 2070; failed records: 0.
Missing trials: 0; extra trials: 0.
Action groups: 207 full, 0 failed, 0 incomplete.

Only action groups with every planned round present exactly once and valid enter action_table.csv. The default is the fully valid action with the largest block_x * block_y; ties prefer larger block_x, then block_y. The tested oracle is the minimum median over fully valid actions. reward_log_vs_default = log(default_median_ms / action_median_ms); regret = action_median_ms / best_median_ms - 1. These are this pilot's definitions, not a reproduction of the original paper's score.

relative_mad = median(abs(trial_ms - median_ms)) / median_ms; it is a ratio, not a percentage.

These comparisons describe the recorded kernels and sizes. The same measurements choose the tested best action and estimate its speedup.

- "gemm__k16384" size 8192: 9 fully valid actions; default (32, 32), tested best (32, 32); default / tested-best median = 1x.

- "gemm__k2048" size 2048: 9 fully valid actions; default (32, 32), tested best (32, 8); default / tested-best median = 1.0111x.

- "gemm__k4096" size 4096: 9 fully valid actions; default (32, 32), tested best (32, 32); default / tested-best median = 1x.

- "mean__k262144" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 6.1208x.

- "mean__k65536" size 4096: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 6.21527x.

- "nbody" size 65536: 16 fully valid actions; default (1024, 1), tested best (640, 1); default / tested-best median = 1.57967x.

- "nbody" size 262144: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.21897x.

- "nbody" size 1048576: 16 fully valid actions; default (1024, 1), tested best (512, 1); default / tested-best median = 1.0482x.

- "segmented_sum__k1048576" size 256: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 3.04108x.

- "segmented_sum__k65536" size 4096: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 11.7439x.

- "syrk__k2048" size 2048: 9 fully valid actions; default (32, 32), tested best (8, 32); default / tested-best median = 2.0637x.

- "syrk__k4096" size 4096: 9 fully valid actions; default (32, 32), tested best (8, 8); default / tested-best median = 2.06403x.

- "syrk__k8192" size 4096: 9 fully valid actions; default (32, 32), tested best (8, 8); default / tested-best median = 2.06646x.

- "syrk__k8192" size 8192: 9 fully valid actions; default (32, 32), tested best (8, 32); default / tested-best median = 2.06154x.

- "variance__k16384" size 2048: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 3.25691x.

- "variance__k65536" size 4096: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 3.23344x.
