# LS-CAT pilot collection summary

Status: **COMPLETE**

COMPLETE means the recorded plan has exactly one successful, correct, finite-positive timing for every planned trial. It does not establish statistical sufficiency or model generalization.

Planned trials: 5120; recorded trials: 5120; valid timings: 5120; failed records: 0.
Missing trials: 0; extra trials: 0.
Action groups: 512 full, 0 failed, 0 incomplete.

Only action groups with every planned round present exactly once and valid enter action_table.csv. The default is the fully valid action with the largest block_x * block_y; ties prefer larger block_x, then block_y. The tested oracle is the minimum median over fully valid actions. reward_log_vs_default = log(default_median_ms / action_median_ms); regret = action_median_ms / best_median_ms - 1. These are this pilot's definitions, not a reproduction of the original paper's score.

relative_mad = median(abs(trial_ms - median_ms)) / median_ms; it is a ratio, not a percentage.

These comparisons describe the recorded kernels and sizes. The same measurements choose the tested best action and estimate its speedup.

- "accumulate__k16384" size 2048: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.08756x.

- "accumulate__k2097152" size 128: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.00017x.

- "accumulate__k262144" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.04804x.

- "accumulate__k4194304" size 64: 16 fully valid actions; default (1024, 1), tested best (640, 1); default / tested-best median = 1.00002x.

- "binarize_input__k2097152" size 64: 16 fully valid actions; default (1024, 1), tested best (384, 1); default / tested-best median = 1.00005x.

- "binarize_input__k262144" size 512: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.05358x.

- "binarize_input__k4194304" size 32: 16 fully valid actions; default (1024, 1), tested best (768, 1); default / tested-best median = 1.00001x.

- "binarize_input__k65536" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.22348x.

- "binarize_weights__k16384" size 512: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 8.52953x.

- "binarize_weights__k16777216" size 8: 16 fully valid actions; default (1024, 1), tested best (448, 1); default / tested-best median = 1.00001x.

- "binarize_weights__k262144" size 512: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 8.5601x.

- "binarize_weights__k4194304" size 32: 16 fully valid actions; default (1024, 1), tested best (192, 1); default / tested-best median = 1.00003x.

- "matrix_transpose" size 12288: 16 fully valid actions; default (1024, 1), tested best (512, 1); default / tested-best median = 1.02032x.

- "mean__k262144" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 11.3849x.

- "mean__k65536" size 4096: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 7.6565x.

- "mean__k8388608" size 32: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.00002x.

- "nbody" size 65536: 16 fully valid actions; default (1024, 1), tested best (1024, 1); default / tested-best median = 1x.

- "nbody" size 262144: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.01641x.

- "nbody" size 524288: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.04873x.

- "nbody" size 786432: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.04703x.

- "periodic_tile" size 22528: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.64634x.

- "repeat_vector" size 22528: 16 fully valid actions; default (1024, 1), tested best (192, 1); default / tested-best median = 1.12689x.

- "segmented_sum__k262144" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 17.1687x.

- "segmented_sum__k65536" size 4096: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 7.497x.

- "softmax__k16384" size 512: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 5.99134x.

- "softmax__k2097152" size 64: 16 fully valid actions; default (1024, 1), tested best (512, 1); default / tested-best median = 1.00054x.

- "softmax__k4194304" size 32: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.00001x.

- "softmax__k65536" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 12.0786x.

- "variance__k1024" size 256: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 3.27614x.

- "variance__k16384" size 2048: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 13.0629x.

- "variance__k262144" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 13.0686x.

- "vector_add" size 12288: 16 fully valid actions; default (1024, 1), tested best (704, 1); default / tested-best median = 1.00171x.
