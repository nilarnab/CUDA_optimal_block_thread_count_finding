# LS-CAT pilot collection summary

Status: **COMPLETE**

COMPLETE means the recorded plan has exactly one successful, correct, finite-positive timing for every planned trial. It does not establish statistical sufficiency or model generalization.

Planned trials: 5120; recorded trials: 5120; valid timings: 5120; failed records: 0.
Missing trials: 0; extra trials: 0.
Action groups: 512 full, 0 failed, 0 incomplete.

Only action groups with every planned round present exactly once and valid enter action_table.csv. The default is the fully valid action with the largest block_x * block_y; ties prefer larger block_x, then block_y. The tested oracle is the minimum median over fully valid actions. reward_log_vs_default = log(default_median_ms / action_median_ms); regret = action_median_ms / best_median_ms - 1. These are this pilot's definitions, not a reproduction of the original paper's score.

relative_mad = median(abs(trial_ms - median_ms)) / median_ms; it is a ratio, not a percentage.

These comparisons describe the recorded kernels and sizes. The same measurements choose the tested best action and estimate its speedup.

- "accumulate__k16384" size 2048: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.11754x.

- "accumulate__k2097152" size 128: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.0161x.

- "accumulate__k262144" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.16416x.

- "accumulate__k4194304" size 64: 16 fully valid actions; default (1024, 1), tested best (896, 1); default / tested-best median = 1.0006x.

- "binarize_input__k1048576" size 128: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.01948x.

- "binarize_input__k16384" size 512: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.10481x.

- "binarize_input__k262144" size 512: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.11257x.

- "binarize_input__k4194304" size 32: 16 fully valid actions; default (1024, 1), tested best (640, 1); default / tested-best median = 1.00001x.

- "binarize_weights__k16384" size 512: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 6.03995x.

- "binarize_weights__k262144" size 512: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 5.80801x.

- "binarize_weights__k4194304" size 32: 16 fully valid actions; default (1024, 1), tested best (384, 1); default / tested-best median = 1.00033x.

- "binarize_weights__k8388608" size 16: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.00001x.

- "matrix_transpose" size 12288: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.03049x.

- "mean__k262144" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 6.1192x.

- "mean__k4194304" size 64: 16 fully valid actions; default (1024, 1), tested best (896, 1); default / tested-best median = 1.00001x.

- "mean__k65536" size 4096: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 6.18701x.

- "mean__k8388608" size 32: 16 fully valid actions; default (1024, 1), tested best (448, 1); default / tested-best median = 1.00001x.

- "nbody" size 65536: 16 fully valid actions; default (1024, 1), tested best (640, 1); default / tested-best median = 1.57964x.

- "nbody" size 262144: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.21766x.

- "nbody" size 786432: 16 fully valid actions; default (1024, 1), tested best (384, 1); default / tested-best median = 1.11409x.

- "periodic_tile" size 22528: 16 fully valid actions; default (1024, 1), tested best (192, 1); default / tested-best median = 1.34391x.

- "repeat_vector" size 22528: 16 fully valid actions; default (1024, 1), tested best (512, 1); default / tested-best median = 1.19021x.

- "segmented_sum__k1048576" size 256: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 3.03903x.

- "segmented_sum__k65536" size 4096: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 11.7263x.

- "segmented_sum__k8388608" size 32: 16 fully valid actions; default (1024, 1), tested best (576, 1); default / tested-best median = 1.00002x.

- "softmax__k1048576" size 128: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.21628x.

- "softmax__k16384" size 512: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 4.13597x.

- "softmax__k2097152" size 64: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.05039x.

- "softmax__k65536" size 1024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 8.19295x.

- "variance__k16384" size 2048: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 3.25691x.

- "variance__k65536" size 4096: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 3.23421x.

- "vector_add" size 12288: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.00865x.
