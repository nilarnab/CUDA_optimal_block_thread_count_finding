# LS-CAT pilot collection summary

Status: **COMPLETE**

COMPLETE means the recorded plan has exactly one successful, correct, finite-positive timing for every planned trial. It does not establish statistical sufficiency or model generalization.

Planned trials: 7000; recorded trials: 7000; valid timings: 7000; failed records: 0.
Missing trials: 0; extra trials: 0.
Action groups: 700 full, 0 failed, 0 incomplete.

Only action groups with every planned round present exactly once and valid enter action_table.csv. The default is the fully valid action with the largest block_x * block_y; ties prefer larger block_x, then block_y. The tested oracle is the minimum median over fully valid actions. reward_log_vs_default = log(default_median_ms / action_median_ms); regret = action_median_ms / best_median_ms - 1. These are this pilot's definitions, not a reproduction of the original paper's score.

relative_mad = median(abs(trial_ms - median_ms)) / median_ms; it is a ratio, not a percentage.

These comparisons describe the recorded kernels and sizes. The same measurements choose the tested best action and estimate its speedup.

- "matrix_add" size 240: 20 fully valid actions; default (32, 32), tested best (1, 128); default / tested-best median = 2.73475x.

- "matrix_add" size 496: 20 fully valid actions; default (32, 32), tested best (1, 128); default / tested-best median = 3.41254x.

- "matrix_add" size 784: 20 fully valid actions; default (32, 32), tested best (1, 512); default / tested-best median = 1.89079x.

- "matrix_add" size 1016: 20 fully valid actions; default (32, 32), tested best (1, 512); default / tested-best median = 1.55482x.

- "matrix_add" size 1232: 20 fully valid actions; default (32, 32), tested best (1, 512); default / tested-best median = 1.80699x.

- "matrix_add" size 1680: 20 fully valid actions; default (32, 32), tested best (1, 896); default / tested-best median = 1.8462x.

- "matrix_add" size 2024: 20 fully valid actions; default (32, 32), tested best (1, 704); default / tested-best median = 1.50165x.

- "matrix_transpose" size 240: 16 fully valid actions; default (1024, 1), tested best (448, 1); default / tested-best median = 1.09881x.

- "matrix_transpose" size 496: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.12805x.

- "matrix_transpose" size 784: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.08762x.

- "matrix_transpose" size 1016: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.07379x.

- "matrix_transpose" size 1232: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.07651x.

- "matrix_transpose" size 1680: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 1.06453x.

- "matrix_transpose" size 2024: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.05693x.

- "periodic_tile" size 240: 16 fully valid actions; default (1024, 1), tested best (320, 1); default / tested-best median = 1.08674x.

- "periodic_tile" size 496: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.37318x.

- "periodic_tile" size 784: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.49134x.

- "periodic_tile" size 1016: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.55318x.

- "periodic_tile" size 1232: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.46206x.

- "periodic_tile" size 1680: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.61865x.

- "periodic_tile" size 2024: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.63912x.

- "repeat_vector" size 240: 16 fully valid actions; default (1024, 1), tested best (1024, 1); default / tested-best median = 1x.

- "repeat_vector" size 496: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.20885x.

- "repeat_vector" size 784: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.26214x.

- "repeat_vector" size 1016: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.34224x.

- "repeat_vector" size 1232: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.36665x.

- "repeat_vector" size 1680: 16 fully valid actions; default (1024, 1), tested best (128, 1); default / tested-best median = 1.34471x.

- "repeat_vector" size 2024: 16 fully valid actions; default (1024, 1), tested best (320, 1); default / tested-best median = 1.31463x.

- "segmented_sum" size 240: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 3.23629x.

- "segmented_sum" size 496: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 7.28062x.

- "segmented_sum" size 784: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 12.55x.

- "segmented_sum" size 1016: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 20.894x.

- "segmented_sum" size 1232: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 17.8602x.

- "segmented_sum" size 1680: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 16.5916x.

- "segmented_sum" size 2024: 16 fully valid actions; default (1024, 1), tested best (64, 1); default / tested-best median = 18.3075x.

- "vector_add" size 240: 16 fully valid actions; default (1024, 1), tested best (448, 1); default / tested-best median = 1.0343x.

- "vector_add" size 496: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.1716x.

- "vector_add" size 784: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.02052x.

- "vector_add" size 1016: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.00531x.

- "vector_add" size 1232: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.00398x.

- "vector_add" size 1680: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.0034x.

- "vector_add" size 2024: 16 fully valid actions; default (1024, 1), tested best (256, 1); default / tested-best median = 1.00231x.
