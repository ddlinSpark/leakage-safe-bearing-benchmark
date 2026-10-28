# Experiment Report

## Scientific scope

These analyses test implementation sanity, preprocessing dependence, label boundaries, baseline fairness, and claim robustness. They are not used to make the primary split easier or to replace the primary result with a favorable test score.

## Domain-shift diagnostics

| fold | metric | source_mean | target_mean | standardized_mean_difference |
| --- | --- | --- | --- | --- |
| 0 | rms | 1.7682 | 1.6711 | -3.3506 |
| 0 | kurtosis | -1.4983 | -1.4907 | 0.7048 |
| 0 | crest_factor | 1.6492 | 1.6465 | -0.0566 |
| 0 | spectral_centroid_hz | 125.6480 | 87.0916 | -10.0941 |
| 0 | spectral_entropy | 0.0809 | 0.1408 | 9.3243 |
| 1 | rms | 1.7211 | 0.9475 | -14.2027 |
| 1 | kurtosis | -1.4946 | -1.4918 | 0.2103 |
| 1 | crest_factor | 1.6508 | 1.8384 | 3.3777 |
| 1 | spectral_centroid_hz | 107.9890 | 167.7378 | 3.5033 |
| 1 | spectral_entropy | 0.1110 | 0.0835 | -0.9573 |
| 2 | rms | 1.3011 | 1.7789 | 1.3438 |
| 2 | kurtosis | -1.4885 | -1.4999 | -0.8536 |
| 2 | crest_factor | 1.7831 | 1.6305 | -1.0700 |
| 2 | spectral_centroid_hz | 137.3516 | 122.5156 | -0.3219 |
| 2 | spectral_entropy | 0.1139 | 0.0806 | -1.2689 |
| 3 | rms | 1.3488 | 1.7595 | 1.0311 |
| 3 | kurtosis | -1.4924 | -1.4968 | -0.3981 |
| 3 | crest_factor | 1.7816 | 1.6441 | -0.9554 |
| 3 | spectral_centroid_hz | 155.3940 | 122.5671 | -1.0743 |
| 3 | spectral_entropy | 0.0832 | 0.0807 | -0.4504 |

The raw amplitudes and spectra change with the held-out domains, and their directions differ across folds. The evidence supports a joint distributional shift, not a single-cause explanation.

## Label granularity

| model_label | granularity | macro_f1_mean | macro_f1_sd | accuracy_mean |
| --- | --- | --- | --- | --- |
| CWT 2D-CNN | binary | 0.6591 | 0.1996 | 0.7349 |
| CWT 2D-CNN | four_class | 0.2913 | 0.0857 | 0.3992 |
| CWT 2D-CNN | three_class | 0.3955 | 0.1330 | 0.5552 |
| CWT-GRU | binary | 0.6070 | 0.1963 | 0.6817 |
| CWT-GRU | four_class | 0.2796 | 0.1067 | 0.3835 |
| CWT-GRU | three_class | 0.3687 | 0.1261 | 0.5151 |
| CWT-GRU without SE | binary | 0.6690 | 0.2193 | 0.7403 |
| CWT-GRU without SE | four_class | 0.3067 | 0.1032 | 0.4083 |
| CWT-GRU without SE | three_class | 0.3989 | 0.1388 | 0.5466 |
| CWT-GRU + self-training | binary | 0.6009 | 0.2034 | 0.6770 |
| CWT-GRU + self-training | four_class | 0.2898 | 0.1156 | 0.3871 |
| CWT-GRU + self-training | three_class | 0.3628 | 0.1192 | 0.5002 |
| GRU only | binary | 0.4286 | 0.0000 | 0.7500 |
| GRU only | four_class | 0.2215 | 0.0303 | 0.2421 |
| GRU only | three_class | 0.2296 | 0.0126 | 0.4970 |
| Raw 1D-CNN | binary | 0.7013 | 0.2031 | 0.7877 |
| Raw 1D-CNN | four_class | 0.2892 | 0.1039 | 0.3917 |
| Raw 1D-CNN | three_class | 0.4100 | 0.1215 | 0.5591 |

The coarse views use the same fitted four-class outputs. Higher binary healthy-versus-fault performance indicates that part of the failure is fine-grained fault localization rather than complete loss of anomaly detection. The primary task remains four-class.

## CWT preprocessing controls

| variant | runs | macro_f1_mean | macro_f1_sd | accuracy_mean | accuracy_sd | mean_best_epoch | mean_validation_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 0.2913 | 0.0857 | 0.3992 | 0.0812 | 7.8500 | 0.8077 |
| log_magnitude | 20 | 0.3108 | 0.0910 | 0.4181 | 0.0833 | 7.2000 | 0.7890 |
| no_max_norm | 20 | 0.2324 | 0.1156 | 0.3349 | 0.1045 | 4.9500 | 0.8187 |
| real_imag | 20 | 0.2976 | 0.0827 | 0.3893 | 0.0725 | 8.0000 | 0.8120 |
| resolution_128 | 20 | 0.3036 | 0.1288 | 0.3992 | 0.1184 | 7.6500 | 0.8276 |
| scales_32 | 20 | 0.3235 | 0.0827 | 0.4222 | 0.0818 | 8.0500 | 0.8046 |
| scales_64 | 20 | 0.2937 | 0.0780 | 0.3984 | 0.0789 | 8.6500 | 0.8180 |

| variant | mean_delta | hierarchical_ci_low | hierarchical_ci_high | fold_wins | fold_ties | fold_losses |
| --- | --- | --- | --- | --- | --- | --- |
| no_max_norm | -0.0589 | -0.1265 | 0.0242 | 1 | 0 | 3 |
| log_magnitude | 0.0195 | -0.0234 | 0.0649 | 2 | 0 | 2 |
| resolution_128 | 0.0123 | -0.0661 | 0.0819 | 3 | 0 | 1 |
| scales_32 | 0.0322 | 0.0003 | 0.0689 | 4 | 0 | 0 |
| scales_64 | 0.0024 | -0.0224 | 0.0266 | 2 | 0 | 2 |
| real_imag | 0.0063 | -0.0557 | 0.0608 | 2 | 0 | 2 |
| no_max_norm | -0.0568 | -0.1617 | 0.0773 | 1 | 0 | 3 |
| log_magnitude | 0.0215 | -0.0763 | 0.1607 | 1 | 0 | 3 |
| resolution_128 | 0.0144 | -0.1173 | 0.1905 | 1 | 0 | 3 |
| scales_32 | 0.0342 | -0.0568 | 0.1637 | 1 | 0 | 3 |
| scales_64 | 0.0045 | -0.0872 | 0.1399 | 1 | 0 | 3 |
| real_imag | 0.0084 | -0.0851 | 0.1134 | 2 | 0 | 2 |

## Normalization controls

| method | normalization | runs | test_macro_f1_mean | test_macro_f1_sd | validation_macro_f1_mean | validation_macro_f1_sd | validation_test_gap_mean | validation_test_gap_sd | relative_drop_mean | relative_drop_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cwt_cnn | A_window_zscore | 20 | 0.2913 | 0.0857 | 0.8077 | 0.1119 | 0.5164 | 0.1746 | 0.6242 | 0.1486 |
| cwt_cnn | B_training_statistics | 20 | 0.2877 | 0.0884 | 0.8055 | 0.1158 | 0.5177 | 0.1844 | 0.6251 | 0.1578 |
| cwt_gru | A_window_zscore | 20 | 0.2796 | 0.1067 | 0.7975 | 0.1155 | 0.5178 | 0.2023 | 0.6303 | 0.1722 |
| cwt_gru | B_training_statistics | 20 | 0.2996 | 0.0858 | 0.8097 | 0.1265 | 0.5102 | 0.1928 | 0.6107 | 0.1547 |
| raw_cnn | A_window_zscore | 20 | 0.2892 | 0.1039 | 0.8532 | 0.1526 | 0.5640 | 0.1279 | 0.6641 | 0.1190 |
| raw_cnn | B_training_statistics | 20 | 0.1196 | 0.0364 | 0.5273 | 0.2042 | 0.4076 | 0.2035 | 0.7464 | 0.1010 |

| method | model_label | n_pairs | mean_delta_normalization_b_minus_a | hierarchical_ci_low | hierarchical_ci_high | fold_wins_b | fold_losses_b |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_cnn | Raw 1D-CNN | 20 | -0.1696 | -0.2315 | -0.0944 | 0 | 4 |
| cwt_cnn | CWT 2D-CNN | 20 | -0.0036 | -0.0155 | 0.0077 | 1 | 3 |
| cwt_gru | CWT-GRU | 20 | 0.0199 | -0.0172 | 0.0612 | 2 | 2 |

## Coverage-controlled self-training

| budget | condition | runs | macro_f1_mean | macro_f1_sd | accuracy_mean | accuracy_sd |
| --- | --- | --- | --- | --- | --- | --- |
| 50_percent | coverage_controlled_0.50 | 20 | 0.2781 | 0.0986 | 0.3810 | 0.0908 |
| 50_percent | fixed_threshold_0.60 | 20 | 0.2830 | 0.1257 | 0.3944 | 0.1102 |
| 50_percent | supervised | 20 | 0.2440 | 0.0982 | 0.3512 | 0.0900 |
| full_balanced | coverage_controlled_0.50 | 20 | 0.2794 | 0.0752 | 0.3740 | 0.0718 |
| full_balanced | fixed_threshold_0.60 | 20 | 0.2898 | 0.1156 | 0.3871 | 0.0989 |
| full_balanced | supervised | 20 | 0.2796 | 0.1067 | 0.3835 | 0.0922 |

| budget | comparison | left | right | n_pairs | mean_delta | hierarchical_ci_low | hierarchical_ci_high | fold_signflip_p | fold_wins | fold_ties | fold_losses | fold_deltas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50_percent | coverage_controlled_0.50 minus supervised | supervised | coverage_controlled_0.50 | 20 | 0.0342 | -0.0205 | 0.0958 | 0.3750 | 3 | 0 | 1 | -0.025623;0.075444;0.019087;0.067722 |
| 50_percent | coverage_controlled_0.50 minus fixed_threshold_0.60 | fixed_threshold_0.60 | coverage_controlled_0.50 | 20 | -0.0048 | -0.0836 | 0.0844 | 0.8750 | 1 | 0 | 3 | -0.007911;-0.062069;-0.033840;0.084460 |
| full_balanced | coverage_controlled_0.50 minus supervised | supervised | coverage_controlled_0.50 | 20 | -0.0002 | -0.0460 | 0.0441 | 1.0000 | 2 | 0 | 2 | 0.019667;-0.044952;0.048993;-0.024702 |
| full_balanced | coverage_controlled_0.50 minus fixed_threshold_0.60 | fixed_threshold_0.60 | coverage_controlled_0.50 | 20 | -0.0104 | -0.0775 | 0.0500 | 1.0000 | 3 | 0 | 1 | 0.001401;-0.095555;0.050098;0.002412 |

## Baseline fairness

| method | model_label | candidate | runs | parameter_count | validation_macro_f1_mean | validation_macro_f1_sd | test_macro_f1_mean | test_macro_f1_sd | mean_best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cwt_cnn | CWT 2D-CNN | lr0.001_wd0.0001 | 20 | 884564.0000 | 0.8124 | 0.1225 | 0.2733 | 0.1071 | 7.5000 |
| cwt_cnn | CWT 2D-CNN | lr0.002_wd0.0001 | 20 | 884564.0000 | 0.8077 | 0.1119 | 0.2913 | 0.0857 | 7.8500 |
| drsn | DRSN adaptation | lr0.001_wd0.0001 | 20 | 233040.0000 | 0.7767 | 0.1437 | 0.1809 | 0.1173 | 7.4000 |
| drsn | DRSN adaptation | lr0.002_wd0.0001 | 20 | 233040.0000 | 0.7902 | 0.1583 | 0.1485 | 0.0843 | 6.3000 |
| raw_cnn | Raw 1D-CNN | lr0.001_wd0.0001 | 20 | 244740.0000 | 0.8415 | 0.1566 | 0.3024 | 0.0616 | 8.2000 |
| raw_cnn | Raw 1D-CNN | lr0.002_wd0.0001 | 20 | 244740.0000 | 0.8532 | 0.1526 | 0.2892 | 0.1039 | 7.2000 |
| tfnet | TFN adaptation | lr0.001_wd0.0001 | 20 | 184900.0000 | 0.7081 | 0.2580 | 0.1894 | 0.0920 | 5.7000 |
| tfnet | TFN adaptation | lr0.002_wd0.0001 | 20 | 184900.0000 | 0.6889 | 0.2640 | 0.1750 | 0.0775 | 5.1500 |

| method | model_label | default_mean_macro_f1 | validation_selected_mean_macro_f1 | mean_tuned_minus_default | hierarchical_ci_low | hierarchical_ci_high | fold_wins | fold_ties | fold_losses | selected_configurations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cwt_cnn | CWT 2D-CNN | 0.2913 | 0.2829 | -0.0084 | -0.0313 | 0.0022 | 0 | 3 | 1 | 0:lr0.001_wd0.0001;1:lr0.002_wd0.0001;2:lr0.002_wd0.0001;3:lr0.002_wd0.0001 |
| drsn | DRSN adaptation | 0.1485 | 0.1540 | 0.0055 | -0.0006 | 0.0209 | 1 | 3 | 0 | 0:lr0.002_wd0.0001;1:lr0.001_wd0.0001;2:lr0.002_wd0.0001;3:lr0.002_wd0.0001 |
| raw_cnn | Raw 1D-CNN | 0.2892 | 0.2758 | -0.0135 | -0.0398 | 0.0074 | 0 | 2 | 2 | 0:lr0.001_wd0.0001;1:lr0.002_wd0.0001;2:lr0.001_wd0.0001;3:lr0.002_wd0.0001 |
| tfnet | TFN adaptation | 0.1750 | 0.1894 | 0.0143 | -0.0444 | 0.0827 | 2 | 0 | 2 | 0:lr0.001_wd0.0001;1:lr0.001_wd0.0001;2:lr0.001_wd0.0001;3:lr0.001_wd0.0001 |

## Training opportunity

| method | model_label | validation_delta_mean | validation_delta_sd | validation_ci_low | validation_ci_high | test_delta_mean | test_delta_sd | test_ci_low | test_ci_high | budget_30_best_epoch_mean | budget_30_best_epoch_max | budget_30_cap_selected_runs | validation_improved_runs | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_cnn | Raw 1D-CNN | -0.0138 | 0.0581 | -0.0608 | 0.0115 | -0.0162 | 0.1120 | -0.0816 | 0.0639 | 6.8000 | 17 | 0 | 12 | 20 |
| cwt_cnn | CWT 2D-CNN | 0.0391 | 0.0354 | 0.0188 | 0.0622 | 0.0139 | 0.0669 | -0.0157 | 0.0445 | 8.5000 | 18 | 0 | 17 | 20 |
| cwt_gru_no_se | CWT-GRU without SE | 0.0065 | 0.0267 | -0.0076 | 0.0185 | -0.0443 | 0.0830 | -0.0922 | -0.0034 | 7.8500 | 12 | 0 | 12 | 20 |
| cwt_gru | CWT-GRU | 0.0344 | 0.0661 | -0.0036 | 0.0837 | 0.0118 | 0.0564 | -0.0170 | 0.0390 | 8.5500 | 19 | 0 | 12 | 20 |

## Calibration

| model | runs | temperature_mean | temperature_sd | temperature_min | temperature_max | lower_bound_runs | upper_bound_runs | test_ece_before | test_ece_after | test_nll_before | test_nll_after | test_brier_before | test_brier_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cwt_cnn | 20 | 1.9205 | 0.6766 | 0.8578 | 2.9589 | 0 | 0 | 0.4870 | 0.3755 | 3.6152 | 2.4978 | 0.2597 | 0.2285 |
| cwt_gru | 20 | 2.6173 | 1.0974 | 1.2257 | 4.5706 | 0 | 0 | 0.5426 | 0.3912 | 4.5282 | 2.3038 | 0.2799 | 0.2377 |
| cwt_gru_no_se | 20 | 2.6193 | 1.1613 | 1.3297 | 4.6096 | 0 | 0 | 0.5056 | 0.3516 | 4.2318 | 2.3048 | 0.2660 | 0.2249 |
| cwt_gru_selftrain | 20 | 2.3942 | 1.2441 | 0.8376 | 4.2026 | 0 | 0 | 0.5224 | 0.3968 | 3.9687 | 2.4822 | 0.2707 | 0.2382 |
| cwt_gru_temp | 20 | 2.6173 | 1.0974 | 1.2257 | 4.5706 | 0 | 0 | 0.5426 | 0.3912 | 4.5282 | 2.3038 | 0.2799 | 0.2377 |
| gru_only | 20 | 1.4407 | 1.0957 | 0.4244 | 5.1535 | 0 | 0 | 0.0582 | 0.0579 | 1.3990 | 1.4020 | 0.1891 | 0.1894 |
| raw_cnn | 20 | 0.9539 | 0.8358 | 0.0500 | 2.5092 | 3 | 0 | 0.3882 | 0.4238 | 2.5992 | 6.2369 | 0.2336 | 0.2481 |

Per-run deterministic reproduction checks: 560 metric comparisons, 560 verified with no observed test-metric difference.

## Per-class threshold diagnostics

| threshold | runs | selected_fraction_mean | pseudo_accuracy_mean | coverage_class_0 | coverage_class_1 | coverage_class_2 | coverage_class_3 | pseudo_accuracy_class_0 | pseudo_accuracy_class_1 | pseudo_accuracy_class_2 | pseudo_accuracy_class_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3000 | 20.0000 | 0.7762 | 0.9268 | 0.6197 | 0.8983 | 0.8478 | 0.8987 | 0.9940 | 0.9505 | 0.9331 | 0.8306 |
| 0.4000 | 20.0000 | 0.7699 | 0.9299 | 0.6197 | 0.8664 | 0.8432 | 0.8913 | 0.9940 | 0.9555 | 0.9353 | 0.8360 |
| 0.5000 | 20.0000 | 0.7423 | 0.9422 | 0.6139 | 0.7939 | 0.8097 | 0.8550 | 0.9955 | 0.9729 | 0.9462 | 0.8574 |
| 0.6000 | 20.0000 | 0.6888 | 0.9548 | 0.5957 | 0.7035 | 0.7420 | 0.7744 | 0.9956 | 0.9840 | 0.9633 | 0.8825 |
| 0.7000 | 20.0000 | 0.5888 | 0.9663 | 0.5330 | 0.6038 | 0.6102 | 0.6486 | 0.9966 | 0.9878 | 0.9811 | 0.8939 |

## Feature space

| fold | seed | train_windows | validation_windows | test_windows | pca_explained_variance | domain_silhouette | test_class_silhouette | all_bearing_silhouette | all_condition_silhouette | nearest_same_class_fraction | nearest_same_bearing_fraction | nearest_same_condition_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 42.0000 | 1488.0000 | 714.0000 | 248.0000 | 0.9967 | -0.0597 | 0.1563 | 0.1411 | -0.0588 | 0.9353 | 0.7940 | 0.7105 |
| 1.0000 | 42.0000 | 1489.0000 | 744.0000 | 248.0000 | 0.9951 | -0.0319 | 0.1487 | 0.0635 | -0.0354 | 0.9100 | 0.7142 | 0.7090 |
| 2.0000 | 42.0000 | 1459.0000 | 744.0000 | 248.0000 | 0.9947 | -0.0982 | 0.2145 | 0.0947 | -0.0945 | 0.9250 | 0.7787 | 0.7490 |
| 3.0000 | 42.0000 | 1458.0000 | 745.0000 | 248.0000 | 0.9957 | -0.0638 | 0.4108 | 0.0606 | -0.0637 | 0.9217 | 0.7554 | 0.7073 |

## Engineering overfit check

| method | fold | seed | subset_fraction_requested | subset_windows | epochs_requested | epochs_ran | best_epoch | subset_macro_f1 | reached_0_95 | reached_0_99 | final_train_loss | final_train_macro_f1 | test_data_used | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_cnn | 0 | 42 | 0.0400 | 40 | 80 | 80 | 23 | 1.0000 | True | True | 0.0354 | 1.0000 | False | engineering sanity check only |
| cwt_cnn | 0 | 42 | 0.0400 | 40 | 80 | 80 | 23 | 1.0000 | True | True | 0.0033 | 1.0000 | False | engineering sanity check only |
| cwt_gru | 0 | 42 | 0.0400 | 40 | 80 | 80 | 34 | 1.0000 | True | True | 0.0004 | 1.0000 | False | engineering sanity check only |

## Sensitivity matrix completeness

| result_family | path | rows | expected_rows | methods | duplicate_experiment_keys | statuses | git_dirty_rows | status | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coverage_self_training | results/self_training_sensitivity/raw_runs.csv | 40 | 40 | cwt_gru_selftrain_coverage | 0 | success | 40 | VERIFIED |  |
| normalization | results/normalization_sensitivity/raw_runs.csv | 60 | 60 | cwt_cnn;cwt_gru;raw_cnn | 0 | success | 60 | VERIFIED |  |
| baseline_tuning | results/baseline_tuning/raw_runs.csv | 80 | 80 | cwt_cnn;drsn;raw_cnn;tfnet | 0 | success | 80 | VERIFIED |  |
| training_budget | results/training_budget_30/raw_runs.csv | 80 | 80 | cwt_cnn;cwt_gru;cwt_gru_no_se;raw_cnn | 0 | success | 80 | VERIFIED |  |
| calibration_recheck | results/calibration_recheck/raw_runs.csv | 80 | 80 | cwt_cnn;cwt_gru;cwt_gru_selftrain;raw_cnn | 0 | success | 80 | VERIFIED |  |
| cwt_no_max_norm | results/cwt_sensitivity/no_max_norm/raw_runs.csv | 20 | 20 | cwt_cnn | 0 | success | 20 | VERIFIED |  |
| cwt_log_magnitude | results/cwt_sensitivity/log_magnitude/raw_runs.csv | 20 | 20 | cwt_cnn | 0 | success | 20 | VERIFIED |  |
| cwt_resolution_128 | results/cwt_sensitivity/resolution_128/raw_runs.csv | 20 | 20 | cwt_cnn | 0 | success | 20 | VERIFIED |  |
| cwt_scales_32 | results/cwt_sensitivity/scales_32/raw_runs.csv | 20 | 20 | cwt_cnn | 0 | success | 20 | VERIFIED |  |
| cwt_scales_64 | results/cwt_sensitivity/scales_64/raw_runs.csv | 20 | 20 | cwt_cnn | 0 | success | 20 | VERIFIED |  |
| cwt_real_imag | results/cwt_sensitivity/real_imag/raw_runs.csv | 20 | 20 | cwt_cnn | 0 | success | 20 | VERIFIED |  |
| threshold_per_class_diagnostics | results/threshold_per_class_diagnostics.csv | 100 | 100 | cwt_gru_selftrain | 0 | success | 0 | VERIFIED |  |

## Claim boundary

The results support configuration-specific conclusions. They do not establish universal failure of CWT, recurrent networks, calibration, self-training, DRSN, TFN, or any original study. The primary evidence remains the four class Paderborn benchmark with four independent operating-condition folds.
