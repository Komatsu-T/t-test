import os
import tomllib
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from joblib import Parallel, delayed

def read_setting(setting_file_path: str) -> dict[str, Any]:
    """設定ファイルを読み込む"""
    with open(setting_file_path, mode="rb") as setting_file:
        settings: dict[str, Any] = tomllib.load(setting_file)
    return settings

def generate_2group_normal_random_values(
    loc_group1: float,
    loc_group2: float,
    sigma_group1: float,
    sigma_group2: float,
    sample_size_group1: int,
    sample_size_group2: int,
    n_simulation: int,
    seed,
) -> tuple:
    """指定の平均・標準偏差を持つ正規乱数を2群分生成する"""
    rng = np.random.default_rng(seed)
    group1 = rng.normal(loc=loc_group1, scale=sigma_group1, size=(sample_size_group1, n_simulation)).astype(np.float32)
    group2 = rng.normal(loc=loc_group2, scale=sigma_group2, size=(sample_size_group2, n_simulation)).astype(np.float32)
    return group1, group2

def make_sigma_list(low, high, num, group1_sigma) -> np.ndarray:
    """設定ファイルからsigmaのリストを作成する"""
    sigma_ratio = np.logspace(low, high,  num=num, endpoint=True, base=10)
    return sigma_ratio * group1_sigma

def make_sample_size_list(low, high, num, group1_sample_size) -> np.ndarray:
    """設定ファイルからsample sizeのリストを作成する"""
    sample_size_list = np.logspace(low, high,  num=num, endpoint=True, base=10.0)
    return np.unique((sample_size_list * group1_sample_size).astype(int))

def t_test(group1: np.ndarray, group2: np.ndarray, method: str) -> tuple:
    """t検定を実行する"""
    if method == 'student':
        student_ttest = stats.ttest_ind(group1, group2, equal_var=True)
        return student_ttest
    elif method == 'welch':
        welch_ttest = stats.ttest_ind(group1, group2, equal_var=False)
        return welch_ttest
    else:
        raise ValueError(f"unknown method: {method!r}")

def calc_alpha_error_and_interval(p_values: np.ndarray, alpha: float = 0.05, ci_alpha: float = 0.05, method: str = 'wilson') -> tuple:
    """αエラーとその信頼区間を算出する"""
    p_values = p_values[~np.isnan(p_values)]
    reject_count = np.sum(p_values < alpha)
    sample_size = p_values.shape[0]
    alpha_error = reject_count / sample_size
    low, high = proportion_confint(reject_count, sample_size, alpha=ci_alpha, method=method)
    return alpha_error, low, high, sample_size

def run_one_cell(
    loc_group1,
    loc_group2,
    sigma_group1,
    sigma_group2,
    sample_size_group1,
    sample_size_group2,
    quantiles,
    n_simulation,
    batch_size,
    seed,
):
    """バッチに分けてシミュレーションを実行する"""

    # バッチ数とシードの設定
    n_batch = int(np.ceil(n_simulation / batch_size))
    batch_seeds = seed.spawn(n_batch)

    # バッチごとに計算を実行して統合
    student_pvals = []
    welch_pvals = []
    remaining = n_simulation
    for bi in range(n_batch):

        # 乱数生成
        n_this = min(batch_size, remaining)
        group1, group2 = generate_2group_normal_random_values(
            loc_group1=loc_group1,
            loc_group2=loc_group2,
            sigma_group1=sigma_group1,
            sigma_group2=sigma_group2,
            sample_size_group1=sample_size_group1,
            sample_size_group2=sample_size_group2,
            n_simulation=n_this,
            seed=batch_seeds[bi]
        )
        remaining -= n_this

        # t-検定
        student_ttest_result = t_test(group1, group2, method='student')
        welch_ttest_result = t_test(group1, group2, method='welch')

        # p値を保存
        student_pvals.append(student_ttest_result.pvalue)
        welch_pvals.append(welch_ttest_result.pvalue)

    # p値のリストを全バッチ結合
    student_pvals = np.concatenate(student_pvals)
    welch_pvals = np.concatenate(welch_pvals)

    # αエラーと信頼区間を算出
    student_alpha_error, student_ci_low, student_ci_high, student_valid_n = calc_alpha_error_and_interval(student_pvals)
    welch_alpha_error, welch_ci_low, welch_ci_high, welch_valid_n = calc_alpha_error_and_interval(welch_pvals)

    # p値の分位点を算出する
    student_p_quantiles = np.nanquantile(student_pvals, quantiles)
    welch_p_quantiles = np.nanquantile(welch_pvals, quantiles)

    # 結果保存
    result = {
        'MU_G1': loc_group1, 'SIGMA_G1': sigma_group1, 'SAMPLE_SIZE_G1': sample_size_group1,
        'MU_G2': loc_group2, 'SIGMA_G2': sigma_group2, 'SAMPLE_SIZE_G2': sample_size_group2,
        'student_alpha_error': student_alpha_error, 'welch_alpha_error': welch_alpha_error,
        'student_ci_low': student_ci_low, 'student_ci_high': student_ci_high,
        'welch_ci_low': welch_ci_low, 'welch_ci_high': welch_ci_high,
        'student_valid_n': student_valid_n, 'welch_valid_n': welch_valid_n
    }

    for q, p_value in zip(quantiles, student_p_quantiles):
        result[f'student_p_value_quantile_{q}'] = p_value

    for q, p_value in zip(quantiles, welch_p_quantiles):
        result[f'welch_p_value_quantile_{q}'] = p_value

    return result

def main():
    # 設定ファイル読み込み
    settings = read_setting('./settings.toml')
    MU_G1 = settings['alpha_error_simulation']['group1_mu']
    SIGMA_G1 = settings['alpha_error_simulation']['group1_sigma']
    SAMPLE_SIZE_LIST_G1 = settings['alpha_error_simulation']['group1_sample_size']
    SIGMA_RATIO_LOG_RANGE = settings['alpha_error_simulation']['sigma_ratio_log_range']
    SAMPLE_SIZE_RATIO_LOG_RANGE = settings['alpha_error_simulation']['sample_size_log_range']
    N_SIGMA_POINT = settings['alpha_error_simulation']['sigma_ratio_n_point']
    N_SAPLE_SIZE_POINT = settings['alpha_error_simulation']['sample_size_n_point']
    QUANTILES = settings['alpha_error_simulation']['quantile']
    N_SIMULATION = settings['alpha_error_simulation']['simulation_n_repete']
    BATCH_SIZE = settings['alpha_error_simulation']['batch_size']
    PARENT_SEED = settings['alpha_error_simulation']['parent_seed']
    N_JOBS = settings['alpha_error_simulation']['n_jobs']

    # 標準偏差を動かす値を設定 (Group2の標準偏差)
    SIGMA_LIST_G2 = make_sigma_list(SIGMA_RATIO_LOG_RANGE[0], SIGMA_RATIO_LOG_RANGE[-1], N_SIGMA_POINT, SIGMA_G1)

    # 先に全タスク（パラメータ＋子シード）を逐次でリスト化
    tasks = []
    seed_seq = np.random.SeedSequence(PARENT_SEED)
    for SAMPLE_SIZE_G1 in SAMPLE_SIZE_LIST_G1:

        # サンプルサイズを動かす値を設定 (Group2の標準偏差)
        SAMPLE_SIZE_LIST_G2 = make_sample_size_list(SAMPLE_SIZE_RATIO_LOG_RANGE[0], SAMPLE_SIZE_RATIO_LOG_RANGE[-1], N_SAPLE_SIZE_POINT, SAMPLE_SIZE_G1)

        # タスクのリスト化
        for SAMPLE_SIZE_G2 in SAMPLE_SIZE_LIST_G2:
            for SIGMA_G2 in SIGMA_LIST_G2:
                seed = seed_seq.spawn(1)[0]
                tasks.append((SAMPLE_SIZE_G1, SAMPLE_SIZE_G2, SIGMA_G2, seed))

    # 並列実行
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_one_cell)(
            loc_group1=MU_G1, 
            loc_group2=MU_G1,
            sigma_group1=SIGMA_G1, 
            sigma_group2=sigma_g2,
            sample_size_group1=s1, 
            sample_size_group2=s2,
            quantiles=QUANTILES, 
            n_simulation=N_SIMULATION,
            batch_size=BATCH_SIZE, 
            seed=seed,
        )
        for (s1, s2, sigma_g2, seed) in tasks
    )

    # データフレームにして出力
    sim_result_data = pd.DataFrame(results)
    os.makedirs('output', exist_ok=True)
    sim_result_data.to_parquet('./output/alpha_error_sim_result.parquet', index=False)

if __name__ == '__main__':
    main()
    