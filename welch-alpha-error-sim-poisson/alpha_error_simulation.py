import os
import tomllib
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from joblib import Parallel, delayed
from tqdm import tqdm

def read_setting(setting_file_path: str) -> dict[str, Any]:
    """設定ファイルを読み込む"""
    with open(setting_file_path, mode="rb") as setting_file:
        settings: dict[str, Any] = tomllib.load(setting_file)
    return settings

def generate_2group_poisson_random_values(
    lam: float,
    sample_size_group1: int,
    sample_size_group2: int,
    n_simulation: int,
    seed,
) -> tuple:
    """指定のパラメータを持つポアソン乱数を2群分生成する"""
    rng = np.random.default_rng(seed)
    group1 = rng.poisson(lam=lam, size=(sample_size_group1, n_simulation))
    group2 = rng.poisson(lam=lam, size=(sample_size_group2, n_simulation))
    return group1, group2

def make_lambda_list_from_skerwness(low, high, num) -> np.ndarray:
    """設定ファイルから歪度が等間隔になるようにポアソンのパラメータのリストを作成する"""
    skewness = np.linspace(low, high, num=num, endpoint=True)
    lam = 1.0 / (skewness ** 2)
    return lam

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

def calc_alpha_error_and_interval(p_values: np.ndarray, alpha: float, ci_alpha: float, method: str = 'wilson') -> tuple:
    """αエラーとその信頼区間を算出する"""
    p_values = p_values[~np.isnan(p_values)]
    reject_count = np.sum(p_values < alpha)
    sample_size = p_values.shape[0]
    alpha_error = reject_count / sample_size
    low, high = proportion_confint(reject_count, sample_size, alpha=ci_alpha, method=method)
    return alpha_error, low, high, sample_size

def contingency_table(student_pvals, welch_pvals, alpha):
    """2つの検定の棄却に対する分割表の度数を算出する"""
    valid = ~np.isnan(student_pvals) & ~np.isnan(welch_pvals)
    rej_s = student_pvals[valid] < alpha
    rej_w = welch_pvals[valid] < alpha

    both_reject = int(np.sum(rej_s &  rej_w))
    student_only = int(np.sum(rej_s & ~rej_w))
    welch_only = int(np.sum(~rej_s &  rej_w))
    n_paired = int(valid.sum())
    return both_reject, student_only, welch_only, n_paired

def newcombe_paired_diff_ci(
        p1, l1, u1, p2, l2, u2,
        both_reject, student_only, welch_only, n
):
    """Newcombe型の対応のある比率の差の信頼区間。calc_alpha_error_and_intervalで使用された信頼水準を引き継ぐ。"""
    p1, l1, u1 = np.asarray(p1, float), np.asarray(l1, float), np.asarray(u1, float)
    p2, l2, u2 = np.asarray(p2, float), np.asarray(l2, float), np.asarray(u2, float)
    a = np.asarray(both_reject, float)
    b = np.asarray(student_only, float)
    c = np.asarray(welch_only, float)
    n = np.asarray(n, float)
    d = n - a - b - c

    diff = p1 - p2

    # 2x2表の phi係数(棄却の相関)。周辺度数が0なら相関補正なし(phi=0)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    safe = denom > 0
    phi = np.where(safe, (a * d - b * c) / np.sqrt(np.where(safe, denom, 1.0)), 0.0)

    # 既存Wilson端の距離を二乗して足し、phiで相関補正
    low = diff - np.sqrt(np.clip((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0, None))
    high = diff + np.sqrt(np.clip((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0, None))
    return diff, low, high

def run_one_cell(
    lam,
    sample_size_group1,
    sample_size_group2,
    significance_alpha,
    confidence_interval_alpha,
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
        group1, group2 = generate_2group_poisson_random_values(
            lam=lam,
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

    # αエラーとその信頼区間を算出
    student_alpha_error, student_ci_low, student_ci_high, student_valid_n = calc_alpha_error_and_interval(student_pvals, significance_alpha, confidence_interval_alpha)
    welch_alpha_error, welch_ci_low, welch_ci_high, welch_valid_n = calc_alpha_error_and_interval(welch_pvals, significance_alpha, confidence_interval_alpha)

    # p値の分位点を算出する
    student_p_quantiles = np.nanquantile(student_pvals, quantiles)
    welch_p_quantiles = np.nanquantile(welch_pvals, quantiles)

    # αエラーの差とその信頼区間を算出
    both_reject, student_only, welch_only, n_paired = contingency_table(student_pvals, welch_pvals, significance_alpha)
    diff, diff_low, diff_high = newcombe_paired_diff_ci(
        student_alpha_error, student_ci_low, student_ci_high,
        welch_alpha_error, welch_ci_low, welch_ci_high,
        both_reject, student_only, welch_only, n_paired
    )

    # 結果保存
    result = {
        'Lambda': lam, 'SAMPLE_SIZE_G1': sample_size_group1, 'SAMPLE_SIZE_G2': sample_size_group2,
        'student_alpha_error': student_alpha_error, 'welch_alpha_error': welch_alpha_error,
        'student_ci_low': student_ci_low, 'student_ci_high': student_ci_high,
        'welch_ci_low': welch_ci_low, 'welch_ci_high': welch_ci_high,
        'student_valid_n': student_valid_n, 'welch_valid_n': welch_valid_n,
        'alpha_error_diff': diff, 'alpha_error_diff_ci_low': diff_low,
        'alpha_error_diff_ci_high': diff_high, 'alpha_error_diff_valid_n': n_paired
    }

    for q, p_value in zip(quantiles, student_p_quantiles):
        result[f'student_p_value_quantile_{q}'] = p_value

    for q, p_value in zip(quantiles, welch_p_quantiles):
        result[f'welch_p_value_quantile_{q}'] = p_value

    return result

def main():
    # 設定ファイル読み込み
    settings = read_setting('./settings.toml')['alpha_error_simulation']
    SKEWNESS_RANGE = settings['skewness_range']
    N_SKEWNESS_POINT = settings['skewness_n_point']
    SAMPLE_SIZE_LIST_G1 = settings['group1_sample_size']
    SAMPLE_SIZE_RATIO_LOG_RANGE = settings['sample_size_log_range']
    N_SAMPLE_SIZE_POINT = settings['sample_size_n_point']
    SIGNIFICANCE_ALPHA = settings['significance_alpha']
    CONFIDENCE_INTERVAL_ALPHA = settings['confidence_interval_alpha']
    QUANTILES = settings['quantile']
    N_SIMULATION = settings['simulation_n_repete']
    BATCH_SIZE = settings['batch_size']
    PARENT_SEED = settings['parent_seed']
    N_JOBS = settings['n_jobs']

    # ポアソン分布のパラメータが動く値を設定
    LAMBDA_LIST = make_lambda_list_from_skerwness(SKEWNESS_RANGE[0], SKEWNESS_RANGE[-1], N_SKEWNESS_POINT)

    # 先に全タスク（パラメータ＋子シード）を逐次でリスト化
    tasks = []
    seed_seq = np.random.SeedSequence(PARENT_SEED)
    for SAMPLE_SIZE_G1 in SAMPLE_SIZE_LIST_G1:

        # サンプルサイズを動かす値を設定 (Group2のサンプルサイズ)
        SAMPLE_SIZE_LIST_G2 = make_sample_size_list(SAMPLE_SIZE_RATIO_LOG_RANGE[0], SAMPLE_SIZE_RATIO_LOG_RANGE[-1], N_SAMPLE_SIZE_POINT, SAMPLE_SIZE_G1)

        # タスクのリスト化
        for SAMPLE_SIZE_G2 in SAMPLE_SIZE_LIST_G2:
            for LAMBDA in LAMBDA_LIST:
                seed = seed_seq.spawn(1)[0]
                tasks.append((SAMPLE_SIZE_G1, SAMPLE_SIZE_G2, LAMBDA, seed))

    # 並列実行 (ジェネレータを作ってからtqdmでラップして実行)
    sim_generator = Parallel(n_jobs=N_JOBS, return_as='generator_unordered')(
        delayed(run_one_cell)(
            lam=poisson_lam,
            sample_size_group1=s1,
            sample_size_group2=s2,
            significance_alpha=SIGNIFICANCE_ALPHA,
            confidence_interval_alpha=CONFIDENCE_INTERVAL_ALPHA,
            quantiles=QUANTILES,
            n_simulation=N_SIMULATION,
            batch_size=BATCH_SIZE,
            seed=seed,
        )
        for (s1, s2, poisson_lam, seed) in tasks
    )
    results = list(tqdm(sim_generator, total=len(tasks)))

    # データフレームにして出力
    sim_result_data = pd.DataFrame(results)
    os.makedirs('output', exist_ok=True)
    sim_result_data.to_parquet('./output/alpha_error_sim_poisson_result.parquet', index=False)

if __name__ == '__main__':
    main()
    
