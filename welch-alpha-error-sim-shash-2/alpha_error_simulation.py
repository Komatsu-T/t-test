import os
import tomllib
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from joblib import Parallel, delayed
from tqdm import tqdm

@dataclass(frozen=True)
class CommonSimConfig:
    sigma_g1: float
    sigma_ratio_log_range: list
    n_sigma_point: int
    total_sample_size: list
    min_sample_size_prop: float
    target_skew_exkurt: list
    significance_alpha: float
    confidence_interval_alpha: float
    quantiles: list
    n_simulation: int
    batch_size: int

@dataclass(frozen=True)
class CellConfig:
    target_skewness: float
    target_excess_kurtosis: float
    actual_skewness: float
    actual_excess_kurtosis: float
    eps: float
    delta: float
    sigma_group1: float
    sigma_group2: float
    sample_size_group1: int
    sample_size_group2: int

# sinh-arcsinh分布の期待値・分散を計算するための関数
_GX, _GW = np.polynomial.hermite.hermgauss(200)
_GZ = np.sqrt(2.0) * _GX
_GWN = _GW / np.sqrt(np.pi)
_GA = np.arcsinh(_GZ)

def _raw_moments(eps, delta):
    y = np.sinh((_GA + eps) / delta)
    return [np.sum(_GWN * y ** k) for k in range(1, 5)]

def _mean_sd(eps, delta):
    m = _raw_moments(eps, delta)
    return m[0], np.sqrt(m[1] - m[0] ** 2)

# シミュレーション用関数
def read_setting(setting_file_path: str):
    """設定ファイルを読み込む"""
    with open(setting_file_path, mode="rb") as setting_file:
        settings = tomllib.load(setting_file)
    return settings

def read_shash_dist_parameter(parameter_file_path):
    """sinh-arcsinh分布のパラメータを読み込む"""
    data = pd.read_parquet(parameter_file_path)
    data = data[(data['converged']) & (data['target_skewness']>=0)].copy()
    data = data[['target_skewness', 'target_excess_kurtosis', 'eps', 'delta']].copy()
    return data

def make_sigma_list(low, high, num, group1_sigma) -> np.ndarray:
    """設定ファイルからsigmaのリストを作成する"""
    sigma_ratio = np.logspace(low, high,  num=num, endpoint=True, base=10)
    return sigma_ratio * group1_sigma

def make_sample_size_list(total_sample_size, min_size_prop):
    sample_size_min = int(total_sample_size * min_size_prop)
    sample_size_max = total_sample_size - sample_size_min
    group1_sample_size = np.arange(sample_size_min, sample_size_max+1, 1)
    group2_sample_size = total_sample_size - group1_sample_size
    return group1_sample_size, group2_sample_size

def sinh_arcsinh_transform(z, eps, delta):
    """Y = sinh((asinh(z) + eps) / delta)を計算する"""
    return np.sinh((np.arcsinh(z) + eps) / delta)

def generate_2group_shash_random_values(
    eps,
    delta,
    sigma_group1,
    sigma_group2,
    sample_size_group1,
    sample_size_group2,
    n_simulation,
    seed
):
    """指定のパラメータをもつsinh-arcsinh分布から乱数を2群分生成する"""
    # 標準正規分布から乱数生成
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(size=(sample_size_group1, n_simulation), dtype=np.float32)
    z2 = rng.standard_normal(size=(sample_size_group2, n_simulation), dtype=np.float32)

    # 標準化用の期待値・分散を算出
    mu, sigma = _mean_sd(eps, delta)

    # sinh-arcsinh変換 (標準化済み)
    group1 = (sinh_arcsinh_transform(z1, eps, delta) - mu) / sigma
    group2 = (sinh_arcsinh_transform(z2, eps, delta) - mu) / sigma

    # 指定の分散の分布に変換
    group1 *= sigma_group1
    group2 *= sigma_group2
    return group1, group2

def t_test(group1, group2, method):
    """t検定を実行する"""
    if method not in ('student', 'welch'):
        raise ValueError(f"unknown method: {method!r}")
    return stats.ttest_ind(group1, group2, equal_var=(method=='student'))

def calc_alpha_error_and_interval(p_values, alpha, ci_alpha, method = 'wilson'):
    """αエラーとその信頼区間を算出する"""
    p_values = p_values[~np.isnan(p_values)]
    reject_count = np.sum(p_values < alpha)
    sample_size = p_values.shape[0]
    alpha_error = reject_count / sample_size
    low, high = proportion_confint(reject_count, sample_size, alpha=ci_alpha, method=method)
    return alpha_error, low, high, sample_size

def run_one_cell(cell: CellConfig, config: CommonSimConfig, seed: np.random.SeedSequence):
    """1つのセルのシミュレーションをバッチに分けて実行する"""

    # バッチ数とシードの設定
    n_batch = int(np.ceil(config.n_simulation / config.batch_size))
    batch_seeds = seed.spawn(n_batch)

    # バッチごとに計算を実行して統合
    student_pvals = []
    welch_pvals = []
    remaining = config.n_simulation
    for batch_seed in batch_seeds:
        # 乱数生成
        n_this = min(config.batch_size, remaining)
        group1, group2 = generate_2group_shash_random_values(
            eps=cell.eps,
            delta=cell.delta,
            sigma_group1=cell.sigma_group1,
            sigma_group2=cell.sigma_group2,
            sample_size_group1=cell.sample_size_group1,
            sample_size_group2=cell.sample_size_group2,
            n_simulation=n_this,
            seed=batch_seed
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
    student_alpha_error, student_ci_low, student_ci_high, student_valid_n = calc_alpha_error_and_interval(
        student_pvals, config.significance_alpha, config.confidence_interval_alpha
    )
    welch_alpha_error, welch_ci_low, welch_ci_high, welch_valid_n = calc_alpha_error_and_interval(
        welch_pvals, config.significance_alpha, config.confidence_interval_alpha
    )

    # p値の分位点を算出する
    student_p_quantiles = np.nanquantile(student_pvals, config.quantiles)
    welch_p_quantiles = np.nanquantile(welch_pvals, config.quantiles)

    # 結果保存
    result = asdict(cell)
    result['student_alpha_error'] = student_alpha_error
    result['welch_alpha_error'] = welch_alpha_error
    result['student_ci_low'] = student_ci_low
    result['student_ci_high'] = student_ci_high
    result['welch_ci_low'] = welch_ci_low
    result['welch_ci_high'] = welch_ci_high
    result['student_valid_n'] = student_valid_n
    result['welch_valid_n'] = welch_valid_n

    for q, p_value in zip(config.quantiles, student_p_quantiles):
        result[f'student_p_value_quantile_{q}'] = p_value

    for q, p_value in zip(config.quantiles, welch_p_quantiles):
        result[f'welch_p_value_quantile_{q}'] = p_value
    
    return result

def main():
    # 設定データ読み込み
    settings = read_setting('./settings.toml')['alpha_error_simulation']
    parent_seed = settings['parent_seed']
    n_jobs = settings['n_jobs']
    config = CommonSimConfig(
        sigma_g1=settings['group1_sigma'],
        sigma_ratio_log_range=settings['sigma_ratio_log_range'],
        n_sigma_point=settings['sigma_ratio_n_point'],
        total_sample_size=settings['total_sample_size'],
        min_sample_size_prop=settings['min_sample_size_proportion'],
        target_skew_exkurt=settings['target_skew_exkurt'],
        significance_alpha=settings['significance_alpha'],
        confidence_interval_alpha=settings['confidence_interval_alpha'],
        quantiles=settings['quantile'],
        n_simulation=settings['simulation_n_repete'],
        batch_size=settings['batch_size']
    )

    # sinh-arcsinh分布のパラメータを読み込み
    sash_params = read_shash_dist_parameter('./sinh_arcsinh_params.parquet')

    # 全タスクをリスト化
    tasks = []
    seed_seq = np.random.SeedSequence(parent_seed)

    # 標準偏差を振る範囲を設定
    sigma_list_g2 = make_sigma_list(config.sigma_ratio_log_range[0], config.sigma_ratio_log_range[1], config.n_sigma_point, config.sigma_g1)

    # 狙いの歪度・超過尖度に最も近い分布を実現するパラメータを選択
    for skew, exkurt in config.target_skew_exkurt:
        sash_params['sim_skew'] = skew
        sash_params['sim_exkurt'] = exkurt
        sash_params['distance'] = np.sqrt((sash_params['target_skewness'] - sash_params['sim_skew']) ** 2 + (sash_params['target_excess_kurtosis'] - sash_params['sim_exkurt']) ** 2)
        sash_params.sort_values('distance', inplace=True)
        target_params = sash_params.iloc[[0], :]
        act_skew = target_params['target_skewness'].values[0]
        act_exkurt = target_params['target_excess_kurtosis'].values[0]
        eps = target_params['eps'].values[0]
        delta = target_params['delta'].values[0]

        print(target_params)

        # サンプルサイズを動かす値を設定
        for total_sample_size in config.total_sample_size:
            group1_sample_size_list, group2_sample_size_list = make_sample_size_list(total_sample_size, config.min_sample_size_prop)

            # タスクのリスト化
            for sample_size_g1, sample_size_g2 in zip(group1_sample_size_list, group2_sample_size_list):
                for sigma_g2 in sigma_list_g2:
                    cell = CellConfig(
                        target_skewness=skew,
                        target_excess_kurtosis=exkurt,
                        actual_skewness=act_skew,
                        actual_excess_kurtosis=act_exkurt,
                        eps=eps,
                        delta=delta,
                        sigma_group1=config.sigma_g1,
                        sigma_group2=sigma_g2,
                        sample_size_group1=sample_size_g1,
                        sample_size_group2=sample_size_g2
                    )
                    tasks.append((cell, seed_seq.spawn(1)[0]))

    # 並列実行
    sim_generator = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
        delayed(run_one_cell)(cell, config, seed) for cell, seed in tasks
    )
    results = list(tqdm(sim_generator, total=len(tasks)))

    # データフレームにして出力
    sim_result_data = pd.DataFrame(results)
    os.makedirs('output', exist_ok=True)
    sim_result_data.to_parquet('./output/alpha_error_sim_shash-2_result.parquet', index=False)

if __name__ == '__main__':
    main()
