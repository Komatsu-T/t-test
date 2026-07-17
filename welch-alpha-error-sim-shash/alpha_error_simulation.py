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
    significance_alpha: float
    confidence_interval_alpha: float
    quantiles: tuple[float, ...]
    n_simulation: int
    batch_size: int

@dataclass(frozen=True)
class CellConfig:
    skewness: float
    excess_kurtosis: float
    eps: float
    delta: float
    sample_size_group1: int
    sample_size_group2: int

def read_setting(setting_file_path):
    """設定ファイルを読み込む"""
    with open(setting_file_path, mode="rb") as setting_file:
        settings = tomllib.load(setting_file)
    return settings

def read_shash_dist_parameter(parameter_file_path):
    """sinh-arcsinh分布のパラメータを読み込む"""
    data = pd.read_parquet(parameter_file_path)
    data = data[(data['converged']) & (data['target_skewness']>=0)].copy()
    data = data[['target_skewness', 'target_excess_kurtosis', 'eps', 'delta']].copy()
    return data.head(100)

def sinh_arcsinh_transform(z, eps, delta):
    """Y = sinh((asinh(z) + eps) / delta)を計算する"""
    return np.sinh((np.arcsinh(z) + eps) / delta)

def generate_2group_shash_random_values(
    eps,
    delta,
    sample_size_group1,
    sample_size_group2,
    n_simulation,
    seed
):
    """指定のパラメータをもつsinh-arcsinh分布から乱数を2群分生成する"""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(size=(sample_size_group1, n_simulation), dtype=np.float32)
    z2 = rng.standard_normal(size=(sample_size_group2, n_simulation), dtype=np.float32)
    group1 = sinh_arcsinh_transform(z1, eps, delta)
    group2 = sinh_arcsinh_transform(z2, eps, delta)
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
    sample_size_g1g2 = settings['sample_size']
    config = CommonSimConfig(
        significance_alpha=settings['significance_alpha'],
        confidence_interval_alpha=settings['confidence_interval_alpha'],
        quantiles=tuple(settings['quantile']),
        n_simulation=settings['simulation_n_repete'],
        batch_size=settings['batch_size']
    )

    # sinh-arcsinh分布のパラメータを読み込み
    sash_params = read_shash_dist_parameter('./sinh_arcsinh_params.parquet')
    
    # 全タスクをリスト化
    tasks = []
    seed_seq = np.random.SeedSequence(parent_seed)
    for sample_size_g1, sample_size_g2 in sample_size_g1g2:
        for row in sash_params.itertuples(index=False):
            cell = CellConfig(
                skewness=row.target_skewness,
                excess_kurtosis=row.target_excess_kurtosis,
                eps=row.eps,
                delta=row.delta,
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
    sim_result_data.to_parquet('./output/alpha_error_sim_shash_result.parquet', index=False)

if __name__ == '__main__':
    main()
