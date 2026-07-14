import os
import tomllib
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from tqdm import tqdm

# Gauss-Hermite求積のノードと重みの計算
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(200)
_GH_Z = np.sqrt(2.0) * _GH_X
_GH_WN = _GH_W / np.sqrt(np.pi)
_ARG_CLIP = 175.0

def read_setting(setting_file_path):
    """設定ファイルを読み込む"""
    with open(setting_file_path, mode="rb") as setting_file:
        settings = tomllib.load(setting_file)
    return settings

def sinh_arcsinh_transform(z, eps, delta):
    """Y = sinh((asinh(z) + eps) / delta)を計算する"""
    arg = (np.arcsinh(z) + eps) / delta
    arg = np.clip(arg, (-1) * _ARG_CLIP, _ARG_CLIP) # sinhに渡す引数が大きくなりすぎないようクリップ
    arg = np.sinh(arg)
    return arg

def calc_moment(eps, delta):
    """sinh-arcsinh変換した変数の(mean, var, skewness, excess_kurtosis)を算出する"""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        arg = (np.arcsinh(_GH_Z) + eps) / delta
        if np.sum(_GH_WN[np.abs(arg) >= _ARG_CLIP]) > 1e-10:
            return np.nan, np.nan, 1e3, 1e3
        y = sinh_arcsinh_transform(_GH_Z, eps, delta)
        m1 = np.sum(_GH_WN * y)
        m2 = np.sum(_GH_WN * y ** 2)
        m3 = np.sum(_GH_WN * y ** 3)
        m4 = np.sum(_GH_WN * y ** 4)
        mean = m1
        var = m2 - m1 ** 2
        if not np.isfinite(var) or var <= 1e-10:
            return mean, var, 1e3, 1e3
        sd = np.sqrt(var)
        mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
        mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
        skew = mu3 / sd ** 3
        exkurt = mu4 / var ** 2 - 3.0
    if not (np.isfinite(skew) and np.isfinite(exkurt)):
        return mean, var, 1e3, 1e3
    return mean, var, skew, exkurt

def sinh_arcsinh_equations(vars, target_skew, target_exkurt):
    """fsolveで解く方程式。(eps, log_delta)について解くことでdelta>0が常に満たされるようにする。"""
    eps, log_delta = vars
    delta = np.exp(log_delta)
    _, _, skew, exkurt = calc_moment(eps, delta)
    return [skew - target_skew, exkurt - target_exkurt]

def check_universal_feasibility(skew, exkurt):
    """超過尖度と歪度の関係 kurtosis >= skew^2 - 2 が満たされる領域のみを選択する"""
    return exkurt >= skew ** 2 - 2.0

def make_skew_exkurt_combination(skew_min, skew_max, exkurt_min, exkurt_max, n):
    """歪度と超過尖度の組み合わせを生成する"""
    # 歪度・超過尖度の範囲が0を含んでいるかチェック
    if (skew_min > 0) or (skew_max < 0):
        raise ValueError('Skewness range must contain zero.')
    if (exkurt_min > 0) or (exkurt_max < 0):
        raise ValueError('Excess-kurtosis range must contain zero.')

    # 歪度・超過尖度の組み合わせを生成
    skews = np.linspace(skew_min, skew_max, n)
    exkurts = np.linspace(exkurt_min, exkurt_max, n)
    skew_exkurt_comb = np.array(list(product(skews, exkurts)))

    # 原点(歪度0, 超過尖度0)を組み合わせに追加
    if not (np.abs(skew_exkurt_comb) < 1e-9).all(axis=1).any():
        skew_exkurt_comb = np.vstack([skew_exkurt_comb, np.zeros(2)])

    # 存在する歪度・超過尖度の組み合わせを選択
    is_feasible = np.array([check_universal_feasibility(s, k) for s, k in skew_exkurt_comb])
    return skew_exkurt_comb[is_feasible]

def sort_from_origin(skew_exkurt_combination):
    """歪度・尖度の組み合わせ原点に近い順に並べ替える"""
    distance_sq = (skew_exkurt_combination ** 2).sum(axis=1)
    return skew_exkurt_combination[np.argsort(distance_sq)]

def solve(skew_exkurt_combination):
    """目的の歪度・超過尖度を持つ分布を作るためのパラメータを推定する"""
    result = []
    solved = []
    solved_arr = np.empty((0, 2))
    for i, (skew, exkurt) in enumerate(tqdm(skew_exkurt_combination)):
        # 初期値の設定。原点の場合は(0, 0)、原点以外は最も近い点の解を初期値とする。
        if i == 0:
            guess = [0.1, 0.1]
        else:
            d = np.sqrt((solved_arr[:, 0] - skew) ** 2 + (solved_arr[:, 1] - exkurt) ** 2)
            j = int(np.argmin(d))
            guess = [solved[j][2], solved[j][3]]
        guess = [0.0 if abs(v) < 1e-12 else v for v in guess]

        # 目的の歪度・超過尖度を実現するパラメータの推定
        sol, info, ier, msg = fsolve(sinh_arcsinh_equations, guess, args=(skew, exkurt), full_output=True)

        # 収束判定 (ierではなく残差で判定する)
        resid = float(np.max(np.abs(info['fvec'])))
        converged = resid < 1e-8

        # 推定結果の保存
        eps, log_delta = sol
        delta = float(np.exp(log_delta))
        skew_fvec, exkurt_fvec = info['fvec']
        nvec = info['nfev']
        _, _, rs, rk = calc_moment(eps, delta)

        result.append(
            dict(
                target_skewness=skew, target_excess_kurtosis=exkurt, converged=converged,
                eps=eps, delta=delta, actual_skewness=rs, actual_excess_kurtosis=rk,
                skew_fvec=skew_fvec,exkurt_fvec=exkurt_fvec,
                nvec=nvec, ier=ier, msg=msg
            )
        )

        # 次のステップの初期値用に結果を保存
        if converged:
            solved.append((skew, exkurt, eps, log_delta))
            solved_arr = np.vstack([solved_arr, [skew, exkurt]])

    return pd.DataFrame(result)

def main():
    # 設定ファイルの読み込み
    settings = read_setting('./settings.toml')['sinh_arcsinh_moment_matching']
    SKEW_MIN = settings['skewness_min']
    SKEW_MAX = settings['skewness_max']
    EXKURT_MIN = settings['excess_kurtosis_min']
    EXKURT_MAX = settings['excess_kurtosis_max']
    GIRD_N = settings['grid_n']

    # 目標とする歪度と超過尖度の組み合わせを実現するパラメータを推定
    skew_exkurt_comb = make_skew_exkurt_combination(
        skew_min=SKEW_MIN,
        skew_max=SKEW_MAX,
        exkurt_min=EXKURT_MIN,
        exkurt_max=EXKURT_MAX,
        n=GIRD_N
    )
    skew_exkurt_comb_sorted = sort_from_origin(skew_exkurt_comb)
    result = solve(skew_exkurt_comb_sorted)

    # 結果出力
    os.makedirs('output', exist_ok=True)
    result.to_parquet('./output/sinh_arcsinh_params.parquet', index=False)

if __name__ == '__main__':
    main()
