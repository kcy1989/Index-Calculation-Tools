import pandas as pd
import os

# ==========================================
# 全局變數設置 (Global Variables)
# ==========================================
INPUT_FILE = r'csv/HSCI/HSCI.csv'  # 你的輸入檔案名稱
OUTPUT_FILE = r'csv/HSCI/HSCI_CF.csv'  # 輸出檔案名稱
CAP_LEVEL = 0.10  # 10% 上限 (可在此修改，例如 0.15)

# 假設 CSV 中的欄位名稱 (請根據你的實際 CSV 修改)
COL_ID = 'RIC'  # 股票代號
COL_MV = 'MV'  # 市值 (Market Value)


def calculate_capped_weights(df, cap_limit):
    """
    計算符合 Capping 要求的權重與 Cap Factor
    """
    # 複製一份數據以免修改原始 DataFrame
    data = df.copy()

    # 確保市值是浮點數
    data[COL_MV] = data[COL_MV].astype(float)

    # 初始化變數
    n_stocks = len(data)

    # 檢查是否數學上不可能 (例如 10隻股票，每隻都要限制在 5% 以下，總和只有 50%，不可能湊成 100%)
    if cap_limit * n_stocks < 1.0:
        raise ValueError(f"錯誤: CAP_LEVEL ({cap_limit}) 太小，無法分配給 {n_stocks} 隻股票 (總和 < 100%)")

    # 初始狀態：所有股票都標記為 'Uncapped' (未受限)
    # 我們使用一個 Boolean mask，True 代表該股票已被鎖定為 Cap Level
    is_capped = pd.Series([False] * n_stocks, index=data.index)

    iteration = 0
    while True:
        iteration += 1

        # 1. 計算當前已被鎖定的總權重
        # 被鎖定的股票，權重強制等於 CAP_LEVEL
        weight_taken_by_capped = is_capped.sum() * cap_limit

        # 2. 計算剩餘可分配的權重
        weight_remaining = 1.0 - weight_taken_by_capped

        # 3. 計算未受限股票的總市值
        uncapped_mv_sum = data.loc[~is_capped, COL_MV].sum()

        # 邊界情況保護：如果剩餘權重還有，但沒有股票可分了 (理論上不會發生，除非數據異常)
        if uncapped_mv_sum == 0 and weight_remaining > 1e-9:
            break

        # 4. 試算未受限股票的權重 (按市值比例分配剩餘權重)
        # 暫定權重 = (該股市值 / 未受限總市值) * 剩餘權重
        current_weights = pd.Series(0.0, index=data.index)

        # 填入被鎖定的權重
        current_weights[is_capped] = cap_limit

        # 填入未鎖定的試算權重
        if uncapped_mv_sum > 0:
            uncapped_weights = (data.loc[~is_capped, COL_MV] / uncapped_mv_sum) * weight_remaining
            current_weights.update(uncapped_weights)

        # 5. 檢查是否有任何 '未受限' 的股票現在超過了 CAP_LEVEL
        # 我們只檢查 is_capped 為 False 的部分
        violations = (current_weights > cap_limit + 1e-9) & (~is_capped)

        if not violations.any():
            # 如果沒有任何違規，迴圈結束
            print(f"迭代完成，共進行了 {iteration} 次重新分配。")
            break
        else:
            # 6. 找出違規者，將其標記為 'Capped'
            # 策略：通常將所有超標者一次性加入 Capped 清單，然後重新計算
            # 這樣可以加速收斂
            new_capped_indices = violations[violations].index
            is_capped[new_capped_indices] = True
            print(
                f"迭代 {iteration}: 發現 {len(new_capped_indices)} 隻股票超標，將其權重鎖定為 {cap_limit:.1%} 並重新分配...")

    # ==========================================
    # 計算 CF (Cap Factor)
    # 邏輯: Weight = (MV * CF) / Sum(MV * CF)
    # 我們設定未受限股票的 CF = 1.0
    # ==========================================

    final_weights = current_weights

    # 計算 "調整後總市值" (Implied Total Adjusted Market Cap)
    # 對於未受限股票 (CF=1): Weight = MV / Total_Adj_MV
    # 所以 Total_Adj_MV = MV / Weight
    # 我們取所有未受限股票的總市值 / 它們的總權重 來計算這個常數

    uncapped_mask = ~is_capped
    if uncapped_mask.any():
        sum_mv_uncapped = data.loc[uncapped_mask, COL_MV].sum()
        sum_w_uncapped = final_weights[uncapped_mask].sum()
        total_adj_market_cap = sum_mv_uncapped / sum_w_uncapped
    else:
        # 如果所有股票都被 Cap 了 (極端情況，例如 10隻股票，Cap 10%)
        # 則 Total_Adj_MV 可以是任意值，只要相對比例對即可，這裡用原總市值
        total_adj_market_cap = data[COL_MV].sum()

    # 計算 CF
    # CF = (Target_Weight * Total_Adj_MV) / Original_MV
    data['Final_Weight'] = final_weights
    data['Cap_Factor'] = (data['Final_Weight'] * total_adj_market_cap) / data[COL_MV]

    # 修正浮點數誤差，將 CF > 0.999999 設為 1.0
    data.loc[data['Cap_Factor'] > 0.999999, 'Cap_Factor'] = 1.0

    # 計算調整後市值 (用於驗證)
    data['Adjusted_MV'] = data[COL_MV] * data['Cap_Factor']

    return data


if __name__ == "__main__":
    try:
        # 2. 讀取 CSV
        print(f"正在讀取 {INPUT_FILE}...")
        df_input = pd.read_csv(INPUT_FILE)

        # 3. 執行計算
        print(f"開始計算，限制上限 (Cap Level) 為: {CAP_LEVEL:.1%}")
        df_result = calculate_capped_weights(df_input, CAP_LEVEL)

        # 4. 格式化輸出
        # 顯示百分比格式以便閱讀
        output_display = df_result.copy()
        output_display['Final_Weight'] = output_display['Final_Weight'].map('{:.4%}'.format)
        output_display['Cap_Factor'] = output_display['Cap_Factor'].map('{:.6f}'.format)

        print("\n計算結果 (前 10 筆):")
        print(output_display[[COL_ID, COL_MV, 'Final_Weight', 'Cap_Factor', 'Adjusted_MV']].head(10))

        # 5. 存檔
        df_result.to_csv(OUTPUT_FILE, index=False)
        print(f"\n完整結果已儲存至: {OUTPUT_FILE}")

        # 驗證總權重
        total_weight = df_result['Final_Weight'].sum()
        print(f"驗證總權重: {total_weight:.6f} (應接近 1.0)")

    except Exception as e:
        print(f"發生錯誤: {e}")
