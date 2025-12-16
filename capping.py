import pandas as pd
import numpy as np

# ==============================================================================
# 模組 1: 驗證檢查 (Validation)
# ==============================================================================
def validate_cap_feasibility(n_stocks, cap_level):
    """
    檢查 Cap Level 是否數學上可行。
    例如: 10 隻股票，Cap Level 5% -> 總和 50% < 100%，這是不可行的。
    """
    if n_stocks == 0:
        return False, "股票數量為 0"
    
    max_possible_total_weight = n_stocks * cap_level
    if max_possible_total_weight < 1.0:
        return False, f"Cap Level ({cap_level:.1%}) 太低，{n_stocks} 隻股票無法湊滿 100% (最大僅 {max_possible_total_weight:.1%})"
    
    return True, "OK"

# ==============================================================================
# 模組 2: 核心 Capping 演算法 (The Capping Engine)
# ==============================================================================
def apply_capping_logic(df, score_col, cap_level):
    """
    通用 Capping 函數。
    它不關心 score_col 是市值、ESG 分數還是股息率，它只負責把權重限制在 cap_level 以下。
    
    Args:
        df: DataFrame
        score_col: 用來決定權重分配的欄位 (例如 'MarketCap', 'ESG_Score')
        cap_level: 權重上限 (例如 0.10)
        
    Returns:
        Series: 最終權重 (Final Weights)
    """
    data = df.copy()
    n_stocks = len(data)
    
    # 1. 執行可行性檢查
    is_valid, msg = validate_cap_feasibility(n_stocks, cap_level)
    if not is_valid:
        raise ValueError(msg)

    # 2. 初始化狀態
    is_capped = pd.Series([False] * n_stocks, index=data.index)
    final_weights = pd.Series(0.0, index=data.index)
    
    iteration = 0
    while True:
        iteration += 1
        
        # 計算已被鎖定的權重總和
        weight_taken = is_capped.sum() * cap_level
        weight_remaining = 1.0 - weight_taken
        
        # 計算未受限股票的總分 (Score Sum)
        uncapped_score_sum = data.loc[~is_capped, score_col].sum()
        
        # 如果沒有剩餘權重或沒有未受限股票，跳出
        if uncapped_score_sum == 0:
            # 剩餘權重全部分配給未受限者(如果有的話)，或結束
            break

        # 試算權重
        current_weights = pd.Series(0.0, index=data.index)
        current_weights[is_capped] = cap_level # 鎖定者
        
        # 未鎖定者按分數比例分配剩餘權重
        # Weight = (Individual Score / Total Uncapped Score) * Remaining Weight
        uncapped_weights = (data.loc[~is_capped, score_col] / uncapped_score_sum) * weight_remaining
        current_weights.update(uncapped_weights)
        
        # 檢查違規 (誤差容忍度 1e-9)
        violations = (current_weights > cap_level + 1e-9) & (~is_capped)
        
        if not violations.any():
            final_weights = current_weights
            break
        else:
            # 發現違規，將其加入鎖定名單，下一輪迴圈重新分配
            is_capped[violations] = True
            
    return final_weights

# ==============================================================================
# 模組 3: 反算 CF (Back-calculate Cap Factor)
# ==============================================================================
def calculate_cf_from_weight(df, mv_col, weight_col):
    """
    根據最終權重 (Final Weight) 和 原始市值 (Market Cap)，反算 CF。
    公式概念: Weight = (MV * CF) / Total_Adj_MV
    所以 CF = (Weight * Total_Adj_MV) / MV
    
    注意: 這裡的 mv_col 必須是「市值」，即使你的權重是按 ESG 分配的，
    CF 通常還是應用在市值上 (Investable Weight Factor)。
    """
    data = df.copy()
    
    # 找出哪些股票沒有被大幅壓縮 (近似認為 CF=1 的基準組)
    # 這裡我們用一個數學推導：
    # Total_Adj_MV = Sum(MV_uncapped) / Sum(Weight_uncapped)
    # 如果所有股票都被 Cap，則取整個組合的比例
    
    # 這裡我們不依賴 is_capped 標記，而是直接動態計算隱含的常數
    # 為了簡單，我們假設 CF 最大不能超過 1.0。
    # 我們試圖找到一個 Scaling Factor K，使得 CF = (Weight * K) / MV <= 1.0
    # 且盡可能多的股票 CF 接近 1.0
    
    # 方法：對於每一隻股票，計算 K_i = MV_i / Weight_i
    # 理論上，未被 Cap 的股票，其 K_i 應該是相同的，且等於 Total_Adj_MV
    # 被 Cap 的股票，因為 Weight 被壓低，其 K_i 會變大 (不需要參考)
    # 我們取 K 的最小值作為 Total_Adj_MV (保守估計，確保所有 CF <= 1)
    
    # 避免除以零
    valid_rows = data[weight_col] > 0
    implied_k = data.loc[valid_rows, mv_col] / data.loc[valid_rows, weight_col]
    
    # 這裡的邏輯是：未被 Cap 的股票，權重是「自然」的，MV/Weight 比率應該最小 (或最標準)
    # 被 Cap 的股票，權重被強行壓低，分母變小，MV/Weight 會變大
    # 所以我們取 min() 作為基準
    total_adj_market_cap = implied_k.min()
    
    # 計算 CF
    cap_factors = (data[weight_col] * total_adj_market_cap) / data[mv_col]
    
    # 數值修整 (大於 1 的設為 1，極小的設為 0)
    cap_factors = cap_factors.clip(upper=1.0)
    cap_factors = cap_factors.fillna(0.0) # 處理 MV 為 0 的情況
    
    return cap_factors

# ==============================================================================
# 模組 4: 封裝流程 (Orchestrator)
# ==============================================================================
def process_index_capping(df, config):
    """
    處理單個指數的完整流程
    config = {
        'cap_level': 0.10,
        'mv_col': 'MarketCap',      # 用於計算 CF 的物理市值
        'score_col': 'MarketCap',   # 用於分配權重的分數 (可以是 ESG, Yield, 或 MV)
        'ticker_col': 'Ticker'
    }
    """
    # 1. 計算最終權重 (Capping)
    print(f"正在計算權重，上限: {config['cap_level']:.1%}，依據: {config['score_col']}")
    final_weights = apply_capping_logic(
        df, 
        score_col=config['score_col'], 
        cap_level=config['cap_level']
    )
    
    df_result = df.copy()
    df_result['Final_Weight'] = final_weights
    
    # 2. 反算 CF
    # 注意：CF 永遠是作用在市值 (MV) 上的
    df_result['Cap_Factor'] = calculate_cf_from_weight(
        df_result, 
        mv_col=config['mv_col'], 
        weight_col='Final_Weight'
    )
    
    return df_result

# ==============================================================================
# 使用範例 (For Loop 模擬)
# ==============================================================================
if __name__ == "__main__":
    # 模擬數據
    data = pd.DataFrame({
        'Ticker': ['A', 'B', 'C', 'D', 'E'],
        'MarketCap': [1000, 800, 600, 400, 200],  # 總市值 3000
        'ESG_Score': [50, 90, 80, 70, 60]         # 假設我們想用 ESG 分數加權
    })
    
    # 定義不同的指數設定
    indices_to_process = [
        {
            'name': 'Index_Standard_Cap',
            'cap_level': 0.30,          # 30% 上限
            'mv_col': 'MarketCap',
            'score_col': 'MarketCap'    # 傳統市值加權
        },
        {
            'name': 'Index_ESG_Weighted',
            'cap_level': 0.25,          # 25% 上限
            'mv_col': 'MarketCap',
            'score_col': 'ESG_Score'    # ESG 分數加權 (表現好的權重高)
        }
    ]
    
    for idx_config in indices_to_process:
        print(f"\n=== 處理指數: {idx_config['name']} ===")
        
        # 為了不影響原始 dataframe，傳入 copy
        result = process_index_capping(data.copy(), idx_config)
        
        # 顯示結果
        print(result[['Ticker', idx_config['mv_col'], idx_config['score_col'], 'Final_Weight', 'Cap_Factor']])
        
        # 驗證
        print(f"總權重: {result['Final_Weight'].sum():.4f}")
