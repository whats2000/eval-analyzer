import json
import io
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from pathlib import PurePosixPath

st.set_page_config(page_title="Twinkle Eval Analyzer", page_icon=":star2:", layout="wide")

st.title("✨ Twinkle Eval Analyzer (.json / .jsonl)")

# ----------------- Helpers -----------------

def _decode_bytes_to_text(b: bytes) -> str:
    for enc in ("utf-8", "utf-16", "utf-16le", "utf-16be", "big5", "cp950"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")

def read_twinkle_doc(file) -> Dict:
    raw = file.read()
    if isinstance(raw, bytes):
        text = _decode_bytes_to_text(raw)
    else:
        text = raw
    text = text.strip()
    try:
        obj = json.loads(text)
    except Exception:
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                obj = json.loads(line)
                break
            except Exception:
                continue
    if not isinstance(obj, dict):
        raise ValueError("檔案不是有效的 Twinkle Eval JSON 物件。")
    if "timestamp" not in obj or "config" not in obj or "dataset_results" not in obj:
        raise ValueError("缺少必要欄位")
    return obj

def extract_records(doc: Dict) -> Tuple[pd.DataFrame, Dict[str, float]]:
    model = doc.get("config", {}).get("model", {}).get("name", "<unknown>")
    timestamp = doc.get("timestamp", "<no-ts>")
    source_label = f"{model} @ {timestamp}"
    rows = []
    avg_map = {}
    for ds_path, ds_payload in doc.get("dataset_results", {}).items():
        ds_name = ds_path.split("datasets/")[-1].strip("/") if ds_path.startswith("datasets/") else ds_path
        avg_meta = ds_payload.get("average_accuracy") if isinstance(ds_payload, dict) else None
        results = ds_payload.get("results", []) if isinstance(ds_payload, dict) else []
        for item in results:
            if not isinstance(item, dict):
                continue
            file_path = item.get("file")
            acc_mean = item.get("accuracy_mean")
            if file_path is None or acc_mean is None:
                continue
            fname = PurePosixPath(file_path).name
            category = fname.rsplit(".", 1)[0]
            rows.append({
                "dataset": ds_name,
                "category": category,
                "file": fname,
                "accuracy_mean": float(acc_mean),
                "source_label": source_label
            })
        if avg_meta is None and results:
            vals = [float(it.get("accuracy_mean", np.nan)) for it in results if "accuracy_mean" in it]
            if vals:
                avg_meta = float(np.mean(vals))
        if avg_meta is not None:
            avg_map[ds_name] = avg_meta
    return pd.DataFrame(rows), avg_map

def load_all(files) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    frames = []
    meta = {}
    for f in files or []:
        try:
            doc = read_twinkle_doc(f)
        except Exception as e:
            st.error(f"❌ 無法讀取 {getattr(f, 'name', '檔案')}：{e}")
            continue
        df, avg_map = extract_records(doc)
        if not df.empty:
            frames.append(df)
            src = df["source_label"].iloc[0]
            meta[src] = avg_map
    if not frames:
        return pd.DataFrame(columns=["dataset", "category", "file", "accuracy_mean", "source_label"]), {}
    return pd.concat(frames, ignore_index=True), meta

# ----------------- Sidebar -----------------

with st.sidebar:
    files = st.file_uploader("選擇 Twinkle Eval 檔案", type=["json", "jsonl"], accept_multiple_files=True)
    df_all, meta_all = load_all(files)
    normalize_0_100 = st.checkbox("以 0–100 顯示", value=False)
    page_size = st.selectbox("每張圖顯示幾個類別", [10, 20, 30, 50, 100], index=1)
    sort_mode = st.selectbox("排序方式（原始成績）", ["依整體平均由高到低", "依整體平均由低到高", "依字母排序"])

    # === Baseline Δ 圖表的控制 ===
    st.markdown("---")
    st.subheader("差距分析設定（Baseline Δ）")
    options = ["|Δ| 由大到小", "Δ 由大到小（提升最多）", "Δ 由小到大（下降最多）", "依類別名稱"]
    default = "Δ 由大到小（提升最多）"
    delta_sort_mode = st.selectbox("差距排序方式（per-category）", options, index=options.index(default), key="delta_sort_mode")

    abs_threshold = st.number_input("只顯示 |Δ| ≥ 門檻（可選）", min_value=0.0, value=0.0, step=0.1)
    st.caption("Δ = Candidate 分數 − Baseline 分數；建議以 0–100 模式計算更直觀。")

if df_all.empty:
    st.info("請上傳 Twinkle Eval 檔案")
    st.stop()

# ----------------- 原始成績-----------------
all_datasets = sorted(df_all["dataset"].unique().tolist())
selected_dataset = st.selectbox("選擇資料集", options=all_datasets)
work = df_all[df_all["dataset"] == selected_dataset].copy()
metric_plot = "accuracy_mean" + (" (x100)" if normalize_0_100 else "")
work[metric_plot] = work["accuracy_mean"] * (100.0 if normalize_0_100 else 1.0)

order_df = work.groupby("category")[metric_plot].mean().reset_index()
if sort_mode == "依整體平均由高到低":
    order_df = order_df.sort_values(metric_plot, ascending=False)
elif sort_mode == "依整體平均由低到高":
    order_df = order_df.sort_values(metric_plot, ascending=True)
else:
    order_df = order_df.sort_values("category", ascending=True)

cat_order = order_df["category"].tolist()
work["category"] = pd.Categorical(work["category"], categories=cat_order, ordered=True)

n = len(cat_order)
pages = int(np.ceil(n / page_size))

st.markdown("## 📈 原始成績（各模型 × 類別）")
for p in range(pages):
    start, end = p * page_size, min((p + 1) * page_size, n)
    subset_cats = cat_order[start:end]
    sub = work[work["category"].isin(subset_cats)]
    st.subheader(f"📊 {selected_dataset}｜類別 {start+1}-{end} / {n}")
    base = alt.Chart(sub).encode(
        x=alt.X("category:N", sort=subset_cats),
        y=alt.Y(f"{metric_plot}:Q"),
        color=alt.Color("source_label:N"),
        tooltip=["source_label", "file", alt.Tooltip(metric_plot, format=".3f")]
    )
    bars = base.mark_bar().encode(xOffset="source_label")
    st.altair_chart(bars.properties(height=420), use_container_width=True)
    pivot = sub.pivot_table(index="category", columns="source_label", values=metric_plot)
    st.dataframe(pivot, use_container_width=True)
    st.download_button(
        label=f"下載此頁 CSV ({start+1}-{end})",
        data=pivot.reset_index().to_csv(index=False).encode("utf-8"),
        file_name=f"twinkle_{selected_dataset}_{start+1}_{end}.csv",
        mime="text/csv"
    )

# ----------------- 差距（Baseline Δ）分析 -----------------

st.markdown("---")
st.markdown("## ⚖️ 差距分析：Baseline vs. Candidates（Δ = Candidate − Baseline）")

# 使用與上方相同的資料集
dataset_for_delta = selected_dataset

df_delta_scope = df_all[df_all["dataset"] == dataset_for_delta].copy()
if df_delta_scope.empty:
    st.warning(f"在資料集 **{dataset_for_delta}** 找不到資料，請確認上傳的 JSON 含此資料集名稱。")
    try:
        st.stop()
    except Exception:
        raise SystemExit

# 統一與上方尺度（建議用 0–100 再做差）
score_col = "score_0100"
df_delta_scope[score_col] = df_delta_scope["accuracy_mean"] * (100.0 if normalize_0_100 else 1.0)

# 手動指定 Baseline 與 Candidates
all_sources_in_scope = sorted(df_delta_scope["source_label"].unique().tolist())
col1, col2 = st.columns([1, 2])
with col1:
    baseline = st.selectbox("選擇基準模型（Baseline）", options=all_sources_in_scope)
with col2:
    default_candidates = [s for s in all_sources_in_scope if s != baseline]
    candidates = st.multiselect("選擇要比較的候選模型（Candidates）", options=all_sources_in_scope, default=default_candidates)

if not candidates:
    st.info("請至少選擇一個 Candidate。")
    try:
        st.stop()
    except Exception:
        raise SystemExit

# 建立寬表（index=category；已固定 dataset_for_delta）
wide = df_delta_scope.pivot_table(index="category", columns="source_label", values=score_col, aggfunc="mean")

# 只比較 baseline 與 candidates 的交集列
valid_candidates = [c for c in candidates if c in wide.columns]
if baseline not in wide.columns:
    st.error("Baseline 在此資料集沒有任何分數可比。請換一個 Baseline 或資料集。")
    try:
        st.stop()
    except Exception:
        raise SystemExit
if not valid_candidates:
    st.error("選取的 Candidates 在此資料集沒有任何分數可比。請換一組 Candidates 或資料集。")
    try:
        st.stop()
    except Exception:
        raise SystemExit

# 計算 Δ 長表（保留 baseline/candidate 原始分數）
delta_rows = []
for c in valid_candidates:
    pair = wide[[baseline, c]].dropna()  # 僅兩者皆有分數的類別
    if pair.empty:
        continue
    for cat, row in pair.iterrows():
        b = float(row[baseline])
        s = float(row[c])
        delta = s - b
        if abs(delta) < abs_threshold:  # 門檻過濾
            continue
        delta_rows.append({
            "dataset": dataset_for_delta,
            "category": cat,
            "baseline": baseline,
            "candidate": c,
            "baseline_score": b,
            "candidate_score": s,
            "delta": delta
        })

delta_df = pd.DataFrame(delta_rows)
if delta_df.empty:
    st.warning("沒有符合條件的可比較類別（可能因缺漏或門檻過高）。")
    try:
        st.stop()
    except Exception:
        raise SystemExit

# 差距排序
if delta_sort_mode == "|Δ| 由大到小":
    delta_df = delta_df.sort_values("delta", key=lambda s: s.abs(), ascending=False)
elif delta_sort_mode == "Δ 由大到小（提升最多）":
    delta_df = delta_df.sort_values("delta", ascending=False)
elif delta_sort_mode == "Δ 由小到大（下降最多）":
    delta_df = delta_df.sort_values("delta", ascending=True)
else:
    delta_df = delta_df.sort_values("category", ascending=True)

# 圖表（Δ 不分頁，一次顯示全部類別）
tab1, tab2 = st.tabs(["📊 差距排行（per-category）", "📜 模型總結（per-candidate）"])

with tab1:
    sub = delta_df.copy()

    # === 先在 Pandas 內算出每個 candidate 的排序名次 ===
    if delta_sort_mode == "Δ 由大到小（提升最多）":
        sub["rank_in_candidate"] = sub.groupby("candidate")["delta"].rank(ascending=False, method="first")
        table_sort = lambda df: df.sort_values(["candidate", "rank_in_candidate"], ascending=[True, True])
        y_sort = alt.SortField("rank_in_candidate", order="ascending")
        resolve_y = "independent"

    elif delta_sort_mode == "Δ 由小到大（下降最多）":
        sub["rank_in_candidate"] = sub.groupby("candidate")["delta"].rank(ascending=True, method="first")
        table_sort = lambda df: df.sort_values(["candidate", "rank_in_candidate"], ascending=[True, True])
        y_sort = alt.SortField("rank_in_candidate", order="ascending")
        resolve_y = "independent"

    elif delta_sort_mode == "|Δ| 由大到小":
        sub["abs_delta"] = sub["delta"].abs()
        sub["rank_in_candidate"] = sub.groupby("candidate")["abs_delta"].rank(ascending=False, method="first")
        table_sort = lambda df: df.sort_values(["candidate", "rank_in_candidate"], ascending=[True, True])
        y_sort = alt.SortField("rank_in_candidate", order="ascending")
        resolve_y = "independent"

    else:  # 依類別名稱（字母序），共用排序
        # 不用 rank，直接字母序
        table_sort = lambda df: df.sort_values(["category", "candidate"], ascending=[True, True])
        y_sort = alt.SortField("category", order="ascending")
        resolve_y = "shared"

    st.subheader(f"🔎 {dataset_for_delta}｜Δ 排行（全部 {sub['category'].nunique()} 類別）")

    chart_height = 25 * max(1, sub["category"].nunique())

    base = alt.Chart(sub).encode(
        y=alt.Y("category:N", sort=y_sort, title="Category"),
        x=alt.X("delta:Q", title="Δ = Candidate − Baseline"),
        color=alt.Color("candidate:N", title="Candidate"),
        tooltip=[
            alt.Tooltip("category:N", title="Category"),
            alt.Tooltip("candidate:N", title="Candidate"),
            alt.Tooltip("baseline:N", title="Baseline"),
            alt.Tooltip("baseline_score:Q", title="Baseline 分數", format=".3f"),
            alt.Tooltip("candidate_score:Q", title="Candidate 分數", format=".3f"),
            alt.Tooltip("delta:Q", title="Δ", format=".3f"),
        ],
    )

    chart = (
        base.mark_bar()
        .encode(row=alt.Row("candidate:N", header=alt.Header(title=None)))
        .properties(height=chart_height)
        .resolve_scale(y=resolve_y)  # 各 candidate 分面各自排序或共用
    )
    st.altair_chart(chart, use_container_width=True)

    # 表格：依 rank_in_candidate 排序，與圖一致
    table = table_sort(sub)[["category", "candidate", "baseline_score", "candidate_score", "delta"]]
    st.dataframe(table, use_container_width=True)

    st.download_button(
        label="下載 Δ 排行 CSV（全部類別）",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"delta_{dataset_for_delta}_ALL.csv",
        mime="text/csv",
    )



with tab2:
    # per-candidate 總結：mean/median Δ、win/lose/tie、覆蓋率、Top/Bottom-N
    summaries = []
    top_k = st.number_input("Top/Bottom-N（顯示每個 Candidate 的最大/最小差距分類）", min_value=1, value=10, step=1)

    for c in valid_candidates:
        pair = wide[[baseline, c]].dropna()
        if pair.empty:
            continue
        deltas = pair[c] - pair[baseline]
        m = float(np.mean(deltas))
        med = float(np.median(deltas))
        win = int((deltas > 0).sum())
        lose = int((deltas < 0).sum())
        tie = int((deltas == 0).sum())
        coverage = f"{len(deltas)}/{wide.shape[0]}"  # 有共同分數的類別數 / 全部類別數

        # 取 Top/Bottom-N 類別（按 Δ）
        top_rows = (pair.assign(delta=deltas)
                    .sort_values("delta", ascending=False)
                    .head(top_k)
                    .reset_index()[["category", baseline, c, "delta"]])
        bottom_rows = (pair.assign(delta=deltas)
                       .sort_values("delta", ascending=True)
                       .head(top_k)
                       .reset_index()[["category", baseline, c, "delta"]])

        summaries.append({
            "candidate": c,
            "mean_delta": m,
            "median_delta": med,
            "win": win,
            "lose": lose,
            "tie": tie,
            "coverage": coverage,
            "top_list": top_rows,
            "bottom_list": bottom_rows
        })

    if not summaries:
        st.warning("沒有可用的 per-candidate 總結（可能都沒有交集）。")
    else:
        # 概覽表
        overview = pd.DataFrame([{
            "Candidate": s["candidate"],
            "Mean Δ": s["mean_delta"],
            "Median Δ": s["median_delta"],
            "Win": s["win"],
            "Lose": s["lose"],
            "Tie": s["tie"],
            "Coverage (交集/總類別)": s["coverage"],
        } for s in summaries]).sort_values("Mean Δ", ascending=False)
        st.markdown("### 總覽（與 Baseline 成對比較）")
        st.dataframe(overview, use_container_width=True)
        st.download_button(
            label="下載 per-candidate 總覽 CSV",
            data=overview.to_csv(index=False).encode("utf-8"),
            file_name=f"delta_overview_{dataset_for_delta}.csv",
            mime="text/csv"
        )

        # 逐 Candidate 顯示 Top/Bottom-N 清單（可收合）
        st.markdown("### 各 Candidate 的差距清單（Top/Bottom-N）")
        for s in summaries:
            with st.expander(f"🔸 {s['candidate']}"):
                st.write("**Top-N（提升最多）**")
                top_tbl = s["top_list"].rename(columns={baseline: "baseline_score", s["candidate"]: "candidate_score"})
                st.dataframe(top_tbl, use_container_width=True)
                st.download_button(
                    label=f"下載 {s['candidate']} Top-N",
                    data=top_tbl.to_csv(index=False).encode("utf-8"),
                    file_name=f"delta_top_{dataset_for_delta}_{s['candidate']}.csv",
                    mime="text/csv"
                )

                st.write("**Bottom-N（下降最多）**")
                bottom_tbl = s["bottom_list"].rename(columns={baseline: "baseline_score", s["candidate"]: "candidate_score"})
                st.dataframe(bottom_tbl, use_container_width=True)
                st.download_button(
                    label=f"下載 {s['candidate']} Bottom-N",
                    data=bottom_tbl.to_csv(index=False).encode("utf-8"),
                    file_name=f"delta_bottom_{dataset_for_delta}_{s['candidate']}.csv",
                    mime="text/csv"
                )
