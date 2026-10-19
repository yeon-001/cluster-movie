import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    numeric_cols = [
        "first_scrn",
        "first_show",
        "first_date",
        "peak",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 네 가지 원래 속성
    df["스크린 수"] = df["first_scrn"]
    df["누적 관객"] = df["total_audi"]
    df["10위권 일수"] = df["days_in_top10"]

    # 롱런 지수 = 누적 관객 / 첫 주 관객, 최대 20
    df["롱런 지수"] = df["total_audi"] / df["first_week_audi"]
    df["롱런 지수"] = df["롱런 지수"].clip(upper=20)

    # 로그 변환
    df["스크린 수_변환"] = np.log10(df["스크린 수"])
    df["누적 관객_변환"] = np.log10(df["누적 관객"])

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


features = [
    "스크린 수",
    "누적 관객",
    "10위권 일수",
    "롱런 지수",
]

feature_cols = {
    "스크린 수": "스크린 수_변환",
    "누적 관객": "누적 관객_변환",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수",
}


# --------------------------------------------------
# 군집화에 사용할 속성 선택
# --------------------------------------------------

st.subheader("묶는 데 사용할 속성")

selected = st.multiselect(
    "두 개 이상 선택하세요.",
    features,
    default=features,
)

if len(selected) < 2:
    st.warning("묶음에 사용할 속성을 두 개 이상 선택하세요.")
    st.stop()


# --------------------------------------------------
# 결측값 및 첫 주 관객 0인 영화 제외
# --------------------------------------------------

cluster_df = df[
    (df["first_week_audi"] > 0)
    & df[[feature_cols[f] for f in selected]].notna().all(axis=1)
].copy()

# 로그 변환이 가능한 값인지 확인
cluster_df = cluster_df[
    np.isfinite(cluster_df["스크린 수_변환"])
    & np.isfinite(cluster_df["누적 관객_변환"])
    & (cluster_df["스크린 수"] > 0)
    & (cluster_df["누적 관객"] > 0)
].copy()


st.markdown(
    f"**전체 편수: {len(df):,}편　|　묶은 편수: {len(cluster_df):,}편**"
)


if len(cluster_df) < 3:
    st.error("세 묶음으로 나누려면 유효한 영화가 최소 3편 필요합니다.")
    st.stop()


# --------------------------------------------------
# 표준화 + K-means
# --------------------------------------------------

X = cluster_df[
    [feature_cols[f] for f in selected]
].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

cluster_df["cluster"] = kmeans.fit_predict(X_scaled)


# --------------------------------------------------
# 누적 관객 평균이 큰 군집부터 ㉮ ㉯ ㉰
# --------------------------------------------------

cluster_order = (
    cluster_df
    .groupby("cluster")["누적 관객"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)

label_map = {
    cluster: chr(ord("㉮") + i)
    for i, cluster in enumerate(cluster_order)
}

cluster_df["묶음"] = cluster_df["cluster"].map(label_map)

label_order = ["㉮", "㉯", "㉰"]


# --------------------------------------------------
# 2차원 산점도
# --------------------------------------------------

st.subheader("2차원 산점도")

col1, col2 = st.columns(2)

with col1:
    x2 = st.selectbox(
        "가로축",
        selected,
        index=0,
    )

with col2:
    y2 = st.selectbox(
        "세로축",
        selected,
        index=min(1, len(selected) - 1),
    )


fig2 = px.scatter(
    cluster_df,
    x=feature_cols[x2],
    y=feature_cols[y2],
    color="묶음",
    category_orders={
        "묶음": label_order
    },
    hover_name="movieNm",
    labels={
        feature_cols[x2]: x2,
        feature_cols[y2]: y2,
        "묶음": "묶음",
    },
)

fig2.update_traces(
    marker={
        "size": 8
    }
)

st.plotly_chart(
    fig2,
    use_container_width=True,
)


# --------------------------------------------------
# 3차원 산점도
# --------------------------------------------------

st.subheader("3차원 산점도")

if len(selected) < 3:

    st.info(
        "묶는 데 사용할 속성을 세 개 이상 고르면 "
        "3차원 산점도가 표시됩니다."
    )

else:

    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "x축",
            selected,
            index=0,
            key="x3",
        )

    with c2:
        y3 = st.selectbox(
            "y축",
            selected,
            index=1,
            key="y3",
        )

    with c3:
        z3 = st.selectbox(
            "z축",
            selected,
            index=2,
            key="z3",
        )

    fig3 = px.scatter_3d(
        cluster_df,
        x=feature_cols[x3],
        y=feature_cols[y3],
        z=feature_cols[z3],
        color="묶음",
        category_orders={
            "묶음": label_order
        },
        hover_name="movieNm",
        labels={
            feature_cols[x3]: x3,
            feature_cols[y3]: y3,
            feature_cols[z3]: z3,
            "묶음": "묶음",
        },
    )

    fig3.update_traces(
        marker={
            "size": 4
        }
    )

    fig3.update_layout(
        scene={
            "xaxis_title": x3,
            "yaxis_title": y3,
            "zaxis_title": z3,
        }
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
    )


# --------------------------------------------------
# 묶음별 요약
# --------------------------------------------------

st.subheader("묶음별 요약")

summary = (
    cluster_df
    .groupby("묶음")
    .agg(
        편수=("movieNm", "size"),
        스크린_수_평균=("스크린 수", "mean"),
        누적_관객_평균=("누적 관객", "mean"),
        **{
            "10위권 일수_평균": (
                "10위권 일수",
                "mean",
            )
        },
        롱런_지수_평균=("롱런 지수", "mean"),
    )
    .reindex(label_order)
    .reset_index()
)

summary.columns = [
    "묶음",
    "편수",
    "스크린 수 평균",
    "누적 관객 평균",
    "10위권 일수 평균",
    "롱런 지수 평균",
]

st.dataframe(
    summary.style.format(
        {
            "편수": "{:,.0f}",
            "스크린 수 평균": "{:,.1f}",
            "누적 관객 평균": "{:,.1f}",
            "10위권 일수 평균": "{:,.1f}",
            "롱런 지수 평균": "{:,.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 묶음별 누적 관객 상위 5편
# --------------------------------------------------

st.subheader("묶음별 누적 관객 상위 5편")

for label in label_order:

    st.markdown(f"**{label}**")

    top5 = (
        cluster_df[
            cluster_df["묶음"] == label
        ]
        .sort_values(
            "누적 관객",
            ascending=False,
        )
        .head(5)
    )

    if top5.empty:
        st.write("해당 묶음에 영화가 없습니다.")

    else:
        for title in top5["movieNm"]:
            st.write(f"- {title}")
