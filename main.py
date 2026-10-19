import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8",
    )

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
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # 네 가지 속성
    df["스크린 수"] = df["first_scrn"]
    df["누적 관객"] = df["total_audi"]
    df["10위권 일수"] = df["days_in_top10"]

    # 롱런 지수 = 누적 관객 / 첫 주 관객
    df["롱런 지수"] = (
        df["total_audi"] / df["first_week_audi"]
    )

    # 롱런 지수가 20을 넘으면 20으로 제한
    df["롱런 지수"] = df["롱런 지수"].clip(
        upper=20
    )

    # 스크린 수와 누적 관객은 상용로그 변환
    df["스크린 수_로그"] = np.log10(
        df["스크린 수"]
    )

    df["누적 관객_로그"] = np.log10(
        df["누적 관객"]
    )

    return df


try:
    df = load_data()

except Exception as e:
    st.error(
        f"데이터를 불러오지 못했습니다: {e}"
    )
    st.stop()


# --------------------------------------------------
# 네 가지 속성
# --------------------------------------------------

features = [
    "스크린 수",
    "누적 관객",
    "10위권 일수",
    "롱런 지수",
]


# 실제 군집화에 사용하는 열
cluster_columns = {
    "스크린 수": "스크린 수_로그",
    "누적 관객": "누적 관객_로그",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수",
}


# --------------------------------------------------
# 묶는 데 사용할 속성 선택
# --------------------------------------------------

st.subheader("묶는 데 사용할 속성")

selected = st.multiselect(
    "두 개 이상 선택하세요.",
    options=features,
    default=features,
)


if len(selected) < 2:

    st.warning(
        "묶는 데 사용할 속성을 두 개 이상 선택하세요."
    )

    st.stop()


# --------------------------------------------------
# 데이터 정리
# --------------------------------------------------

selected_columns = [
    cluster_columns[col]
    for col in selected
]


# 첫 주 관객이 0인 영화 제외
cluster_df = df[
    df["first_week_audi"] > 0
].copy()


# 선택한 속성 중 결측값이 있는 영화 제외
cluster_df = cluster_df.dropna(
    subset=selected_columns
)


# 로그 변환 대상은 양수여야 함
cluster_df = cluster_df[
    (cluster_df["스크린 수"] > 0)
    & (cluster_df["누적 관객"] > 0)
].copy()


# 무한값 제거
cluster_df = cluster_df[
    np.isfinite(
        cluster_df[selected_columns]
    ).all(axis=1)
].copy()


# --------------------------------------------------
# 전체 편수 / 묶은 편수
# --------------------------------------------------

st.markdown(
    f"**전체 편수: {len(df):,}편"
    f"　|　"
    f"묶은 편수: {len(cluster_df):,}편**"
)


if len(cluster_df) < 3:

    st.error(
        "세 묶음으로 나누려면 "
        "유효한 영화가 최소 3편 필요합니다."
    )

    st.stop()


# --------------------------------------------------
# 표준화
# --------------------------------------------------

X = cluster_df[
    selected_columns
].values


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# K-means로 세 묶음
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)


cluster_df["cluster"] = (
    kmeans.fit_predict(X_scaled)
)


# --------------------------------------------------
# 누적 관객 평균이 큰 순서로
# ㉮ ㉯ ㉰ 부여
# --------------------------------------------------

cluster_order = (
    cluster_df
    .groupby("cluster")["누적 관객"]
    .mean()
    .sort_values(
        ascending=False
    )
    .index
    .tolist()
)


label_map = {
    cluster: label
    for cluster, label in zip(
        cluster_order,
        ["㉮", "㉯", "㉰"],
    )
}


cluster_df["묶음"] = (
    cluster_df["cluster"]
    .map(label_map)
)


label_order = [
    "㉮",
    "㉯",
    "㉰",
]


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
        index=min(
            1,
            len(selected) - 1,
        ),
    )


fig2 = px.scatter(
    cluster_df,
    x=cluster_columns[x2],
    y=cluster_columns[y2],
    color="묶음",
    category_orders={
        "묶음": label_order
    },
    hover_name="movieNm",
    labels={
        cluster_columns[x2]: x2,
        cluster_columns[y2]: y2,
        "묶음": "묶음",
    },
)


fig2.update_traces(
    marker=dict(
        size=8
    )
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
        "묶는 데 사용할 속성을 "
        "세 개 이상 고르면 "
        "3차원 산점도가 표시됩니다."
    )

else:

    col1, col2, col3 = st.columns(3)


    with col1:

        x3 = st.selectbox(
            "x축",
            selected,
            index=0,
            key="x3",
        )


    with col2:

        y3 = st.selectbox(
            "y축",
            selected,
            index=1,
            key="y3",
        )


    with col3:

        z3 = st.selectbox(
            "z축",
            selected,
            index=2,
            key="z3",
        )


    fig3 = px.scatter_3d(
        cluster_df,
        x=cluster_columns[x3],
        y=cluster_columns[y3],
        z=cluster_columns[z3],
        color="묶음",
        category_orders={
            "묶음": label_order
        },
        hover_name="movieNm",
        labels={
            cluster_columns[x3]: x3,
            cluster_columns[y3]: y3,
            cluster_columns[z3]: z3,
            "묶음": "묶음",
        },
    )


    # 3차원 점을 작게
    fig3.update_traces(
        marker=dict(
            size=4
        )
    )


    st.plotly_chart(
        fig3,
        use_container_width=True,
    )


# --------------------------------------------------
# 묶음별 편 수 + 네 속성 평균
# --------------------------------------------------

st.subheader("묶음별 요약")


summary = (
    cluster_df
    .groupby("묶음")
    .agg(
        편수=("movieNm", "size"),
        스크린_수_평균=(
            "스크린 수",
            "mean",
        ),
        누적_관객_평균=(
            "누적 관객",
            "mean",
        ),
        십위권_일수_평균=(
            "10위권 일수",
            "mean",
        ),
        롱런_지수_평균=(
            "롱런 지수",
            "mean",
        ),
    )
    .reindex(label_order)
    .reset_index()
)


summary.columns = [
    "묶음",
    "편 수",
    "스크린 수 평균",
    "누적 관객 평균",
    "10위권 일수 평균",
    "롱런 지수 평균",
]


# 단위 표시
st.dataframe(
    summary.style.format(
        {
            "편 수": "{:,.0f}편",
            "스크린 수 평균": "{:,.1f}",
            "누적 관객 평균": "{:,.1f}",
            "10위권 일수 평균": "{:,.1f}일",
            "롱런 지수 평균": "{:,.2f}배",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 묶음별 누적 관객 상위 5편
# --------------------------------------------------

st.subheader(
    "묶음별 누적 관객 상위 5편"
)


for label in label_order:

    st.markdown(
        f"**{label}**"
    )


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


    for title in top5["movieNm"]:

        st.write(
            f"- {title}"
        )
