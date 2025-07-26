import streamlit as st
import pandas as pd
from transformers import pipeline, BertJapaneseTokenizer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import os

# unidicのインポートをtry-exceptで囲む（エラー時は画面に表示して処理停止）
try:
    import unidic
except Exception as e:
    st.error(f"unidicのインポートでエラーが発生しました:\n{e}")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    # MeCab辞書のパスをunidic-liteから取得
    mecab_dic_path = unidic.DICDIR
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        "koheiduck/bert-japanese-finetuned-sentiment",
        mecab_kwargs={"mecab_args": f"-d {mecab_dic_path}"}
    )
    return pipeline(
        "sentiment-analysis",
        model="koheiduck/bert-japanese-finetuned-sentiment",
        tokenizer=tokenizer,
        return_all_scores=True,
    )


model = load_sentiment_model()

st.title("Sentia - 高精度日本語感情分析（複数ファイル＋カスタム辞書対応＋拡張機能）")

if "custom_dict" not in st.session_state:
    st.session_state.custom_dict = {}

st.sidebar.header("カスタム辞書管理")

with st.sidebar.expander("カスタム辞書の追加・更新"):
    st.write("■ 単語を1つだけ追加・更新")
    new_word = st.text_input("単語を入力")
    new_score = st.selectbox("感情ラベルを選択", ["ポジティブ", "ネガティブ", "中立"])
    if st.button("辞書に追加・更新"):
        if new_word:
            st.session_state.custom_dict[new_word] = new_score
            st.success(f"単語「{new_word}」を辞書に追加・更新しました。")

    st.write("---")
    st.write("■ Excelファイルから一括登録（1列目: 単語, 2列目: 感情ラベル）")
    dict_file = st.file_uploader("辞書用Excelファイルをアップロード", type=["xlsx"], key="dict_uploader")
    if dict_file:
        try:
            df_dict = pd.read_excel(dict_file)
            added_count = 0
            for idx, row in df_dict.iterrows():
                word = str(row.iloc[0]).strip()
                label = str(row.iloc[1]).strip()
                if word and label in ["ポジティブ", "ネガティブ", "中立"]:
                    st.session_state.custom_dict[word] = label
                    added_count += 1
            st.success(f"{added_count}件の単語をカスタム辞書に追加・更新しました。")
        except Exception as e:
            st.error(f"辞書ファイルの読み込みに失敗しました: {e}")

with st.sidebar.expander("カスタム辞書の表示・検索・編集・削除"):
    search_query = st.text_input("🔍 単語で検索")
    filtered_dict = {k: v for k, v in st.session_state.custom_dict.items() if search_query in k}

    for word, label in filtered_dict.items():
        cols = st.columns([4, 3, 2, 2])
        with cols[0]:
            new_key = f"edit_{word}"
            new_word_edit = st.text_input("単語", word, key=new_key)
        with cols[1]:
            new_label_key = f"label_{word}"
            new_label = st.selectbox("ラベル", ["ポジティブ", "ネガティブ", "中立"], index=["ポジティブ", "ネガティブ", "中立"].index(label), key=new_label_key)
        with cols[2]:
            if st.button("更新", key=f"update_{word}"):
                del st.session_state.custom_dict[word]
                st.session_state.custom_dict[new_word_edit] = new_label
                st.success(f"「{word}」を「{new_word_edit}」に更新しました。")
                st.experimental_rerun()
        with cols[3]:
            if st.button("削除", key=f"delete_{word}"):
                if st.session_state.get("confirm_delete") == word:
                    del st.session_state.custom_dict[word]
                    st.success(f"単語「{word}」を削除しました。")
                    st.session_state.confirm_delete = None
                    st.experimental_rerun()
                else:
                    st.session_state.confirm_delete = word
                    st.warning(f"⚠️ 本当に「{word}」を削除しますか？ もう一度削除を押すと確定します。")

# --- ファイルアップロード ---
uploaded_files = st.file_uploader("Excelファイルをアップロード", type=["xlsx"], accept_multiple_files=True)

label_map = {
    "POSITIVE": "ポジティブ",
    "NEGATIVE": "ネガティブ",
    "NEUTRAL": "中立"
}

if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_excel(uploaded_file)
        st.subheader(f"ファイル: {uploaded_file.name}")

        selected_col = st.selectbox(f"分析する列を選択 ({uploaded_file.name})", df.columns, key=f"select_col_{uploaded_file.name}")
        date_col = st.selectbox(f"（任意）日付列を選択 ({uploaded_file.name})", [None] + list(df.columns), key=f"date_col_{uploaded_file.name}")

        def apply_custom_dictionary(text):
            if not isinstance(text, str): return None
            for word, label in st.session_state.custom_dict.items():
                if word in text:
                    return label
            return None

        result_key = f"result_{uploaded_file.name}"
        analyze_button_key = f"analyze_btn_{uploaded_file.name}"

        if st.button(f"{uploaded_file.name} を分析開始", key=analyze_button_key):
            with st.spinner("分析中..."):
                texts = df[selected_col].astype(str)
                custom_labels = texts.apply(apply_custom_dictionary)
                to_analyze = custom_labels.isna()
                raw_results = model(texts[to_analyze].tolist(), batch_size=16)
                predicted_labels = [max(scores, key=lambda x: x['score']) for scores in raw_results]

                df.loc[to_analyze, "感情判定"] = [label_map[res['label']] for res in predicted_labels]
                df.loc[to_analyze, "スコア"] = [res['score'] for res in predicted_labels]
                df.loc[~to_analyze, "感情判定"] = custom_labels[~to_analyze]
                df.loc[~to_analyze, "スコア"] = 1.0

                st.session_state[result_key] = df

        if result_key in st.session_state:
            df = st.session_state[result_key]

            st.write("### 感情分布")
            fig = px.pie(
                df,
                names="感情判定",
                title="感情内訳",
                color="感情判定",
                color_discrete_map={
                    "ポジティブ": "#2ecc71",  # 緑系
                    "ネガティブ": "#e74c3c",  # 赤系
                    "中立": "#95a5a6",       # グレー系
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            def plot_wordcloud(texts, title):
                if len(texts) == 0:
                    st.write(f"{title} のテキストがありません。")
                    return
                text_all = ' '.join(texts)
                font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc'
                wordcloud = WordCloud(
                    font_path=font_path,
                    background_color='white',
                    width=800,
                    height=400
                ).generate(text_all)

                font_prop = fm.FontProperties(fname=font_path)

                buf = io.BytesIO()
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.title(title, fontproperties=font_prop)
                plt.tight_layout()
                plt.savefig(buf, format='png')
                plt.close()
                st.image(buf)

            st.write("### ワードクラウド")
            for emotion in ["ポジティブ", "ネガティブ", "中立"]:
                filtered = df[df["感情判定"] == emotion][selected_col].dropna()
                plot_wordcloud(filtered.tolist(), f"{emotion} のワードクラウド")

            if date_col and date_col in df.columns:
                st.write("### 感情の時系列推移")
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                time_df = (
                    df.dropna(subset=[date_col])
                      .groupby([df[date_col].dt.date, "感情判定"])
                      .size()
                      .reset_index(name="件数")
                )
                fig_line = px.line(
                    time_df,
                    x=date_col,
                    y="件数",
                    color="感情判定",
                    title="感情の時系列推移"
                )
                st.plotly_chart(fig_line, use_container_width=True)

            st.write("### 抜粋サンプル")
            for emotion in ["ポジティブ", "ネガティブ", "中立"]:
                st.markdown(f"#### {emotion}の例")
                examples = df[df["感情判定"] == emotion][selected_col].dropna()
                if len(examples) == 0:
                    st.write("例がありません。")
                else:
                    samples = examples.sample(n=min(3, len(examples)), random_state=42)
                    for text in samples:
                        with st.expander(text[:50] + "..."):
                            st.write(text)

            st.write("### 絞り込み機能")
            filter_label = st.selectbox("感情でフィルタ", ["すべて", "ポジティブ", "ネガティブ", "中立"], key=f"filter_label_{uploaded_file.name}")
            filter_score = st.slider("スコアがこの値以上のデータのみ表示", 0.0, 1.0, 0.0, 0.01, key=f"filter_score_{uploaded_file.name}")

            filtered_df = df[df["スコア"] >= filter_score]
            if filter_label != "すべて":
                filtered_df = filtered_df[filtered_df["感情判定"] == filter_label]

            st.dataframe(filtered_df)

            st.download_button(
                "CSVでダウンロード",
                data=filtered_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{uploaded_file.name}_result.csv",
                mime="text/csv"
            )
else:
    st.info("Excelファイルをアップロードしてください。※複数可")
