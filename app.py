import streamlit as st
import pandas as pd
import collections
import os
import altair as alt
import itertools
import ast
import numpy as np

pd.set_option("display.max_colwidth", 100)

FILTER = 'filter'
ALL = 'all'
CUTOFF = 0.3
MAX_WIDTH = 700
CHART_HEIGHT = 300

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False).encode('utf-8-sig')

def count_word(wakati_list):
    return collections.Counter(wakati_list).most_common()

def join_word(word_list):
    return " ".join(word_list)

def string_to_list(s):
    return ast.literal_eval(s)

def get_df_filter(com_list=[], con_list=[], minscore=0, maxscore=5, word=''):
    if not com_list:
        com_list = companies
    if not con_list:
        con_list = countries
    qu = '配給会社==@com_list and '\
    '制作国==@con_list and '\
    '評価>@minscore and '\
    '評価<@maxscore'
    df = st.session_state.df_revs.query(qu)
    df = df[df['タイトル'].str.contains(word)]
    return df

def get_words_list(df):
    word_list = df['レビュー_分かち']
    word_list = list(itertools.chain.from_iterable(word_list))
    count_words = count_word(word_list)
    return count_words, word_list
    
def plot_words_list(count_words):
    key_list, val_list = [], []
    for i in count_words[:10]:
        key_list.append(i[0])
        val_list.append(i[1])
        
    source = pd.DataFrame({
        '単語':key_list,
        '単語数':val_list
    })

    bargram = alt.Chart(source).mark_bar().encode(
        x=alt.X('単語数'),
        y=alt.Y('単語', sort='-x')
    ).properties(
        width=round(MAX_WIDTH/cols_num), 
        height=CHART_HEIGHT
    )
    
    return bargram

def plot_tfidf(tfidf_list):
    keys = []
    values = []
    for item in tfidf_list:
        keys.append(item[0])
        values.append(float(item[1]))
        
    source = pd.DataFrame({
        '単語':keys,
        '重要度':values
    })

    bargram = alt.Chart(source).mark_bar().encode(
        x=alt.X('重要度'),
        y=alt.Y('単語', sort='-x')
    ).properties(
        width=round(MAX_WIDTH/cols_num),
        height=CHART_HEIGHT
    )
    
    return bargram

# データの読み込み
with open('companies.txt', 'r', encoding='utf-8-sig') as f:
    companies = f.read().split('\n')
    
with open('countries.txt', 'r', encoding='utf-8-sig') as f:
    countries = f.read().split('\n')
    

# セッションの開始時のみ処理する
if 'is_init' not in st.session_state:
    st.session_state.is_init = True
    
if st.session_state.is_init:
    st.session_state.df_revs = pd.read_csv('movie_data_finaly.csv')
    st.session_state.df_revs['レビュー_分かち'] = \
        st.session_state.df_revs['レビュー_分かち'].apply(string_to_list)
    st.session_state.df_revs['tfidf'] = \
        st.session_state.df_revs['tfidf'].apply(string_to_list)
    st.session_state.is_init = False
    
MAX_NEGA = st.session_state.df_revs['ネガポジ_出力'].min()
MAX_POSI = st.session_state.df_revs['ネガポジ_出力'].max()
    
# 入力する部分
cols_num = st.number_input('比較する数', min_value=1, max_value=5, 
                           value=2, step=1)

cols = st.columns(cols_num)

com_list = []
con_list = []
mins, maxs = [], []
words = []
for i, col in enumerate(cols):
    with col:
        com_list.append(st.multiselect('配給会社の選択（未選択で全選択になる）',
                                options=companies, key=i))
        con_list.append(st.multiselect('制作国の選択（未選択で全選択になる）',
                                options=countries, key=i))
        minscore, maxscore = st.slider('最低評価と最高評価の指定',
                                    min_value=0.0,
                                    max_value=5.0,
                                    value=(0.0, 5.0),
                                    step=0.1,
                                    format='%.1f', key=i)
        mins.append(minscore)
        maxs.append(maxscore)
        words.append(st.text_input('フリーワード検索(部分一致、タイトルのみ)',
                                   key=i))

is_run = st.button('Go')

# 計算する部分
dfs = []
df_result = pd.DataFrame(columns=["条件", "レビュー_分かち", "単語数",
                                  "tfidf", "ネガポジ"])
if is_run:
    for i in range(cols_num):
        df_result.loc[i, '条件'] = f'[条件1] 配給会社:{companies[i]}, \
                                    制作国:{countries[i]}, \
                                    最小評価:{mins[i]}, \
                                    最大評価:{maxs[i]}, \
                                    フリーワード:{words[i]}'
                                    
        # 条件に合う映画を抽出
        df = get_df_filter(com_list=com_list[i], con_list=con_list[i],
                            minscore=mins[i], maxscore=maxs[i],
                            word=words[i])
        dfs.append(df.copy())   # 抽出した映画をリストに入れる
        
        # 単語数を計算
        count_words, word_list = get_words_list(df)
        
        # 分かち書きと、単語数を最終結果のデータフレームに入れる
        df_result.loc[i, 'レビュー_分かち'] = str(word_list)
        df_result.loc[i, '単語数'] = str(count_words)
        df_result.loc[i, ['レビュー_分かち', '単語数']] = \
            df_result.loc[i, ['レビュー_分かち', '単語数']].apply(string_to_list)
        
        # ネガポジ判定を行う
        df_result.loc[i, 'ネガポジ'] = df['ネガポジ_出力'].mean()
        
        # tfidfの集計を行う
        keys, values = [], []
        for items in df['tfidf']:
            for item in items:
                if item[0] in keys:
                    index = keys.index(item[0])
                    values[index] = round((item[1]+values[index])/2, 3)
                else:
                    keys.append(item[0])
                    values.append(item[1])
        df_tfidf = pd.DataFrame(np.array([keys, values]).T, columns=['単語', '重要度'])
        df_tfidf = df_tfidf.sort_values(by='重要度', ascending=False)
        df_tfidf = df_tfidf.set_axis(list(range(len(df_tfidf))), axis=0)
        
        tfidf_list = []
        for d in df_tfidf[:10].iterrows():
            tfidf_list.append((d[1][0], d[1][1]))
        df_result.loc[i, 'tfidf'] = str(tfidf_list)

    df_result.loc[:, 'tfidf'] = df_result.loc[:, 'tfidf'].apply(string_to_list)
        
    # 結果を表示する
    re_col = st.columns(cols_num)
    for i, col in enumerate(re_col):
        with col:
            st.markdown('### レビューの単語数上位10個')
            chart_word_list = plot_words_list(df_result.loc[i, '単語数'])
            st.write(chart_word_list)
            
            st.markdown('### レビューの中で特徴的な単語上位10個')
            chart_tfidf = plot_tfidf(df_result.loc[i, 'tfidf'])
            st.write(chart_tfidf)
            
            st.markdown('### ネガポジ判定')
            ng = df_result.loc[i, 'ネガポジ']
            if ng > 0:
                st.write('ポジ')
            elif ng < 0:
                st.write('ネガ')
            else:
                st.write('無し')
                
            
            st.markdown('### 映画情報')
            st.write(dfs[i].loc[:, 'タイトル':'制作国'].
                     style.format(formatter={('評価'): "{:.1f}"}))
            
    csv_file = convert_df(df_result)
    if st.download_button('データを保存（ダウンロード）', 
                          csv_file, 'movie_data.csv'):
        file_list = os.listdir('save_data')
        file_list = list(map(int, file_list))
        if file_list:
            df_result.to_csv(f'./save_data/{max(file_list)+1}.csv', 
                             index=False, encoding='utf-8-sig')
        else:
            df_result.to_csv(f'./save_data/{0}.csv',
                             index=False, encoding='utf-8-sig')
