import streamlit as st
import pandas as pd
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import numpy as np
import time

font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕
#font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" #Ubuntu

# 폰트 이름 등록
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# ✅ ③ 음수(-) 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

st.title('데이터 시각화 대시보드')

# 데이터 로드
df = pd.read_csv('data/data.csv')

# 데이터 시각화
fig, ax = plt.subplots()
df.plot(kind='line', ax=ax)
st.pyplot(fig) 

st.title('인터랙티브 데이터 분석 도구')

# 데이터 로드
df = pd.read_csv('data/data.csv')

# 사용자 입력 받기
column = st.selectbox('분석할 컬럼을 선택하세요', df.columns)
st.write(df[column].describe())


st.title('실시간 데이터 모니터링')

# 실시간 데이터 모니터링
for i in range(10):
    st.write(f'현재 시간: {time.strftime("%H:%M:%S")}')
    time.sleep(2)
