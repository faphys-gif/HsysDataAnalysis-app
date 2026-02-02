# modules/plot_graph.py
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns
import numpy as np

# 시각화 함수 - Sales
def plot_sales_by_month(data, Customer):    
    filtered_data = data[data['고객명'].isin(Customer)] 
    filtered_data['SalesDate'] = pd.to_datetime(filtered_data['판매일자']) 
    filtered_data['Month'] = filtered_data['SalesDate'].dt.month
    sales_totals = filtered_data.groupby('Month')['판매량'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(sales_totals.index, sales_totals.values)
    ax.set_title('Monthly Sales Volume Trend')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Sales Volume')
    fig.autofmt_xdate(rotation=0)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_sales_by_weekly(data, Customer):
    filtered_data = data[data['고객명'].isin(Customer)] 
    filtered_data['SalesDate'] = pd.to_datetime(filtered_data['판매일자'])
    filtered_data['WeekNumber'] = filtered_data['SalesDate'].dt.isocalendar().week 
    sales_totals = filtered_data.groupby('WeekNumber')['판매량'].sum()

    # Line 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sales_totals.index, sales_totals.values, marker='o')
    ax.set_title('Weekly Sales Volume Trend')
    ax.set_xlabel('Week')
    ax.set_ylabel('Total Sales Volume')
    fig.autofmt_xdate(rotation=0)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_sales_by_customer(data, Customer):
    filtered_data = data[data['고객명'].isin(Customer)]
    sales_totals = filtered_data.groupby('고객명')['판매량'].sum()
    sales_totals = sales_totals.sort_values(ascending=False)
    sales_top10_series = sales_totals.head(10)
    sales_top10 = sales_top10_series.reset_index()
    sales_top10.columns = ['고객명', '판매량']

     # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    #ax.bar(sales_top10.index, sales_top10.values)
    sns.barplot(data=sales_top10, x='고객명', y='판매량', hue='고객명', palette='viridis', ax=ax, legend=False)
    ax.set_title('Top 10 Sales Volume by Customer')
    ax.set_xlabel('Customer')
    ax.set_ylabel('Total Quantity')    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig

def plot_sales_by_item(data, items):    
    filtered_data = data[data['제품코드'].isin(items)]
    sale_totals = filtered_data.groupby('제품 분류')['판매량'].sum()
    sale_totals = sale_totals.sort_values(ascending=False)
    sale_top10_series = sale_totals.head(10)
    sale_top10 = sale_top10_series.reset_index()
    sale_top10.columns = ['제품분류', '판매량']

    # 파이 차트 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sale_top10['판매량'], 
           labels=sale_top10['제품분류'], 
           autopct='%1.1f%%',        
           startangle=140,
           textprops={'fontsize': 8}
           )
    #ax.set_title('Sales Data by Customer', fontsize=6)
    plt.tight_layout()
    return fig

def plot_sales_by_weekday(data, customers):
    # 시간 정보 변환
    data['Date'] = pd.to_datetime(data['판매일자'])
    data['DayName_en'] = data['Date'].dt.day_name()
    
    filtered_data = data[data['고객명'].isin(customers)]
    
    # 2. 상위 10개 설비(Machine) 추출
    top_10_customers = filtered_data.groupby('고객명')['판매량'].sum().nlargest(10).index
    
    # 4. 상위 10x10 데이터만 필터링
    final_data = filtered_data[
        (filtered_data['고객명'].isin(top_10_customers))
    ]

    heatmap_data = final_data.pivot_table(
        index='DayName_en', 
        columns='고객명', 
        values='판매량', 
        aggfunc='sum',
        fill_value=0 # 데이터가 없는 조합은 0으로 채웁니다.
    )
    
    # 열지도 생성
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        ax=ax,
        annot=True, # 셀에 생산량 값 표시
        fmt='.0f',    # 값을 정수 형식으로 표시
        cmap='YlGnBu', # 컬러맵 지정 (값이 클수록 진한 파란색)
        linewidths=0.5, # 셀 경계선
        linecolor='black',
        cbar_kws={'label': '총 판매량'} # 컬러바 레이블
        )
    
    ax.set_title('Top 10 고객의 요일별 판매량 히트맵', fontsize=15)
    ax.set_xlabel('고객', fontsize=12)
    ax.set_ylabel('요일', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    #ax.set_yticks(rotation=0)
    plt.tight_layout()

    return fig

# 시각화 함수 - Productions
def plot_prod_by_month(data, machines):
    df = pd.DataFrame(data)
    df['ProdDate'] = pd.to_datetime(df['생산일자'])    
    df['Month'] = df['ProdDate'].dt.month
    filtered_data = df[df['생산설비'].isin(machines)]
    
    prod_totals = filtered_data.groupby('Month')['생산량'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(prod_totals.index, prod_totals.values)
    ax.set_title('Monthly Production Volume Trend')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Production Volume')
    fig.autofmt_xdate(rotation=0)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_prod_by_weekly(data, machines):
    filtered_data = data[data['생산설비'].isin(machines)] 
    filtered_data['date'] = pd.to_datetime(filtered_data['생산일자'])
    filtered_data['주차'] = filtered_data['date'].dt.isocalendar().week 
    daily_totals = filtered_data.groupby('주차')['생산량'].sum()

    # Line 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_totals.index, daily_totals.values, marker='o')
    ax.set_title('Weekly Production Volume Trend')
    ax.set_xlabel('Week')
    ax.set_ylabel('Total Production Volume')
    fig.autofmt_xdate(rotation=0)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_prod_by_item(data, items):    
    filtered_data = data[data['제품코드'].isin(items)]
    prod_totals = filtered_data.groupby('제품코드')['생산량'].sum()
    prod_totals = prod_totals.sort_values(ascending=False)
    prod_top10_series = prod_totals.head(10)
    prod_top10 = prod_top10_series.reset_index()
    prod_top10.columns = ['제품코드', '생산량']

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    #ax.bar(prod_totals.index, prod_totals.values)
    sns.barplot(data=prod_top10, x='제품코드', y='생산량', hue='제품코드', palette='viridis', ax=ax, legend=False)
    ax.set_title('Top 10 Production Volume by Item Code')
    ax.set_xlabel('Item Code')
    ax.set_ylabel('Total Quantity')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig

def plot_prod_by_machine(data, machines):
    filtered_data = data[data['생산설비'].isin(machines)]
    prod_totals = filtered_data.groupby('생산설비')['생산량'].sum()
    prod_totals = prod_totals.sort_values(ascending=False)
    prod_top10_series = prod_totals.head(10)
    prod_top10 = prod_top10_series.reset_index()
    prod_top10.columns = ['생산설비', '생산량']
    
    # 파이 차트 생성
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        prod_top10['생산량'], 
        labels=prod_top10['생산설비'], 
        autopct='%1.1f%%', 
        startangle=140,
        textprops={'fontsize': 8}
    )
    #ax.set_title('Production Data by Machine', fontsize=6)
    plt.tight_layout()
    return fig
    
def plot_prod_heatmap(data, machines):
    # 시간 정보 변환
    filtered_data = data[data['생산설비'].isin(machines)]
    
    # 2. 상위 10개 설비(Machine) 추출
    top_10_machines = filtered_data.groupby('생산설비')['생산량'].sum().nlargest(10).index
    
    # 3. 상위 10개 제품(Product) 추출
    top_10_prods = filtered_data.groupby('제품코드')['생산량'].sum().nlargest(10).index
    
    # 4. 상위 10x10 데이터만 필터링
    final_data = filtered_data[
        (filtered_data['생산설비'].isin(top_10_machines)) & 
        (filtered_data['제품코드'].isin(top_10_prods))
    ]

    # 5. 피벗 테이블 생성
    heatmap_data = final_data.pivot_table(
        index='생산설비', 
        columns='제품코드', 
        values='생산량', 
        aggfunc='sum',
        fill_value=0 # 데이터가 없는 조합은 0으로 채웁니다.
    )
    
    # 6. 정렬 (생산량이 많은 순서대로 행/열 재배치 - 선택사항)
    #heatmap_data = heatmap_data.loc[top_10_machines, top_10_prods]

    # 7. 열지도 생성
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        ax=ax,
        annot=True, # 셀에 생산량 값 표시
        fmt='.0f',    # 값을 정수 형식으로 표시
        cmap='YlGnBu', # 컬러맵 지정 (값이 클수록 진한 파란색)
        linewidths=0.5, # 셀 경계선
        linecolor='black',
        cbar_kws={'label': '총 생산량'} # 컬러바 레이블
        )
    
    ax.set_title('Top 10 생산설비 및 제품별 총 생산량 히트맵', fontsize=15)
    ax.set_xlabel('제품', fontsize=12)
    ax.set_ylabel('생산설비', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    #ax.set_yticks(rotation=0)
    plt.tight_layout()

    return fig

# 시각화 함수 - Quality
def plot_qc_by_month(data, machines):
    filtered_data = data[data['생산설비'].isin(machines)]
    filtered_data['ProdDate'] = pd.to_datetime(filtered_data['일자'])    
    filtered_data['Month'] = filtered_data['ProdDate'].dt.month
    
    ng_totals = filtered_data.groupby('Month')['불량 수량'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ng_totals.index, ng_totals.values, marker='o')
    ax.set_title('Monthy NG Quality Volume Trend')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total NG Quality Volume')
    fig.autofmt_xdate(rotation=45)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_qc_by_item(data, items):    
    filtered_data = data[data['제품코드'].isin(items)]
    ng_totals = filtered_data.groupby('제품코드')['불량 수량'].sum()
    ng_totals = ng_totals.sort_values(ascending=False)
    ng_top10_series = ng_totals.head(10)
    ng_top10 = ng_top10_series.reset_index()
    ng_top10.columns = ['제품코드', '불량 수량']

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ng_top10['제품코드'], ng_top10['불량 수량'])
    ax.set_title('Total NG Volume by Item Type')
    ax.set_xlabel('Item Code')
    ax.set_ylabel('Total Quantity')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_qc_by_machine(data, machines):
    # 1. 데이터 필터링
    filtered_data = data[data['생산설비'].isin(machines)]
    
    # 2. 설비별 불량 합계 계산 및 내림차순 정렬
    ng_totals = filtered_data.groupby('생산설비')['불량 수량'].sum().sort_values(ascending=False)
    
    # 3. 상위 10개와 나머지 분리
    top10 = ng_totals.head(7)
    others = ng_totals.iloc[7:]
    
    # 4. '기타' 항목 생성 (나머지가 있을 경우에만)
    if not others.empty:
        others_series = pd.Series({'기타': others.sum()})
        final_series = pd.concat([top10, others_series])
    else:
        final_series = top10

    # 5. 파이 차트 생성
    fig, ax = plt.subplots(figsize=(8, 8))
    
    wedges, texts, autotexts = ax.pie(
        final_series, 
        labels=final_series.index, 
        radius=0.9,
        autopct='%1.1f%%', 
        startangle=140,
        pctdistance=0.85, # 퍼센트 숫자를 중심에서 약간 바깥으로 이동
        colors=plt.cm.Paired.colors
    )
    
    # 6. 폰트 사이즈 미세 조절
    plt.setp(texts, size=9)      # 항목 이름 크기
    plt.setp(autotexts, size=8)  # 퍼센트 숫자 크기
    
    ax.set_title('생산설비별 불량 비중 (상위 7개 및 기타)', fontsize=12)
    
    # 도넛 차트 형태로 만들고 싶다면 아래 주석 해제 (중앙에 흰색 원 추가)
    centre_circle = plt.Circle((0,0), 0.50, fc='white')
    fig.gca().add_artist(centre_circle)
    
    plt.tight_layout()
    return fig

def plot_qc_by_ngtype(data, machines):
    filtered_data = data[data['생산설비'].isin(machines)]
    ng_totals = filtered_data.groupby('불량 유형')['불량 수량'].sum()
    
    # 파이 차트 생성
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(ng_totals, labels=ng_totals.index, radius=0.8, autopct='%1.1f%%',  textprops={'fontsize': 9}, startangle=140)
    ax.set_title('Quality Data by NG Type')
    plt.tight_layout()
    return fig
    

def plot_qc_heatmap(data, items):
    # 시간 정보 변환
    filtered_data = data[data['제품코드'].isin(items)]
     # 2. 상위 10개 설비(Machine) 추출
    top_10_items = filtered_data.groupby('제품코드')['불량 수량'].sum().nlargest(10).index
    
    # 3. 상위 10개 제품(Product) 추출
    top_10_types = filtered_data.groupby('불량 유형')['불량 수량'].sum().nlargest(10).index
    
    # 4. 상위 10x10 데이터만 필터링
    final_data = filtered_data[
        (filtered_data['제품코드'].isin(top_10_items)) & 
        (filtered_data['불량 유형'].isin(top_10_types))
    ]

    heatmap_data = final_data.pivot_table(
        index='제품코드', 
        columns='불량 유형', 
        values='불량 수량', 
        aggfunc='sum',
        fill_value=0 # 데이터가 없는 조합은 0으로 채웁니다.
    )
    
    # 열지도 생성
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        ax=ax,
        annot=True, # 셀에 생산량 값 표시
        fmt='.0f',    # 값을 정수 형식으로 표시
        cmap='YlGnBu', # 컬러맵 지정 (값이 클수록 진한 파란색)
        linewidths=0.5, # 셀 경계선
        linecolor='black',
        cbar_kws={'label': '불량'} # 컬러바 레이블
        )
    
    ax.set_title('불량유형 및 품목별 불량 히트맵', fontsize=15)
    ax.set_xlabel('품목코드', fontsize=12)
    ax.set_ylabel('불량유형', fontsize=12)
    #ax.set_xticks(rotation=45, ha='right')
    #ax.set_yticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_pur_by_month(data, items):
    filtered_data = data[data['ITEM_CODE'].isin(items)]
    filtered_data['RCV_DATE'] = pd.to_datetime(filtered_data['RCV_DT'])    
    filtered_data['RCV_MONTH'] = filtered_data['RCV_DATE'].dt.month
    
    pur_totals = filtered_data.groupby('RCV_MONTH')['RCV_QTY'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(pur_totals.index, pur_totals.values)
    ax.set_title('Total Purchasing Volume Monthly', fontsize=15)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Total Quantity', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_pur_by_item(data, items):
    filtered_data = data[data['ITEM_CODE'].isin(items)]
    # 2. 상위 10개 품목(Item) 추출
    pur_totals = filtered_data.groupby('ITEM_CODE')['RCV_QTY'].sum()
    pur_totals = pur_totals.sort_values(ascending=False)

    pur_top10_series = pur_totals.head(10)
    pur_top10 = pur_top10_series.reset_index()
    pur_top10.columns = ['ITEM_CODE', 'RCV_QTY']

     # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.bar(sales_top10.index, sales_top10.values)
    sns.barplot(data=pur_top10, x='ITEM_CODE', y='RCV_QTY', hue='ITEM_CODE', palette='viridis', ax=ax, legend=False)
    ax.set_title('Top 10 Purchsing Volume by Item')
    ax.set_xlabel('Item Code')
    ax.set_ylabel('Total Quantity')    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig

def plot_pur_by_ptnr(data, items):
    filtered_data = data[data['ITEM_CODE'].isin(items)]
    # 2. 상위 10개 품목(Item) 추출
    pur_totals = filtered_data.groupby('PTNR_NAME')['RCV_QTY'].sum()
    pur_totals = pur_totals.sort_values(ascending=False)

    pur_top10_series = pur_totals.head(10)
    pur_top10 = pur_top10_series.reset_index()
    pur_top10.columns = ['PTNR_NAME', 'RCV_QTY']

     # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.bar(sales_top10.index, sales_top10.values)
    sns.barplot(data=pur_top10, x='PTNR_NAME', y='RCV_QTY', hue='PTNR_NAME', palette='viridis', ax=ax, legend=False)
    ax.set_title('Top 10 Purchsing Volume by Partner')
    ax.set_xlabel('Ptnr Name')
    ax.set_ylabel('Total Quantity')    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig

def plot_pur_heatmap(data, items):
    # 시간 정보 변환
    filtered_data = data[data['ITEM_CODE'].isin(items)]
     # 2. 상위 10개 설비(Machine) 추출
    top_10_items = filtered_data.groupby('ITEM_CODE')['RCV_QTY'].sum().nlargest(10).index
    
    # 3. 상위 10개 제품(Product) 추출
    top_10_ptnrs = filtered_data.groupby('PTNR_NAME')['RCV_QTY'].sum().nlargest(10).index
    
    # 4. 상위 10x10 데이터만 필터링
    final_data = filtered_data[
        (filtered_data['ITEM_CODE'].isin(top_10_items)) & 
        (filtered_data['PTNR_NAME'].isin(top_10_ptnrs))
    ]

    heatmap_data = final_data.pivot_table(
        index='ITEM_CODE', 
        columns='PTNR_NAME', 
        values='RCV_QTY', 
        aggfunc='sum',
        fill_value=0 # 데이터가 없는 조합은 0으로 채웁니다.
    )
    
    # 열지도 생성
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        ax=ax,
        annot=True, # 셀에 생산량 값 표시
        fmt='.0f',    # 값을 정수 형식으로 표시
        cmap='YlGnBu', # 컬러맵 지정 (값이 클수록 진한 파란색)
        linewidths=0.5, # 셀 경계선
        linecolor='black',
        cbar_kws={'label': '구매량'} # 컬러바 레이블
        )
    
    ax.set_title('거래처 및 품목별 구매량 히트맵', fontsize=15)
    ax.set_xlabel('품목코드', fontsize=12)
    ax.set_ylabel('거래처', fontsize=12)
    #ax.set_xticks(rotation=45, ha='right')
    #ax.set_yticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_inv_by_month(data, items):
    filtered_data = data[data['Item Code'].isin(items)]
    inv_counts = filtered_data.groupby('Date')['재고량'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(inv_counts.index, inv_counts.values)
    ax.set_title('Total Inventory Volume Monthly', fontsize=15)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Total Quantity', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_psi_by_month(data, items):
    data['Date'] = pd.to_datetime(data['Date'])
    
    filtered_data = data[data['Item Code'].isin(items)]
    filtered_data['YearMonth'] = filtered_data['Date'].dt.to_period('M')
    
    prod_counts = filtered_data.groupby('YearMonth')['입고량'].sum()
    sale_counts = filtered_data.groupby('YearMonth')['출고량'].sum()
    inv_counts = filtered_data.groupby('YearMonth')['재고량'].sum()

    # 세개 시리즈를 하나의 데이터프레임으로 결합하여 정렬된 인덱스를 확보
    combined_df = pd.DataFrame({
        'Production': prod_counts,
        'Sales': sale_counts,
        'Inventory': inv_counts        
    }).fillna(0) # 값이 없는 월은 0으로 채웁니다.
    
    # 3. Bar 차트 생성 준비
    labels = combined_df.index.astype(str) # X축 레이블 (YYYY-MM)
    x = np.arange(len(labels))             # 막대 위치 (0, 1, 2, ...)
    width = 0.25                           # 각 막대의 너비
    
    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 4. 묶음 막대 그리기
    # 생산량 막대: 중앙에서 오른쪽으로 이동
    rects1 = ax.bar(x - width, combined_df['Production'], width, label='생산량 (Production)', color='salmon')
    
    rects2 = ax.bar(x, combined_df['Sales'], width, label='판매량 (Sales)', color='skyblue')
    
    # 재고량 막대: 중앙에서 왼쪽으로 이동
    rects3 = ax.bar(x + width, combined_df['Inventory'], width, label='재고량 (Inventory)', color='blue')
    
    
    # 5. 그래프 꾸미기
    ax.set_title('월별 총 상산량, 판매량 및 재고량 비교', fontsize=15)
    ax.set_xlabel('월 (Month - YYYY-MM)', fontsize=12)
    ax.set_ylabel('수량 (Quantity)', fontsize=12)
    
    ax.set_xticks(x)        # X축 눈금 위치 설정
    ax.set_xticklabels(labels, rotation=45, ha='right') # X축 레이블 설정 및 회전
    
    ax.legend(loc='upper right') # 범례 추가
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_inv_by_item(data, items):
    filtered_data = data[data['Item Code'].isin(items)]
    # 2. 상위 10개 품목(Item) 추출
    inv_totals = filtered_data.groupby('Item Code')['재고량'].sum()
    inv_totals = inv_totals.sort_values(ascending=False)

    inv_top10_series = inv_totals.head(10)
    inv_top10 = inv_top10_series.reset_index()
    inv_top10.columns = ['Item Code', '재고량']

     # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.bar(sales_top10.index, sales_top10.values)
    sns.barplot(data=inv_top10, x='Item Code', y='재고량', hue='Item Code', palette='viridis', ax=ax, legend=False)
    ax.set_title('Top 10 Sales Volume by Partner')
    ax.set_xlabel('Item Code')
    ax.set_ylabel('Total Quantity')    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig

def plot_inv_by_loc(data, items):
    filtered_data = data[data['Item Code'].isin(items)]
    inv_counts = filtered_data.groupby('Location')['재고량'].sum()

    plt.rc('font', family='MalgunGothic')
    # 파이 차트 생성
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.pie(
        inv_counts, 
        labels=inv_counts.index, 
        autopct='%1.1f%%', 
        startangle=90,
        textprops={'fontsize': 8}
    )
    plt.tight_layout()
    return fig

def plot_filtered_psi_by_daily(data, item):
    data['Date'] = pd.to_datetime(data['Date'])

    filtered_data = data[data['Item Code'].isin(item)]

    prod_counts = filtered_data.groupby('Date')['입고량'].sum()
    sale_counts = filtered_data.groupby('Date')['출고량'].sum()
    inv_counts = filtered_data.groupby('Date')['재고량'].sum()

    # 세개 시리즈를 하나의 데이터프레임으로 결합하여 정렬된 인덱스를 확보
    combined_df = pd.DataFrame({
        'Production': prod_counts,
        'Sales': sale_counts,
        'Inventory': inv_counts        
    }).fillna(0) # 값이 없는 월은 0으로 채웁니다.
    
    # 3. Bar 차트 생성 준비
    labels = combined_df.index.astype(str) # X축 레이블 (YYYY-MM)
    x = np.arange(len(labels))             # 막대 위치 (0, 1, 2, ...)
    width = 0.25                           # 각 막대의 너비
    
    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 4. 묶음 막대 그리기
    # 생산량 막대: 중앙에서 오른쪽으로 이동
    rects1 = ax.plot(x - width, combined_df['Production'], label='생산량 (Production)', color='salmon')    
    rects2 = ax.plot(x, combined_df['Sales'], label='판매량 (Sales)', color='skyblue')    
    # 재고량 막대: 중앙에서 왼쪽으로 이동
    rects3 = ax.plot(x + width, combined_df['Inventory'], label='재고량 (Inventory)', color='blue')    
    
    # 5. 그래프 꾸미기
    ax.set_title('일별 생산량, 판매량 및 재고량 비교', fontsize=15)
    ax.set_xlabel('(Month - YYYY-MM)', fontsize=12)
    ax.set_ylabel('수량 (Quantity)', fontsize=12)
    
    ax.set_xticks(x)        # X축 눈금 위치 설정
    ax.set_xticklabels(labels, rotation=45, ha='right') # X축 레이블 설정 및 회전
    
    ax.legend(loc='upper right') # 범례 추가
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    return fig
