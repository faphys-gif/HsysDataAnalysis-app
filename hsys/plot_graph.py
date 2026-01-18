# modules/plot_graph.py
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns
import numpy as np

# 시각화 함수 - Sales
def plot_filtered_sales_by_daily(data, Customer):
    filtered_data = data[data['고객명'].isin(Customer)] 
    filtered_data['SalesDate'] = pd.to_datetime(filtered_data['판매일자'])

    filtered_data['주차'] = filtered_data['SalesDate'].dt.isocalendar().week 
    daily_totals = filtered_data.groupby('주차')['판매량'].sum()

    # Line 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_totals.index, daily_totals.values, marker='o')
    ax.set_title('Weekly Sales Volume Trend')
    ax.set_xlabel('Week')
    ax.set_ylabel('Total Sales Volume')
    fig.autofmt_xdate(rotation=0)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_filtered_sales_by_month(data, Customer):
    df = pd.DataFrame(data)
    df['SalesDate'] = pd.to_datetime(df['판매일자'])    
    df['Month'] = df['SalesDate'].dt.month
    filtered_data = df[df['고객명'].isin(Customer)]    
    daily_totals = filtered_data.groupby('Month')['판매량'].sum()

    # Line 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    #ax.plot(daily_totals.index, daily_totals.values, marker='o')
    ax.bar(daily_totals.index, daily_totals.values)
    ax.set_title('Monthly Sales Volume Trend')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Sales Volume')
    fig.autofmt_xdate(rotation=0)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_filtered_sales_by_item(data, items):    
    filtered_data = data[data['제품코드'].isin(items)]
    sale_counts = filtered_data.groupby('제품 분류')['판매량'].sum()

    # 파이 차트 생성
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sale_counts, labels=sale_counts.index, autopct='%1.1f%%', textprops={'fontsize': 8}, startangle=140)
    #ax.set_title('Sales Data by Customer', fontsize=6)
    plt.tight_layout()
    return fig   

def plot_filtered_sales_by_customer(data, Customer):
    filtered_data = data[data['고객명'].isin(Customer)]
    sale_counts = filtered_data.groupby('고객명')['판매량'].sum()

     # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(sale_counts.index, sale_counts.values)
    ax.set_title('Total Sales Volume by Item Type')
    ax.set_xlabel('Item Type')
    ax.set_ylabel('Total Quantity')    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig
    

def plot_filtered_sales_heatmap(data, customers):
    # 시간 정보 변환
    data['Date'] = pd.to_datetime(data['판매일자'])
    data['DayName_en'] = data['Date'].dt.day_name()
    
    filtered_data = data[data['고객명'].isin(customers)]
    
    heatmap_data = filtered_data.pivot_table(
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
    
    ax.set_title('요일 및 제품별 총 생산량 히트맵', fontsize=15)
    ax.set_xlabel('고객', fontsize=12)
    ax.set_ylabel('요일', fontsize=12)
    #ax.set_xticks(rotation=45, ha='right')
    #ax.set_yticks(rotation=0)
    plt.tight_layout()

    return fig

# 시각화 함수 - Productions
def plot_filtered_production_by_daily(data, machines):
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

def plot_filtered_production_by_month(data, machines):
    df = pd.DataFrame(data)
    df['ProdDate'] = pd.to_datetime(df['생산일자'])    
    df['Month'] = df['ProdDate'].dt.month
    filtered_data = df[df['생산설비'].isin(machines)]
    
    daily_totals = filtered_data.groupby('Month')['생산량'].sum()

    # Line 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(daily_totals.index, daily_totals.values)
    #ax.plot(daily_totals.index, daily_totals.values, marker='o')
    ax.set_title('Monthly Production Volume Trend')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Production Volume')
    fig.autofmt_xdate(rotation=0)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_filtered_production_by_machine(data, machines):
    filtered_data = data[data['생산설비'].isin(machines)]
    sale_counts = filtered_data.groupby('생산설비')['생산량'].sum()
    
    # 파이 차트 생성
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sale_counts, 
        labels=sale_counts.index, 
        autopct='%1.1f%%', 
        startangle=160,
        textprops={'fontsize': 8}
    )
    #ax.set_title('Production Data by Machine', fontsize=6)
    plt.tight_layout()
    return fig
    
def plot_filtered_production_by_item(data, items):    
    filtered_data = data[data['제품코드'].isin(items)]
    sale_counts = filtered_data.groupby('제품코드')['생산량'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(sale_counts.index, sale_counts.values)
    ax.set_title('Total Production Volume by Item Type')
    ax.set_xlabel('Item Name')
    ax.set_ylabel('Total Quantity')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig

def plot_filtered_prod_heatmap(data, machines):
    # 시간 정보 변환
    filtered_data = data[data['생산설비'].isin(machines)]
    
    heatmap_data = filtered_data.pivot_table(
        index='생산설비', 
        columns='제품코드', 
        values='생산량', 
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
        cbar_kws={'label': '총 생산량'} # 컬러바 레이블
        )
    
    ax.set_title('생산설비 및 제품별 총 생산량 히트맵', fontsize=15)
    ax.set_xlabel('제품', fontsize=12)
    ax.set_ylabel('생산설비', fontsize=12)
    #ax.set_xticks(rotation=45, ha='right')
    #ax.set_yticks(rotation=0)
    plt.tight_layout()

    return fig

# 시각화 함수 - Quality
def plot_filtered_quality_by_daily(data, machines):
    filtered_data = data[data['Machine'].isin(machines)]
    
    daily_totals = filtered_data.groupby('Date')['Ng Qty'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_totals.index, daily_totals.values, marker='o')
    ax.set_title('Daily NG Quality Volume Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total NG Quality Volume')
    fig.autofmt_xdate(rotation=45)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_filtered_quality_by_machine(data, machines):
    filtered_data = data[data['Machine'].isin(machines)]
    sale_counts = filtered_data.groupby('Machine')['Ng Qty'].sum()
    
    # 파이 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.pie(sale_counts, labels=sale_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Quality Data by Machine')
    plt.tight_layout()
    return fig

def plot_filtered_quality_by_ngtype(data, machines):
    filtered_data = data[data['Machine'].isin(machines)]
    sale_counts = filtered_data.groupby('Ng Type')['Ng Qty'].sum()
    
    # 파이 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.pie(sale_counts, labels=sale_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Quality Data by NG Type')
    plt.tight_layout()
    return fig
    
def plot_filtered_quality_by_item(data, items):    
    filtered_data = data[data['ItemCode'].isin(items)]
    sale_counts = filtered_data.groupby('ItemCode')['Ng Qty'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(sale_counts.index, sale_counts.values)
    ax.set_title('Total NG Volume by Item Type')
    ax.set_xlabel('Item Type')
    ax.set_ylabel('Total Quantity')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_filtered_quality_heatmap(data, items):
    # 시간 정보 변환
    filtered_data = data[data['ItemCode'].isin(items)]
    
    heatmap_data = filtered_data.pivot_table(
        index='ItemCode', 
        columns='Ng Type', 
        values='Ng Qty', 
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

def plot_filtered_inv_by_month(data, items):
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

def plot_filtered_psi_by_month(data, items):
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

def plot_filtered_inv_by_item(data, items):
    filtered_data = data[data['Item Code'].isin(items)]
    inv_counts = filtered_data.groupby('Item Code')['재고량'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(inv_counts.index, inv_counts.values)
    ax.set_title('Total Inventory Volume', fontsize=15)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Total Quantity', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig

def plot_filtered_inv_by_loc(data, items):
    filtered_data = data[data['Item Code'].isin(items)]
    inv_counts = filtered_data.groupby('Location')['재고량'].sum()

    # 파이 차트 생성
    fig, ax = plt.subplots(figsize=(10, 6))
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
