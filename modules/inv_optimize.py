# modules/item_optimize.py
import pandas as pd
import streamlit as st
import mysql.connector
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import seaborn as sns
import numpy as np

# MySQL 연결 정보
db_info = {
    "user": "faphys",
    "password": "gktkdgns6281",
    "host": "faphysdb.cmkxmjzufbmc.ap-northeast-2.rds.amazonaws.com",
    "port": 3306,
    "database": "faphysdb"
}

SQL_ITEM_MASTER = \
    "select mm01.mat_code AS 'Item Code', mm01.mat_name AS 'Item Name', mm01.mat_type3 AS 'Item Grade', " \
    "    mm01.st_unit AS 'Clustert', mm01.svc_level AS 'Service Level', mm01.rop_qty AS 'ROP', mm01.lot_size AS 'LOT Size', " \
    "    mm01.safty_stock as 'Safety Stock', mm01.lead_time as 'Lead Time(Day)' " \
    "from MM01  " \
    "where MM01.biz_id = 31 and MM01.mat_desc2='AI';" \
    
SQL_ITEM_DATA = \
    "select mm01.mat_code AS 'Item Code', mm01.mat_name AS 'Item Name', mm01.mat_type3 AS 'Item Type'," \
	"   sum(ship_qty) AS 'Total Sales Quantity', " \
    "   count(ship_dt) AS 'Total Sales Count',  " \
    "   avg(ship_qty) AS 'Average Sales Quantity', " \
    "   min(ship_qty) AS 'Minimum Sales Quantity', " \
    "   max(ship_qty) AS 'Maximum Sales Quantity', " \
    "   stddev(ship_qty) AS 'Standard Sales Deviation' " \
    "from SD02  " \
    "inner join MM01 on MM01.biz_id = SD02.biz_id and MM01.mat_id = SD02.mat_id and MM01.mat_desc2='AI' " \
    "where SD02.biz_id = 31 " \
    "group by sd02.mat_id; " \

SQL_SALES_DATA = \
        "select SD02.biz_id AS 'biz_id', SD02.mat_id AS 'mat_id', MM01.mat_code AS 'mat_code', MM01.mat_name AS 'mat_name', " \
        "       SD02.ptnr_id AS 'ptnr_id', CO01.ptnr_code AS 'ptnr_code', CO01.ptnr_name AS 'ptnr_name', " \
        "       concat(date_format(SD02.ship_dt, '%Y-W'), lpad(week(SD02.ship_dt)+1, 2, '0')) AS 'week_id', " \
        "       sum(ship_qty) AS 'sale_qty' " \
        "from SD02 " \
        "inner join MM01 on MM01.mat_id = SD02.mat_id " \
        "inner join CO01 on CO01.ptnr_id = SD02.ptnr_id " \
        "where SD02.biz_id = 31 and MM01.mat_code in ({})  " \
        "group by SD02.mat_id, date_format(SD02.ship_dt, '%Y'), week(SD02.ship_dt); " \

SQL_PSI_DATA = \
        "select mm01.mat_code AS 'Item Code', mm01.mat_name AS 'Item Name', mm01.mat_type3 AS 'Item Type'," \
        "   std_yy AS 'Year', " \
        "   week_num AS 'Week Number',  " \
        "   prod_qty AS 'Prod. Quantity', " \
        "   sale_qty AS 'Sales Quantity', " \
        "   inv_qty AS 'Inventory Quantity', " \
        "   prod_cnt AS 'Prod. Count', " \
        "   sale_cnt AS 'Sales Count' " \
        "from PP80  " \
        "inner join MM01 on MM01.biz_id = PP80.biz_id and MM01.mat_id = PP80.mat_id  " \
        "where PP80.biz_id = 31 and MM01.mat_desc2 = 'AI'; " \
        
SQL_SD61_DATA = \
        "select mat_id AS 'mat_id',  mat_code, mat_name, week_id, sale_qty, Upper_Bound, Lower_Bound, Outlier " \
        "from SD61 " \
        "where SD02.biz_id = 31 and MM01.mat_code in ({})  " \
        "order by week_id ; " \

SQL_MM07_DATA = \
        "select MM07.matgrp_id, MM07.biz_id, MM07.matgrp_name AS 'Cluster Name', " \
        "       MM07.attr_1 AS 'attr_1', MM07.attr_2 AS 'attr_2', MM07.attr_3 AS 'attr_3', MM07.attr_4 AS 'attr_4', " \
        "       MM07.attr_6 AS 'attr_6', MM07.attr_7 AS 'attr_7',  " \
        "       date_format(MM07.crt_dt, '%Y-%m-%d %r') AS '생성일자' " \
        "from MM07 " \
        "where MM07.biz_id = 31;  " \

SQL_PP81_DATA = \
    "select week_num AS 'Week Number',  " \
    "   sum(prod_qty) AS 'Prod. Quantity', " \
    "   sum(sale_qty) AS 'Sales Quantity', " \
    "   sum(inv_qty) AS 'Inventory Quantity', " \
    "   sum(prod_cnt) AS 'Prod. Count', " \
    "   sum(sale_cnt) AS 'Sales Count' " \
    "from PP81  " \
    "where PP81.biz_id = 31 and PP81.data_type = 'S' and PP81.mat_code = '{}'  " \
    "group by week_num  " \
                 
SQL_SIM_DATA = \
    "select co01.ptnr_code, co01.ptnr_name, mm01.mat_code AS 'Item Code', mm01.mat_name AS 'Item Name', mm01.mat_type3 AS 'Item Type'," \
    "   date_format(ship_dt,'%Y-%m-%d') AS 'Sales Date', " \
	"   week(ship_dt)+1 AS 'Sales Week', " \
    "   sum(ship_qty) AS 'Sales Quantity',  " \
    "   sum(ship_qty) + 2000 AS 'Inventory Quantity' " \
    "from SD02  " \
    "inner join MM01 on MM01.biz_id = SD02.biz_id and MM01.mat_id = SD02.mat_id  " \
    "inner join CO01 on CO01.biz_id = SD02.biz_id and CO01.ptnr_id = SD02.ptnr_id  " \
    "where SD02.biz_id = 31 and MM01.mat_desc2 = 'AI' group by sd02.mat_id, date_format(ship_dt,'%Y-%m-%d') "\

def load_dataset_itemmaster():

    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    
    data_items = pd.read_sql(SQL_ITEM_MASTER, con=engine)
    
    return data_items
       
def load_dataset_item():

    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    
    data_items = pd.read_sql(SQL_ITEM_DATA, con=engine)
    
    return data_items

def load_cluster_master(biz_id):
    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    
    df = pd.read_sql(SQL_MM07_DATA, con=engine)
    
    return df

def load_dataset_psi():

    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    
    data_psi = pd.read_sql(SQL_PSI_DATA, con=engine)
    
    return data_psi

def item_classifier(data_items):
    df = pd.DataFrame(data_items)
    
    # 특징만 추출
    features = df[['Total Sales Count', 'Average Sales Quantity', 'Standard Sales Deviation']]    
    
    # 스케일링
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # K-Means (클러스터 수=2 예시)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    return df

def item_simulation():
    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    
    df = pd.DataFrame(pd.read_sql(SQL_SIM_DATA, con=engine))
    return df

def plot_filtered_prod(data, items):    
    filtered_data = data[data['Item Code'].isin(items)]
    prod_qty = filtered_data.groupby('Week Number')['Prod. Quantity'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(prod_qty.index, prod_qty.values)
    ax.set_title('Total Production Volume by Weekly')
    ax.set_xlabel('Week Number')
    ax.set_ylabel('Total Production Quantity')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_filtered_sales(data, items):    
    filtered_data = data[data['Item Code'].isin(items)]
    sales_qty = filtered_data.groupby('Week Number')['Sales Quantity'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sales_qty.index, sales_qty.values)
    ax.set_title('Total Sales Volume by Weekly')
    ax.set_xlabel('Week Number')
    ax.set_ylabel('Total Sales Quantity')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_filtered_inv(data, items):    
    filtered_data = data[data['Item Code'].isin(items)]
    inv_qty = filtered_data.groupby('Week Number')['Inventory Quantity'].sum()

    # Bar 차트 생성
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(inv_qty.index, inv_qty.values)
    ax.set_title('Total Inventory Volume by Weekly')
    ax.set_xlabel('Week Number')
    ax.set_ylabel('Total Inventory Quantity')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_filtered_psi(data, items):
    filtered_data = data[data['Item Code'].isin(items)]
    
    prod_counts = filtered_data.groupby('Week Number')['Prod. Quantity'].sum()
    sale_counts = filtered_data.groupby('Week Number')['Sales Quantity'].sum()
    inv_counts = filtered_data.groupby('Week Number')['Inventory Quantity'].sum()

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
    rects2 = ax.bar(x, combined_df['Sales'], width, label='판매량 (Sales)', color='green')    
    # 재고량 막대: 중앙에서 왼쪽으로 이동
    rects3 = ax.bar(x + width, combined_df['Inventory'], width, label='재고량 (Inventory)', color='navy')
        
    # 5. 그래프 꾸미기
    ax.set_title('월별 총 상산량, 판매량 및 재고량 비교', fontsize=15)
    ax.set_xlabel('Week NUmber', fontsize=12)
    ax.set_ylabel('수량 (Quantity)', fontsize=12)
    
    ax.set_xticks(x)        # X축 눈금 위치 설정
    ax.set_xticklabels(labels, rotation=0, ha='right') # X축 레이블 설정 및 회전
    
    ax.legend(loc='upper right') # 범례 추가
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_filtered_items_by_Count(data, Cluster):
    data['Cluster Name'] = 'Cluster-' + data['Cluster'].astype(str)
    filtered_data = data[data['Cluster'].isin(Cluster)]    
    daily_totals = filtered_data.groupby('Cluster Name')['Item Code'].size()

    # Line 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(daily_totals.index, daily_totals.values)
    ax.set_title('Items by Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Items')
    fig.autofmt_xdate(rotation=0)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_filtered_items_by_Cluster(data, Cluster):
    data['Cluster Name'] = 'Cluster-' + data['Cluster'].astype(str)
    filtered_data = data[data['Cluster'].isin(Cluster)]    
    daily_totals = filtered_data.groupby('Cluster Name')['Average Sales Quantity'].sum()

    # Line 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(daily_totals.index, daily_totals.values)
    ax.set_title('Weekly Sales Volume by Clusters')
    ax.set_xlabel('Week')
    ax.set_ylabel('Total Sales Volume')
    fig.autofmt_xdate(rotation=0)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_filtered_items_by_Cluster2(data, Cluster):
    filtered_data = data[data['Cluster'].isin(Cluster)]  
    colors = {0:'blue', 1:'red', 2:'green', 3:'black', 4:'purple'}
    
    # Scatter 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in filtered_data.groupby(['Cluster']):
        cluster_id = name[0]
        ax.scatter(
            group['Average Sales Quantity'], 
            group['Standard Sales Deviation'], 
            color=colors.get(cluster_id,'black'), 
            label=f'Cluster{name}', # 범례에 표시될 레이블
            alpha=0.6,
            s=50
        )
            
    #ax.scatter(x, y)
    ax.set_title('Cluster별 평균 판매량 및 표준 편차 산점도')
    ax.set_xlabel('평균 판매량 (Average Sales Quantity)')
    ax.set_ylabel('판매량 표준 편차 (Standard Sales Deviation)')

    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left') # 범례 위치 조절 (선택 사항)
    ax.grid(True, linestyle='--', alpha=0.7) # 그리드 추가 (선택 사항)
    plt.tight_layout()
    return fig

def plot_filtered_items_by_Cluster3(data, Cluster):
    filtered_data = data[data['Cluster'].isin(Cluster)]  
    colors = {0:'blue', 1:'red', 2:'green', 3:'black', 4:'purple'}
    
    # Scatter 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in filtered_data.groupby(['Cluster']):
        cluster_id = name[0]
        ax.scatter(
            group['Average Sales Quantity'], 
            group['Total Sales Count'], 
            color=colors.get(cluster_id,'black'), 
            label=f'Cluster{name}', # 범례에 표시될 레이블
            alpha=0.6,
            s=50
        )
            
    #ax.scatter(x, y)
    ax.set_title('Cluster별 평균 판매량 및 판매회수 산점도')
    ax.set_xlabel('평균 판매량 (Average Sales Quantity)')
    ax.set_ylabel('총 판매 회수 (Total Sales Count)')

    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left') # 범례 위치 조절 (선택 사항)
    ax.grid(True, linestyle='--', alpha=0.7) # 그리드 추가 (선택 사항)
    plt.tight_layout()
    return fig

def plot_filtered_items_by_Cluster4(data, Cluster):
    filtered_data = data[data['Cluster'].isin(Cluster)]  
    colors = {0:'blue', 1:'red', 2:'green', 3:'black', 4:'purple'}
    
    # Scatter 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in filtered_data.groupby(['Cluster']):
        cluster_id = name[0]
        ax.scatter(
            group['Standard Deviation(Norm.)'], 
            group['Total Sales Count'], 
            color=colors.get(cluster_id,'black'), 
            label=f'Cluster{name}', # 범례에 표시될 레이블
            alpha=0.6,
            s=50
        )
            
    #ax.scatter(x, y)
    ax.set_title('Cluster별 판매량 편차 및 판매회수 산점도')
    ax.set_xlabel('판매량 편차 (Standard Deviation(Norm.))')
    ax.set_ylabel('총 판매 회수 (Total Sales Count)')

    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left') # 범례 위치 조절 (선택 사항)
    ax.grid(True, linestyle='--', alpha=0.7) # 그리드 추가 (선택 사항)
    plt.tight_layout()
    return fig

def plot_filtered_in_heatmap(data, clusters):
    data['Cluster Name'] = 'Cluster-' + data['Cluster'].astype(str)
    filtered_data = data[data['Cluster'].isin(clusters)]
    
    heatmap_data = filtered_data.pivot_table(
        index='Item Type', 
        columns='Cluster Name', 
        values='Item Code', 
        aggfunc='size',
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
        cbar_kws={'label': 'Match'} # 컬러바 레이블
        )
    
    ax.set_title('Item Type 및 Cluster 히트맵', fontsize=15)
    ax.set_ylabel('Item Type(자체)', fontsize=12)
    ax.set_xlabel('Cluster(AI)', fontsize=12)
    #ax.set_xticks(rotation=45, ha='right')
    #ax.set_yticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_filtered_sim_by_item_daily(data, item):
    filtered_data = data[data['Item Code'] == item] 
    daily_totals = filtered_data.groupby('Sales Date')['Sales Quantity'].sum()

    # Line 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    tick_spacing = 7 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.plot(daily_totals.index, daily_totals.values, marker='o')
    ax.set_title('Daily Sales Volume Trend Daily', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Sales Volume')
    fig.autofmt_xdate(rotation=45)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_filtered_sim_by_item(data, item):
    filtered_data = data[data['Item Code'] == item]    
    daily_totals = filtered_data.groupby('Sales Week')['Sales Quantity'].sum()

    # Line 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_totals.index, daily_totals.values, marker='o')
    ax.set_title('Daily Sales Volume Trend Weekly', fontsize=16)
    ax.set_xlabel('Week')
    ax.set_ylabel('Total Sales Volume')
    fig.autofmt_xdate(rotation=0)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig


def load_dataset_sales(mat_code):
    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )

    quoted_items = [f"'{item}'" for item in mat_code]
    # 2. 쉼표와 공백(', ')으로 항목들을 연결합니다.
    output_string = ', '.join(quoted_items)

    formatted_sql = SQL_SALES_DATA.format(output_string)
    data_sales = pd.read_sql(formatted_sql, con=engine)
    
    return data_sales

def clear_db_results(biz_id, mat_code):
    # 1. Establish the connection
    try:
        conn = mysql.connector.connect(
            host="faphysdb.cmkxmjzufbmc.ap-northeast-2.rds.amazonaws.com",
            user="faphys",
            password="gktkdgns6281",
            database="faphysdb"
        )

        # 2. Create a cursor object
        cursor = conn.cursor()

        quoted_items = [f"'{item}'" for item in mat_code]
        # 2. 쉼표와 공백(', ')으로 항목들을 연결합니다.
        output_string = ', '.join(quoted_items)

        # The value for the record to delete
        biz_id_to_delete = biz_id
        mat_code_to_delete = output_string

         # 3. Define the SQL DELETE FROM query with a placeholder (%s)
        sql_query = "DELETE FROM SD61 WHERE biz_id = %s and mat_code in (%s)"

        # The data must be passed as a tuple, even for a single value
        delete_value = (biz_id_to_delete, mat_code_to_delete,)

        # 4. Execute the parameterized query
        cursor.execute(sql_query, delete_value)

        # 5. Commit your changes
        conn.commit()

        # Check how many rows were affected
        #st.write(f"{cursor.rowcount} record(s) deleted.")

    except mysql.connector.Error as err:
        st.write(f"Error: {err}")
        # Roll back in case of error
        if 'conn' in locals() and conn.is_connected():
            conn.rollback()

    finally:
        # 6. Close the cursor and connection
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            #st.write("MySQL connection closed.")

def write_result_ma(df):
    table_name = 'sd61'

     # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='append',  # Options: 'fail', 'replace', 'append'
        index=False         # Do not write the DataFrame's index as a column
    )

def anormaly_ma(df, GRAPH_VIEW):
    #df['Issue_Date'] = pd.to_datetime(df['Issue_Date'])
    df = df.sort_values('week_id').reset_index(drop=True)

    # 설정 변수
    WINDOW_SIZE = 8  # 이동 평균 계산을 위한 윈도우 크기 (예: 7일)
    THRESHOLD_STD = 2 # 이상치를 정의하는 표준 편차 임계값 (예: 2 표준편차)

    # 이동평균 및 표준편차 계산
    window = 8
    df['MA'] = df['sale_qty'].rolling(window=window, center=False).mean()
    df['STD'] = df['sale_qty'].rolling(window=window, center=False).std()

    # 상하 경계 계산
    df['Upper_Bound'] = df['MA'] + 2 * df['STD']
    df['Lower_Bound'] = df['MA'] - 2 * df['STD']

    # 이상치 판별
    df['Outlier'] = (df['sale_qty'] > df['Upper_Bound']) | (df['sale_qty'] < df['Lower_Bound'])

    print("Anomaly Detection - Moving Average")
    print(df[df['Outlier']])

    write_result_ma(df)
    # 이상치만 추출
    outliers = df[df['Outlier']]
    # --- 4. 그래프 시각화 ---
    plt.figure(figsize=(14, 7))

    # 원본 데이터 Plot
    plt.plot(df.index, df['sale_qty'], label='Original Data', color='skyblue', linewidth=1.5, alpha=0.6)

    # 이동 평균선 Plot
    plt.plot(df.index, df['MA'], label=f'{WINDOW_SIZE}-Day Moving Average', color='blue', linewidth=2)

    # 상/하한선 (경계) Plot
    plt.plot(df.index, df['Upper_Bound'], label=f'{THRESHOLD_STD} STD Upper Bound', color='red', linestyle='--', linewidth=1)
    plt.plot(df.index, df['Lower_Bound'], label=f'{THRESHOLD_STD} STD Lower Bound', color='red', linestyle='--', linewidth=1)

    # 이상치(Outlier) 하이라이트
    #plt.scatter(df.index, df['Outlier'], label='Outliers', color='red', s=50, zorder=5)
    plt.scatter(outliers['week_id'], outliers['sale_qty'], color='red', label='Outliers', s=50, zorder=5)
    # 그래프 제목 및 레이블 설정
    plt.title(f'Outlier Detection using Moving Average ({WINDOW_SIZE}-Week, $\pm {THRESHOLD_STD}\sigma$)', fontsize=16)
    plt.xlabel('week_id')
    plt.ylabel('sale_qty')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()

    # 그래프 저장
    plt.savefig('img\moving_average_outliers.png')
    #st.write("Graph saved as moving_average_outliers.png") 

def plot_filtered_psi_actual(sel_items):
    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    
    
    data = pd.read_sql(SQL_PSI_DATA, con=engine)
    
    filtered_data = data[data['Item Code'] == sel_items]
    
    prod_counts = filtered_data.groupby('Week Number')['Prod. Quantity'].sum()
    sale_counts = filtered_data.groupby('Week Number')['Sales Quantity'].sum()
    inv_counts = filtered_data.groupby('Week Number')['Inventory Quantity'].sum()

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
    rects2 = ax.bar(x, combined_df['Sales'], width, label='판매량 (Sales)', color='green')    
    # 재고량 막대: 중앙에서 왼쪽으로 이동
    rects3 = ax.plot(x + width/2, combined_df['Inventory'], label='재고량 (Inventory)', marker='o', linestyle='-', linewidth=2, zorder=3, color='navy')
        
    # 5. 그래프 꾸미기
    ax.set_title('월별 총 상산량, 판매량 및 재고량 비교', fontsize=15)
    ax.set_xlabel('Week NUmber', fontsize=12)
    ax.set_ylabel('수량 (Quantity)', fontsize=12)
    
    ax.set_xticks(x)        # X축 눈금 위치 설정
    ax.set_xticklabels(labels, rotation=0, ha='right') # X축 레이블 설정 및 회전
    
    ax.legend(loc='upper right') # 범례 추가
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def prc_pp81_simulation(biz_id, mat_code):
    conn = mysql.connector.connect(
        host="faphysdb.cmkxmjzufbmc.ap-northeast-2.rds.amazonaws.com",
        user="faphys",
        password="gktkdgns6281",
        database="faphysdb"
    )
    cursor = conn.cursor(dictionary=True)

    try:
        # 프로시저 호출
        cursor.callproc('prc_pp81_Simumation', [biz_id, mat_code])
        conn.commit()
    except mysql.connector.Error as err:
        conn.rollback()

    finally:
        cursor.close()
        conn.close()

def plot_filtered_psi_simulation(sel_items):
    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    
    prc_pp81_simulation(31, sel_items)
    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    
    formatted_sql = SQL_PP81_DATA.format(sel_items)
    data = pd.read_sql(formatted_sql, con=engine)
        
    prod_counts = data.groupby('Week Number')['Prod. Quantity'].sum()
    sale_counts = data.groupby('Week Number')['Sales Quantity'].sum()
    inv_counts = data.groupby('Week Number')['Inventory Quantity'].sum()

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
    rects2 = ax.bar(x, combined_df['Sales'], width, label='판매량 (Sales)', color='green')    
    # 재고량 막대: 중앙에서 왼쪽으로 이동
    rects3 = ax.plot(x + width/2, combined_df['Inventory'], label='재고량 (Inventory)', marker='o', linestyle='-', linewidth=2, zorder=3, color='navy')
        
    # 5. 그래프 꾸미기
    ax.set_title('월별 총 상산량, 판매량 및 재고량 비교', fontsize=15)
    ax.set_xlabel('Week NUmber', fontsize=12)
    ax.set_ylabel('수량 (Quantity)', fontsize=12)
    
    ax.set_xticks(x)        # X축 눈금 위치 설정
    ax.set_xticklabels(labels, rotation=0, ha='right') # X축 레이블 설정 및 회전
    
    ax.legend(loc='upper right') # 범례 추가
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig
