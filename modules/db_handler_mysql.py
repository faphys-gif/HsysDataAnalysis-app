# modules/db_handler_mysql.py
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

SQL_SD02_DATA = \
    "select mm01.mat_code AS 'Item Code', mm01.mat_name AS 'Item Name', mm01.mat_type3 AS 'Item Type'," \
	"   sum(ship_qty) AS 'Total Sales Quantity', " \
    "   count(week_id) AS 'Total Sales Count',  " \
    "   round(avg(ship_qty)) AS 'Average Sales Quantity', " \
    "   min(ship_qty) AS 'Minimum Sales Quantity', " \
    "   max(ship_qty) AS 'Maximum Sales Quantity', " \
    "   round(stddev(ship_qty)) AS 'Standard Sales Deviation', " \
    "   round(stddev(ship_qty)/avg(ship_qty),2) AS 'Standard Deviation(Norm.)'  " \
    "from (select mat_id, week(ship_dt)+1 as week_id, sum(ship_qty) as ship_qty, count(ship_qty) as ship_cnt " \
    "       from sd02 where biz_id = 31 and mat_id in (select mm01.mat_id from mm01 where mm01.biz_id = 31 and mm01.mat_desc2='AI') " \
    "       group by mat_id, week(ship_dt)) AS SD02  " \
    "inner join MM01 on MM01.mat_id = SD02.mat_id " \
    "group by sd02.mat_id; " \

DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_HOST = st.secrets["DB_HOST"]
DB_DATABASE = st.secrets["DB_DATABASE"]

# MySQL 연결 정보
db_info = {
    "user": DB_USER,
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": 3306,
    "database": DB_DATABASE
}

def connect_to_mysql():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_DATABASE
        )

        # 2. Create a cursor object
        cursor = conn.cursor()
        return conn

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

def update_data_in_mm17(updates):
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_DATABASE
        )

        # 2. Create a cursor object
        cursor = conn.cursor()

        for index, row in updates.iterrows():
            # 예시: UPDATE inventory_table SET Stock = 999 WHERE ID = 1
            #st.success(f"DB 업데이트 실행 (ID: {row['trx_id']}, Attr_1: {row['attr_1']}, Attr_2: {row['attr_2']})")
            cursor.execute("UPDATE MM17 SET svc_level = %s WHERE trx_id = %s", (row['Service Level'], row['trx_id']))
            
        st.success(f"총 {len(updates)}건의 데이터가 데이터베이스에 반영되었습니다.")
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

def clear_db_results(biz_id, mat_code):
    # 1. Establish the connection
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_DATABASE
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

def prc_inv_optimize_in_mm16(biz_id):
    conn = mysql.connector.connect(
		host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_DATABASE
    )
    cursor = conn.cursor(dictionary=True)

    try:
        # 프로시저 호출
        cursor.callproc('prc_mm16_optimizeInventory', [biz_id])
        conn.commit()
    except mysql.connector.Error as err:
        conn.rollback()

    finally:
        cursor.close()
        conn.close()

def load_dataset_sales_weekly(biz_id):
   # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    df = pd.read_sql(SQL_SD02_DATA, con=engine)
    
    return df

def update_item_cluster(df):
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_DATABASE
        )

        # 2. Create a cursor object
        cursor = conn.cursor()

        for index, row in df.iterrows():
            # 예시: UPDATE inventory_table SET Stock = 999 WHERE ID = 1
            # st.success(f"DB 업데이트 실행 (Item Code: {row['Item Code']}, Cluster: {row['Cluster']})")
            cursor.execute("UPDATE MM01 SET wh_code = %s WHERE biz_id = 31 and mat_code = %s", (row['Cluster'], row['Item Code']))
            
        cursor.callproc('prc_mm17_createClusterInfo', [31])
        
        st.success(f"총 {len(df)}건의 데이터가 데이터베이스에 반영되었습니다.")
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

def load_dataset_mm16_stat(biz_id):
    SQL_STATMENT = \
        "select MM16.trx_id, MM16.mat_id, MM16.mat_code AS 'Item Code', MM16.mat_name AS 'Item Name', " \
        "       MM16.mat_type3 AS 'Item Type(Manual)', MM16.clust_code AS 'Cluster', MM16.svc_level AS 'Service Level', " \
        "       MM16.safety_stock AS 'Safety Stock',  MM16.lead_time AS 'Lead Time', MM16.rop_qty AS 'Re-order Qty', " \
        "       MM16.lot_size AS 'Lot Size', date_format(MM16.crt_dt,'%Y-%m-%d %r') AS '생성일자' " \
        "from MM16 " \
        "where MM16.biz_id = 31;  " \
        
     # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    df = pd.read_sql(SQL_STATMENT, con=engine)
    
    return df

def load_dataset_mm17_stat(biz_id):
    SQL_STATMENT = \
        "select MM17.trx_id, MM17.biz_id, MM17.cluster_code AS 'Cluster Code', MM17.cluster_name AS 'Cluster Name', " \
        "       MM17.num_of_item AS 'Num. Of Items', MM17.avg_qty AS 'Avg. Sales Qty', MM17.avg_cnt AS 'Avg. Sales Count', " \
        "       round(MM17.avg_dev,2) AS 'Std. Deviation',  MM17.svc_level AS 'Service Level', date_format(crt_dt,'%Y-%m-%d %r') AS '생성일자' " \
        "from MM17 " \
        "where MM17.biz_id = 31;  " \
        
     # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    
    df = pd.read_sql(SQL_STATMENT, con=engine)
    
    return df


