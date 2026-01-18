# hsys/data_handler.py
import streamlit as st
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

# MySQL 연결 정보
db_info = {
    "user": "faphys",
    "password": "gktkdgns6281",
    "host": "faphysdb.cmkxmjzufbmc.ap-northeast-2.rds.amazonaws.com",
    "port": 3306,
    "database": "faphysdb"
}

SQL_STMT_TEXT = \
        "select sql_stmt from sys53 where biz_id = %s  and dataset_id = %s " \
        
def load_dataset(bizId, dsCode):

    try:
        conn = mysql.connector.connect(
            host="faphysdb.cmkxmjzufbmc.ap-northeast-2.rds.amazonaws.com",
            user="faphys",
            password="gktkdgns6281",
            database="faphysdb"
        )

        # 2. Create a cursor object
        cursor = conn.cursor()
        query = "select sql_stmt from sys53 where biz_id = %s  and dataset_code = %s " 
        cursor.execute(query, (bizId, dsCode,))

        sql_stmts = cursor.fetchone()

        # 4. 변수에 할당
        if sql_stmts:
            sql_stmt = sql_stmts[0]  # fetchone()은 튜플 형태로 반환하므로 인덱스로 접근
            #st.write(f"sql_stmt: {sql_stmt}")

            data_sales = pd.read_sql(sql_stmt, con=conn)
    
        else:
            st.write(f"해당하는 데이터셋이 없습니다.")
            data_sales = []

        cursor.close()
        conn.close()

        return data_sales

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

    
def load_dataset_old():

    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )  
    
    data_sales = pd.read_sql(SQL_SALES_DATA, con=engine)
    data_production = pd.read_sql(SQL_PROD_DATA, con=engine)
    #data_production = pd.read_csv('data/production.csv')
    data_quality = pd.read_csv('data/quality.csv')
    data_purchasing = pd.read_csv('data/purchasing.csv')
    data_inventory = pd.read_sql(SQL_INV_DATA, con=engine)
     #data_inventory = pd.read_csv('data/inventory.csv')

    table = pd.DataFrame({'Counts': [data_sales.shape[0], data_sales.shape[1]]}, index=['Rows', 'Columns'])
    return data_sales, data_production, data_quality, data_purchasing, data_inventory

def load_dataset_sales():

    # SQLAlchemy 엔진 생성
    engine = create_engine(
        f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
        f"{db_info['host']}:{db_info['port']}/{db_info['database']}?charset=utf8mb4"
    )
    data_sales = pd.read_sql(SQL_SALES_DATA, con=engine)
    
    return data_sales