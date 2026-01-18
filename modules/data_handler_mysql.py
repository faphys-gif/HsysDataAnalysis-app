# modules/data_handler.py
import pandas as pd
from sqlalchemy import create_engine

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

SQL_SALES_DATA = \
        "select date_format(SD02.ship_dt,'%Y-%m-%d') AS '판매일자', " \
        "       week(SD02.ship_dt)+1 AS '주차', " \
        "       CO01.ptnr_name AS '고객', " \
        "       MM01.mat_type3 AS '제품 분류', " \
        "       MM01.mat_name AS '제품', " \
        "      SD02.ship_qty AS '판매량' " \
        "from SD02 " \
        "inner join MM01 on MM01.biz_id = SD02.biz_id and MM01.mat_id = SD02.mat_id and MM01.mat_desc2 = 'AI' " \
        "inner join CO01 on CO01.biz_id = SD02.biz_id and CO01.ptnr_id = SD02.ptnr_id " \
        "where SD02.biz_id = 31 order by MM01.mat_name, SD02.ship_dt "
         
SQL_PROD_DATA = \
        "select date_format(wrk_dt, '%Y-%m-%d') AS '생산일자', " \
	    "week(wrk_dt)+1 AS '주차', " \
        "PP06.wc_code AS '생산설비', " \
        "MM01.mat_code AS '제품', " \
        "sum(PP06.prod_qty) AS '생산량', " \
        "sum(PP06.ng_qty2) AS '불량', " \
        "sum(PP06.wrk_hour) AS '작업시간(분)', " \
        "60*sum(PP06.wrk_hour)/sum(PP06.prod_qty) AS 'Cycle Time(초)' " \
        "from pp06 " \
        "inner join MM01 on MM01.mat_id = PP06.mat_id and MM01.mat_desc2 = 'AI' " \
        "where PP06.biz_id = 31 " \
        "group by week(wrk_dt), PP06.wc_code, MM01.mat_code"

SQL_INV_DATA = \
        "select date_format(std_dt, '%Y-%m-%d') AS 'Date', " \
	    "   date_format(std_dt, '%W') AS 'Day of week', " \
        "   MM01.mat_code AS 'Item Code', " \
        "   LO20.wh_name AS 'Location', " \
        "   sum(LO20.in_qty) AS '생산량', " \
        "   sum(LO20.out_qty) AS '판매량', " \
        "   sum(LO20.inv_qty) AS '재고량' " \
        "from LO20  " \
        "inner join MM01 on MM01.mat_id = LO20.mat_id and MM01.mat_desc2='AI' " \
        "where LO20.biz_id = 31 " \
        "group by LO20.std_dt, LO20.wh_name, MM01.mat_code; " \

def load_dataset():

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
