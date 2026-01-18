import os
import io
import streamlit as st
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import plotly.express as px
from PyPDF2 import PdfReader

import hsys.data_handler as dh
import hsys.plot_graph as pg
import modules.inv_optimize as invopt
import modules.db_handler_mysql as db_mysql

import google.generativeai as genai 
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

# LangChain 관련
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- 환경변수 로드 (선택) ---
from dotenv import load_dotenv

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

biz_id = 31

st.set_page_config(page_title="AI Chatbot", layout="wide")
font_path = "Fonts/NANUMGOTHIC.TTF"  # 맑은 고딕
#font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" #Ubuntu

# 폰트 이름 등록
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# ✅ ③ 음수(-) 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

st.markdown("""
    <style>   
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: 100;
        width: 80%;
        background: #fff;
        border-top: 1px solid #ddd;
        padding: 10px 20px;
        z-index: 1000;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_gemini():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",     # gemini-2.0-pro 도 가능
        temperature=0.7,
        google_api_key=GEMINI_API_KEY
    )
    return llm

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key=GEMINI_API_KEY
    )

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    
# Streamlit 앱
def main():
    st.set_page_config(page_title="Manufacturing Data Analysis with AI", layout="wide")
    #st.title("Manufacturing Data Analysis with AI")
    
    # ------------------------------
    # 🔹 Sidebar Menu 생성
    # ------------------------------
    st.sidebar.title("📊AI Dashboard")
    st.sidebar.markdown("## 📋 메뉴")
    
    plt.rcParams.update({
        'font.size': 12,         # 기본 폰트 크기
        'axes.titlesize': 10,    # 제목 크기
        'axes.labelsize': 10,    # 축 라벨 크기
        'legend.fontsize': 10,   # 범례 크기
    })
    
    biz_id = st.sidebar.text_input("Biz ID를 입력하세요:", placeholder="예: 31")

    menu = st.sidebar.selectbox(
        "원하는 기능을 선택하세요", 
        ["🏠홈", "📊제조 데이터 분석",  "🧭재고최적화", "🧠데이터 분석(Q&A)", "🤖 AI 챗봇","문서파일 분석"]
    )
    if menu == "🏠홈":
        st.title("🏠 홈 화면")
        st.write("AI 데이터 분석 시스템에 오신 것을 환영합니다.")
        
        with open('modules/ai_intro.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            st.write(content)
        
       
    elif menu == "📊제조 데이터 분석":
        st.title("📈제조 데이터 분석")
        st.write("이곳에서 다양한 제조 데이터 분석을 수행합니다.")

        # 데이터 로드
        #data_sales,  data_production, data_quality, data_purchasing, data_inventory = dh.load_dataset()
    
        # 탭으로 그래프 선택
        st.write("### Explore Manufacturing Data")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Sales", "Productions", "Quality", "Purchasing", "Inventory"]
        )

        with tab1:
            st.write("#### Sales Data")
            
            if st.button("[판매]데이터 불러오기"):                                        
                data_sales = dh.load_dataset(biz_id, 'DS_SALES') 

                customers = data_sales['고객명'].unique()
                items = data_sales['제품코드'].unique()
            
                if not customers.any():
                    st.error("Please select at least one Customer.")
                else:
                    df = pd.DataFrame(data_sales)
                    st.dataframe(df.head(30))
        
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### 주별 판매량 추이")
                    filtered_sales_fig_by_daily = pg.plot_filtered_sales_by_daily(data_sales, customers)
                    st.pyplot(filtered_sales_fig_by_daily)
                    
                    st.write("#### 월별 판매량")
                    filtered_sales_fig_by_month = pg.plot_filtered_sales_by_month(data_sales, customers)
                    st.pyplot(filtered_sales_fig_by_month)   
                    
                    st.write("#### 요일별 판매량")
                    filtered_sales_fig_by_weekday = pg.plot_filtered_sales_heatmap(data_sales, customers)
                    st.pyplot(filtered_sales_fig_by_weekday)

                with col2:
                    st.write("#### 고객별 판매량")
                    filtered_sales_fig_by_item = pg.plot_filtered_sales_by_customer(data_sales, customers)
                    st.pyplot(filtered_sales_fig_by_item)

                    st.write("#### 제품 그룹별 판매 비중")
                    filtered_sales_fig_by_customer = pg.plot_filtered_sales_by_item(data_sales, items)
                    st.pyplot(filtered_sales_fig_by_customer)               
            
        with tab2:
            st.write("#### Production Data")
        
            if st.button("[생산]데이터 불러오기"):                                        
                data_prods = dh.load_dataset(biz_id, 'DS_PRODS') 

                machines = data_prods['생산설비'].unique()
                items = data_prods['제품코드'].unique()
            
            # machines = st.multiselect(
            #     "Choose Machines",
            #     options=data_production['생산설비'].unique(),
            #     default=data_production['생산설비'].unique()
            #     )
        
            # items = st.multiselect(
            #     "Choose Items",
            #     options=data_production['제품'].unique(),
            #     default=data_production['제품'].unique()
            #     )
        
                if not machines.any():
                    st.error("Please select at least one Machine.")
                else:
                    df_prod = pd.DataFrame(data_prods)
                st.dataframe(df_prod.head(30))
       
                col3, col4 = st.columns(2)
            
                with col3:
                    st.write("#### 주차별 생산량")
                    filtered_production_fig_by_daily = pg.plot_filtered_production_by_daily(data_prods, machines)
                    st.pyplot(filtered_production_fig_by_daily)
                
                    st.write("#### 월별 생산량")
                    filtered_production_fig_by_month = pg.plot_filtered_production_by_month(data_prods, machines)
                    st.pyplot(filtered_production_fig_by_month)
                    
                    st.write("#### Heatmap By Machines")
                    filtered_production_fig_by_mc = pg.plot_filtered_prod_heatmap(data_prods, machines)
                    st.pyplot(filtered_production_fig_by_mc)
                    
                with col4:
                    st.write("#### 제품별 생산량")
                    filtered_production_fig_by_item = pg.plot_filtered_production_by_item(data_prods, items)
                    st.pyplot(filtered_production_fig_by_item)
                    
                    st.write("#### 생산설비별 생산 비중")
                    filtered_production_fig_by_machine = pg.plot_filtered_production_by_machine(data_prods, machines)
                    st.pyplot(filtered_production_fig_by_machine)
    
        with tab3:
            st.write("#### 품질 데이터 분석")
        
            if st.button("[품질]데이터 불러오기"):                                        
                data_quality = dh.load_dataset(biz_id, 'DS_QCS') 

                machines = st.multiselect(
                    "Choose Machines",
                    options=data_quality['Machine'].unique(),
                    default=data_quality['Machine'].unique()
                    )
            
                items = st.multiselect(
                    "Choose Items[2]",
                    options=data_quality['ItemCode'].unique(),
                    default=data_quality['ItemCode'].unique()
                    )
            
                if not machines:
                    st.error("Please select at least one Machine.")
                else:
                    df_qc = pd.DataFrame(data_quality)
                    st.dataframe(df_qc.head())
        
                col5, col6 = st.columns(2)
                
                with col5:
                    st.write("#### 일별 불량 발생 추이")
                    filtered_quality_fig_by_daily = pg.plot_filtered_quality_by_daily(data_quality, machines)
                    st.pyplot(filtered_quality_fig_by_daily)
                
                    st.write("#### 품목별 불량 수량")
                    filtered_quality_fig_by_item = pg.plot_filtered_quality_by_item(data_quality, items)
                    st.pyplot(filtered_quality_fig_by_item)

                    st.write("#### Heatmap By Items and Ng Type")
                    filtered_quality_fig_by_mc = pg.plot_filtered_quality_heatmap(data_quality, items)
                    st.pyplot(filtered_quality_fig_by_mc)

                with col6:
                    st.write("#### Quality By NG Type")
                    filtered_quality_fig_by_ngtype = pg.plot_filtered_quality_by_ngtype(data_quality, machines)
                    st.pyplot(filtered_quality_fig_by_ngtype)
                                
                    st.write("#### Quality By Machines")
                    filtered_quality_fig_by_machine = pg.plot_filtered_quality_by_machine(data_quality, machines)
                    st.pyplot(filtered_quality_fig_by_machine)
                            
        with tab4:
            st.write("#### Purchaing Data")

            if st.button("[구매]데이터 불러오기"):                                        
                data_purchasing = dh.load_dataset(biz_id, 'DS_POS') 

                df_pur = pd.DataFrame(data_purchasing)
                st.dataframe(df_pur.head())

        with tab5:
            st.write("#### Inventory Data")

            if st.button("[재고]데이터 불러오기"):                                        
                data_inventory = dh.load_dataset(biz_id, 'DS_INVS') 

                items = data_inventory['Item Code'].unique()
            # items = st.multiselect(
            #     "Choose Items",
            #     options=data_inventory['Item Code'].unique(),
            #     default=data_inventory['Item Code'].unique()
            #     )
        
                if not items.any():
                    st.error("Please select at least one Item Code.")
                else:
                    df_inv = pd.DataFrame(data_inventory)
                    st.dataframe(df_inv.head())
            
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("#### 월별 재고량(Monthly Inventory)")
                        filtered_inv_fig_by_month = pg.plot_filtered_inv_by_month(data_inventory, items)
                        st.pyplot(filtered_inv_fig_by_month)
                    
                        st.write("#### 월별 PSI(생산-판매-재고)")
                        filtered_psi_fig_by_month = pg.plot_filtered_psi_by_month(data_inventory, items)
                        st.pyplot(filtered_psi_fig_by_month)
                        
                        sel_item = st.multiselect(
                        "Choose Items",
                        options=data_inventory['Item Code'].unique()
                        )

                        st.write("#### 일별 PSI(생산-판매-재고)")
                        filtered_psi_fig_by_daily = pg.plot_filtered_psi_by_daily(data_inventory, sel_item)
                        st.pyplot(filtered_psi_fig_by_daily)

                    with col2:
                        st.write("#### 제품별 재고 비중(Inventory ratio by Product)")
                        filtered_inv_fig_by_item = pg.plot_filtered_inv_by_item(data_inventory, items)
                        st.pyplot(filtered_inv_fig_by_item)

                        st.write("#### Location별 재고 비중(Inventory ratio by Location)")
                        filtered_inv_fig_by_loc = pg.plot_filtered_inv_by_loc(data_inventory, items)
                        st.pyplot(filtered_inv_fig_by_loc)

                
    elif menu == "🧠데이터 분석(Q&A)":
        st.title("🧠 AI Data Analysis with Google's Gemini-Bot")

        st.write("데이터 파일(CSV)을 업로드하고, Gemini에게 데이터를 분석하도록 해보세요!")

        # ==============================    
        # 🔹 파일 업로드
        # ==============================
        uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

        if uploaded_file:
            # 데이터 로드
            df_file = pd.read_csv(uploaded_file)
            st.subheader("📊 업로드한 데이터 미리보기")
            st.dataframe(df_file.head())

            # ==============================
            # 🔹 Gemini를 통한 데이터 요약
            # ==============================
            st.subheader("🧠 Gemini 데이터 요약")
            csv_buffer = io.StringIO()
            df_file.to_csv(csv_buffer, index=False)
            csv_text = csv_buffer.getvalue()

            with st.spinner("Gemini가 데이터를 분석 중입니다... ⏳"):
                prompt = f"""
                    다음은 CSV 데이터입니다. 주요 패턴, 통계 요약, 이상값, 추세를 한국어로 요약해줘:
                    \n\n{csv_text[:5000]}  # 너무 큰 경우 일부만 보냄
                    """
                model = genai.GenerativeModel("gemini-2.0-flash")
                summary = model.generate_content(prompt)
            st.success("✅ 분석 완료!")
            st.write(summary.text)

            # ==============================
            # 🔹 그래프 생성
            # ==============================
            st.subheader("📈 기술 통계 분석 도구")
            # 사용자 입력 받기
            column = st.selectbox('분석할 컬럼을 선택하세요', df_file.columns)
            st.write(df_file[column].describe())

            # ==============================
            # 🔹 그래프 생성
            # ==============================
            st.subheader("📈 데이터 시각화")

            numeric_cols = df_file.select_dtypes(include=["number"]).columns.tolist()
            all_cols = df_file.columns.tolist()

            x_axis = st.selectbox("X축 선택", all_cols)
            y_axis = st.selectbox("Y축 선택", numeric_cols)
            chart_type = st.radio("그래프 종류", ["Line", "Bar", "Scatter"], horizontal=True)

            if st.button("그래프 생성"):
                if chart_type == "Line":
                    fig = px.line(df_file, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                elif chart_type == "Bar":
                    fig = px.bar(df_file, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                else:
                    fig = px.scatter(df_file, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                st.plotly_chart(fig, use_container_width=True)

            # ==============================
            # 🔹 Gemini에게 질문하기
            # ==============================
            st.subheader("💬 Gemini에게 데이터 관련 질문하기")

            user_question = st.text_input("질문을 입력하세요 (예: '매출이 가장 높은 달은 언제야?')")
            if user_question:
                prompt_q = f"""
                아래 CSV 데이터 기반으로 '{user_question}'에 답변해줘. 
                데이터: \n\n{csv_text[:5000]}
                """
                response = model.generate_content(prompt_q)
                st.info(response.text)

        else:
            st.warning("CSV 파일을 먼저 업로드하세요.")
            
    elif menu == "🧭재고최적화":
        st.title("🧭 재고최적화")
        st.write("이곳에서 재고 최적화를 위한 운영 기준을 설정합니다.")

        #st.image("img/item_classification.png", caption="하위 폴더 이미지")
        # data_item_master = invopt.load_dataset_itemmaster()
        
        # st.write("#### Item Master")
        
        # df_item_master = pd.DataFrame(data_item_master)
        # st.dataframe(df_item_master.head()) 
        
        tab1, tab2, tab3, tab4 = st.tabs(
            ["PSI(운영데이터)", "Item Classification", "Inventory Simulation", "Inventory Trace"]
        )

        with tab1:  #PSI(Production-Sales-Inventory)
            df_psi = invopt.load_dataset_psi()
            sel_items = df_psi['Item Code'].unique()

            # sel_items = st.multiselect(
            #     "Choose Item Code",
            #     options=df_psi['Item Code'].unique(),
            #     default=df_psi['Item Code'].unique()
            # )
            if not sel_items.any():
                    st.error("Please select at least one Item Code.")
            else:
                st.dataframe(df_psi.head())
        
            col1, col2 = st.columns(2)
                
            with col1:
                st.write("#### Production")
                filtered_items_prod = invopt.plot_filtered_prod(df_psi, sel_items)
                st.pyplot(filtered_items_prod)

                st.write("#### Inventory")
                filtered_items_inv = invopt.plot_filtered_inv(df_psi, sel_items)
                st.pyplot(filtered_items_inv)
                
            with col2:
                st.write("#### Sales")
                filtered_items_sales = invopt.plot_filtered_sales(df_psi, sel_items)
                st.pyplot(filtered_items_sales)

                st.write("#### PSI Graph")
                filtered_items_psi = invopt.plot_filtered_psi(df_psi, sel_items)
                st.pyplot(filtered_items_psi)

        with tab2:  #Item Classification
            st.write("#### 판매 데이터 분석")
            df_sd02 = db_mysql.load_dataset_sales_weekly(biz_id)
            st.dataframe(df_sd02.head())

            # 4. 저장 버튼 로직
            if st.button("아이템 분류(Item Classification)"):                                        
                df_item_cluster = invopt.item_classifier(df_sd02)
                result = "아이템 분류 작업을 완료하였습니다."
                st.success(f"결과: {result}")

                st.dataframe(df_item_cluster.head())
                db_mysql.update_item_cluster(df_item_cluster)

            # data_items = invopt.load_dataset_item()
            # df_items = pd.DataFrame(data_items)

            # df_item = invopt.item_classifier(data_items)
                st.write("#### 아이템 분류 결과")

                item_classes = st.multiselect(
                    "Choose Class",
                    options=df_item_cluster['Cluster'].unique(),
                    default=df_item_cluster['Cluster'].unique()
                )
                if not item_classes:
                    st.error("Please select at least one Class.")
                else:
                    df_item_cluster_summary = db_mysql.load_dataset_mm17_stat(biz_id)
                    st.dataframe(df_item_cluster_summary.head())
            
                col1, col2 = st.columns(2)
                    
                with col1:
                    st.write("#### Item Classification: Item Number by Clusters")
                    filtered_items = invopt.plot_filtered_items_by_Count(df_item_cluster, item_classes)
                    st.pyplot(filtered_items)

                    st.write("#### Item Classification")
                    filtered_items = invopt.plot_filtered_items_by_Cluster(df_item_cluster, item_classes)
                    st.pyplot(filtered_items)

                    st.write("#### Item Classification")
                    filtered_items = invopt.plot_filtered_in_heatmap(df_item_cluster, item_classes)
                    st.pyplot(filtered_items)
                    
                with col2:
                    st.write("#### Item Classification")
                    filtered_items = invopt.plot_filtered_items_by_Cluster2(df_item_cluster, item_classes)
                    st.pyplot(filtered_items)

                    st.write("#### Item Classification")
                    filtered_items = invopt.plot_filtered_items_by_Cluster3(df_item_cluster, item_classes)
                    st.pyplot(filtered_items)

                    st.write("#### Item Classification")
                    filtered_items = invopt.plot_filtered_items_by_Cluster4(df_item_cluster, item_classes)
                    st.pyplot(filtered_items)

                # 3. 데이터 편집기 표시
                st.subheader("서비스 레벨 수정")
                if 'df_mm17' not in st.session_state:
                    st.session_state.df_mm17 = db_mysql.load_dataset_mm17_stat(31) #(st.session_state.db_conn)
                    st.session_state.df_mm17['Service Level'] = 80

                edited_df = st.data_editor(
                    st.session_state.df_mm17,
                    key="editor",
                    hide_index=True,
                    column_config={"trx_id": st.column_config.Column(disabled=True),
                                "biz_id": st.column_config.Column(disabled=True),
                                "Cluster Code": st.column_config.Column(disabled=True),
                                "Cluster Name": st.column_config.Column(disabled=True),
                                "Num. of Items": st.column_config.Column(disabled=True),
                                "Avg. Sales Qty": st.column_config.Column(disabled=True),
                                "attAvg. Sales Count": st.column_config.Column(disabled=True),
                                "Std. Deviation": st.column_config.Column(disabled=True),
                                "생성일자": st.column_config.Column(disabled=True),}
                ) # ID는 수정 불가
                
                # 4. 저장 버튼 로직
                if st.button("변경된 내용 저장 및 DB 반영"):                                        
                    editor_data = st.session_state["editor"]
                        
                    # 변경된 행(Rows) 정보 추출
                    # Streamlit은 changes 딕셔너리에 변경된 내용만 저장합니다.
                    changes = editor_data["edited_rows"]
                        
                    if changes:
                        st.subheader("변경된 내용 미리보기")
                            
                        # 원본 데이터프레임의 복사본 생성
                        original_df = st.session_state.df_mm17.copy()
                            
                        # 변경 사항을 원본 DataFrame에 적용
                        for index, updates in changes.items():
                            # index는 hide_index=True 때문에 0부터 시작하는 내부 인덱스입니다.
                            # 이 인덱스를 사용하여 원본 DataFrame의 행에 변경 사항 적용
                            for col, new_value in updates.items():
                                original_df.at[index, col] = new_value
                            
                        # 변경된 데이터만 추출 (여기서는 단순화하여 전체 업데이트로 처리)
                        # 실제 DB 업데이트 시에는 'ID'를 기준으로 변경된 행만 추출하여 효율적으로 처리해야 합니다.
                        
                        # 여기서는 변경된 전체 DataFrame을 업데이트 함수에 전달하는 예시로 대체합니다.
                        # 실제로는 변경된 행과 변경된 열 정보만을 담은 DataFrame을 만들어 전달하는 것이 효율적입니다.
                        
                        # 변경된 행만 포함하는 DataFrame 생성
                        updated_rows_data = []
                        for index, updates in changes.items():
                            row = original_df.loc[index].to_dict()
                            updated_rows_data.append(row)
                            
                        updates_df = pd.DataFrame(updated_rows_data)
                            
                        st.dataframe(updates_df)
                            
                        # 5. DB 업데이트 함수 호출
                        db_mysql.update_data_in_mm17(updates_df)
                            
                        # 6. 세션 상태의 원본 데이터를 업데이트된 내용으로 갱신
                        st.session_state.df_mm17 = original_df
                        #st.rerun() # 변경 사항 반영을 위해 앱 다시 실행
                            
                    else:
                        st.warning("변경된 내용이 없습니다.")

                    # st.divider()
                    # st.caption("현재 저장된 데이터:")
                    # st.dataframe(st.session_state.df_mm17, hide_index=True)
            
                # 5. ROP Setting
                # if st.button("최적 재고 산출"):                   
                #     db_mysql.prc_inv_optimize_in_mm16(31)
                #     result = "아이템 분류 작업을 완료하였습니다."
                #     st.success(f"결과: {result}")
                #     if 'df_mm16' not in st.session_state:
                #         st.session_state.df_mm16 = db_mysql.load_dataset_mm16_stat(31) #(st.session_state.db_conn)
                #         st.dataframe(st.session_state.df_mm16, hide_index=True)

        with tab3:        #Inventory Simulation
            st.write("#### Item Inventory Simulation")
            df_sim = invopt.item_simulation()
            
            sel_items = st.selectbox(
                "Choose Item",
                options=df_sim['Item Code'].unique(),
                index = 1
             )
            #st.dataframe(df_sim.head())

            col1, col2 = st.columns(2)

            with col1:
                filtered_items = invopt.plot_filtered_sim_by_item_daily(df_sim, sel_items)
                st.pyplot(filtered_items)

                filtered_items = invopt.plot_filtered_psi_actual(sel_items)
                st.pyplot(filtered_items)

            with col2:
                filtered_items = invopt.plot_filtered_sim_by_item(df_sim, sel_items)
                st.pyplot(filtered_items)

                filtered_items = invopt.plot_filtered_psi_simulation(sel_items)
                st.pyplot(filtered_items)
            
            if st.button('수요 데이터 이상치 탐지'):    
                df_sales = invopt.load_dataset_sales(sel_items)
                invopt.clear_db_results(31, sel_items)
                invopt.anormaly_ma(df_sales, True)

                result = "처리 완료!"
                st.success(f"결과: {result}")

                st.image(rf"img\moving_average_outliers.png", caption="Outlier")

        with tab4:            
            if st.button('Restart....'):
                st.write("#### Simulation re-starting...") 
        
    elif menu == "🤖 AI 챗봇":
        st.title("🤖 AI 챗봇")        
        llm = load_gemini()

        # -----------------------------
        # 3️⃣ Streamlit UI 구성
        # -----------------------------
        st.set_page_config(page_title="💬 Gemini ChatBot", page_icon="🤖", layout="wide")
        st.title("🤖 Gemini ChatBot")
        st.markdown("Google Gemini (langchain-google-genai 3.x) 기반 최신 Streamlit 챗봇")

        # 세션 상태 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # -----------------------------
        # 4️⃣ 사용자 입력 처리
        # -----------------------------
        user_input = st.chat_input("메시지를 입력하세요...")

        if user_input:
            # 사용자 메시지 추가
            st.session_state.messages.append(HumanMessage(content=user_input))

            # 최신 LangChain 구조에서는 invoke() 사용
            response = llm.invoke(st.session_state.messages)

            # 모델의 응답을 저장
            st.session_state.messages.append(AIMessage(content=response.content))

        # -----------------------------
        # 5️⃣ 채팅 UI 출력
        # -----------------------------
        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)
                
    elif menu == "문서파일 분석":
        # --------------------------
        # Streamlit UI 셋업
        # --------------------------
        st.set_page_config(page_title="Gemini PDF 요약 & Q&A", layout="wide")
        st.title("📘 Gemini 기반 PDF 요약 및 질의응답")
        llm = get_llm()
        embeddings = get_embeddings()

        # --- PDF 처리 함수 ---
        def extract_text_with_pypdf(file_obj):
            reader = PdfReader(file_obj)
            pages = []
            for p in reader.pages:
                txt = p.extract_text()
                if txt:
                    pages.append(txt)
            return "\n\n".join(pages)

        def load_documents(uploaded_file):
            try:
                loader = PyPDFLoader(uploaded_file)
                docs = loader.load()
            except Exception:
                text = extract_text_with_pypdf(uploaded_file)
                class SimpleDoc:
                    def __init__(self, page_content, metadata=None):
                        self.page_content = page_content
                        self.metadata = metadata or {}
                docs = [SimpleDoc(text, {"source": "uploaded"})]
            return docs

        def chunk_docs(docs, chunk_size=1000, chunk_overlap=200):
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            return splitter.split_documents(docs)

        @st.cache_resource
        def build_vectorstore(docs):
            return FAISS.from_documents(docs, embeddings)

        def ask_question(vectorstore, question, k=3):
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            relevant = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in relevant])
            prompt = f"""다음은 문서 내용입니다:\n\n{context}\n\n질문: {question}\n\n한국어로 간결하게 답변해 주세요."""
            result = llm.invoke([HumanMessage(content=prompt)])
            return result.content

        # --- Streamlit UI ---
        st.set_page_config(page_title="📘 PDF 요약 및 Q&A (Gemini + LangChain 최신 구조)", layout="wide")
        st.title("📘 PDF 요약 및 Q&A (LangChain 1.0 + Gemini)")

        uploaded = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

        if uploaded:
            docs = load_documents(uploaded)
            st.write(f"✅ 문서 로드 완료: {len(docs)}개 문서")

            chunk_size = st.slider("청크 크기", 500, 3000, 1000, 100)
            chunk_overlap = st.slider("청크 중첩 크기", 50, 800, 200, 50)
            docs_chunked = chunk_docs(docs, chunk_size, chunk_overlap)
            st.write(f"✅ {len(docs_chunked)}개의 청크로 분할됨")

            if st.button("벡터스토어 생성"):
                with st.spinner("벡터스토어 생성 중..."):
                    vectorstore = build_vectorstore(docs_chunked)
                st.success("✅ 벡터스토어 생성 완료")

                question = st.text_input("질문을 입력하세요:")
                if question:
                    with st.spinner("답변 생성 중..."):
                        answer = ask_question(vectorstore, question)
                    st.markdown("### 💬 답변")
                    st.write(answer)

# 앱 실행
if __name__ == "__main__":
    main()
    

#https://wikidocs.net/book/14285



