import requests
import math
import itertools
import streamlit as st
from streamlit_folium import st_folium
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import os, json
from dotenv import load_dotenv
import folium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import warnings
# pyproj CRS 경고 숨기기
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
load_dotenv()

VWORLD_KEY = os.getenv("VWORLD_KEY")
tiles = f"https://api.vworld.kr/req/wmts/1.0.0/{VWORLD_KEY}/Base/{{z}}/{{y}}/{{x}}.png" # Base, white, midnight, Hybrid
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 헬퍼 함수들
def Recomm_to_path(region_name, period):
    # print('Recomm')
    system_instructions = (
        "당신은 사용자가 특정 지역과 여행 기간을 입력하면, 그 기간 동안 추천할 여행 코스를 제공하는 도우미입니다.\n"
        f"여행 지역: {region_name}\n"
        f"여행 기간: {period}\n"
        "- 출력은 JSON object 형식이어야 하며, 키는 '1일차', '2일차', ..., 'n일차' 형태이고,"
        "각 키의 값은 다음 두 개의 key를 가진 object입니다:\n"
        "  '테마' (해당 날짜의 여행 테마를 담은 문자열),\n"
        "  '장소들' (해당 날짜에 방문할 3곳에서 7곳 사이의 장소 리스트)\n"
        "- 출력은 반드시 순수 JSON 문자열이어야 합니다. 다른 설명을 절대 포함하지 마!\n"
        "- 마크다운 코드 블록(```json)이나 기타 설명을 절대 포함하지 마세요!\n"
        "- 예:\n"
        "{'1일차': {'테마': '역사와 자연 탐방', '장소들': ['장소1', '장소2']},\n"
        "  '2일차': {'테마': '해변과 휴양', '장소들': ['장소3', '장소4']},\n"
        "  '3일차': {'테마': '문화와 쇼핑', '장소들': ['장소5', '장소6']}}"
    )
    gpt_response = llm.invoke(system_instructions) # 경로 추천 GPT 응답
    gpt_result = json.loads(gpt_response.content) # GPT 응답을 JSON으로 변환

    days = list(gpt_result.keys())

    params = {'query' : region_name} # 관광지역
    header = {'authorization': os.getenv('KAKAO_KEY')}

    Attr_dict = {'Attr_day':[],
                 'Attr_name':[],
                 'Attr_address':[],
                 'Attr_theme':[],
                 'Attr_URL':[]}
    Attr_dict2 = {'Attr_day':[],
                  'Attr_theme':[]}
    Attr_geometry = []
    Attr_geometry_line = []

    for day in days:
        temp_geometry = []
        today_theme = gpt_result[day]['테마']
        for dest in gpt_result[day]['장소들']:
            place_name, address_name, place_url, coord_x, coord_y = geocode_keyword(region_name, header, params, destination=dest)
            # print(place_name, address_name, place_url, coord_x, coord_y)
            
            if coord_x is not None and coord_y is not None:
                Attr_dict['Attr_day'].append(day)
                Attr_dict['Attr_name'].append(place_name)
                Attr_dict['Attr_address'].append(address_name)
                Attr_dict['Attr_URL'].append(place_url)
                Attr_dict['Attr_theme'].append(today_theme)
                Attr_geometry.append(Point(coord_x, coord_y))
                temp_geometry.append(Point(coord_x, coord_y))
        
        Attr_dict2['Attr_day'].append(day)
        Attr_dict2['Attr_theme'].append(today_theme)
        Attr_geometry_line.append(LineString(temp_geometry))

    gdf_Point = gpd.GeoDataFrame(pd.DataFrame(Attr_dict), geometry=Attr_geometry, crs=4326)
    gdf_Line = gpd.GeoDataFrame(pd.DataFrame(Attr_dict2), geometry=Attr_geometry_line, crs=4326)
    # gdf_Point['Attr_URL'] = gdf_Point['Attr_URL'].apply(make_clickable)
    gdf_Point['Attr_URL_html'] = gdf_Point['Attr_URL'].apply(make_clickable)

    return gdf_Point, gdf_Line


def Recomm_to_path_with_accom(region_name, accom, period):
    params = {'query' : region_name} # 관광지역
    header = {'authorization': os.getenv('KAKAO_KEY')}
    accom = geocode_keyword(region_name, header, params, destination=accom)[0]

    system_instructions = (
        "당신은 사용자가 특정 지역과 숙소 명칭, 여행 기간을 입력하면, 그 기간 동안 추천할 여행 코스를 제공하는 도우미입니다.\n"
        "각 날짜별 첫 장소와 마지막 장소는 사용자가 입력한 숙소이어야 합니다.\n"
        f"여행 지역: {region_name}\n"
        f"숙소: {accom}\n"
        f"여행 기간: {period}\n"
        "- 출력은 JSON object 형식이어야 하며, 키는 '1일차', '2일차', ..., 'n일차' 형태이고,"
        "각 키의 값은 다음 두 개의 key를 가진 object입니다:\n"
        "  '테마' (해당 날짜의 여행 테마를 담은 문자열),\n"
        "  '장소들' (숙소를 포함한 해당 날짜에 방문할 4곳에서 8곳 사이의 장소 리스트)\n"
        "- 출력은 반드시 순수 JSON 문자열이어야 합니다. 다른 설명을 절대 포함하지 마!\n"
        "- 마크다운 코드 블록(```json)이나 기타 설명을 절대 포함하지 마세요!\n"
        "- 예:\n"
        "{'1일차': { '테마': '역사와 자연 탐방', '장소들': ['숙소', '장소1', '장소2', '숙소'] },\n"
        "  '2일차': { '테마': '해변과 휴양', '장소들': ['숙소', '장소3', '장소4', '숙소'] },\n"
        "  '3일차': { '테마': '문화와 쇼핑', '장소들': ['숙소', '장소5', '장소6', '숙소'] } }"
    )
    gpt_response = llm.invoke(system_instructions) # 경로 추천 GPT 응답
    gpt_result = json.loads(gpt_response.content) # GPT 응답을 JSON으로 변환

    days = list(gpt_result.keys())

    Attr_dict = {'Attr_day':[],
                 'Attr_name':[],
                 'Attr_address':[],
                 'Attr_theme':[],
                 'Attr_URL':[]}
    Attr_dict2 = {'Attr_day':[],
                  'Attr_theme':[]}
    Attr_geometry = []
    Attr_geometry_line = []

    for day in days:
        temp_geometry = []
        today_theme = gpt_result[day]['테마']
        for dest in gpt_result[day]['장소들']:
            place_name, address_name, place_url, coord_x, coord_y = geocode_keyword(region_name, header, params, destination=dest)
            # print(place_name, address_name, place_url, coord_x, coord_y)
            
            if coord_x is not None and coord_y is not None:
                Attr_dict['Attr_day'].append(day)
                Attr_dict['Attr_name'].append(place_name)
                Attr_dict['Attr_address'].append(address_name)
                Attr_dict['Attr_URL'].append(place_url)
                Attr_dict['Attr_theme'].append(today_theme)
                Attr_geometry.append(Point(coord_x, coord_y))
                temp_geometry.append(Point(coord_x, coord_y))
        
        Attr_dict2['Attr_day'].append(day)
        Attr_dict2['Attr_theme'].append(today_theme)
        Attr_geometry_line.append(LineString(temp_geometry))

    gdf_Point = gpd.GeoDataFrame(pd.DataFrame(Attr_dict), geometry=Attr_geometry, crs=4326)
    gdf_Line = gpd.GeoDataFrame(pd.DataFrame(Attr_dict2), geometry=Attr_geometry_line, crs=4326)
    gdf_Point['Attr_URL_html'] = gdf_Point['Attr_URL'].apply(make_clickable)

    return gdf_Point, gdf_Line


def geocode_keyword(region_name, header, params, destination):
    # print('geocoding')
    try:
        loc_info = requests.get('https://dapi.kakao.com/v2/local/search/address.json?&query=' + region_name,  # 관광지역 검색
                                headers=header, params=params).json()
        ref_destn = [loc_info['documents'][0]['address']['region_1depth_name'], loc_info['documents'][0]['address']['region_2depth_name']] # 관광지역 시도, 시군구 단위
    except (IndexError, KeyError, TypeError):
        ref_destn = ["", ""]
    loc_info = requests.get('https://dapi.kakao.com/v2/local/search/keyword.json?&query=' + destination, # 관광지 검색
                                headers=header, params=params).json()

    place_name = None
    address_name = None
    place_url = None
    coord_x = None
    coord_y = None

    for loc in loc_info['documents']: # 카카오 결과 목록에서
        if ref_destn[0] == "" and ref_destn[1] == "": # 지역 필터가 아예 없을 때 (복합 지역인 경우) -> 가장 정확도 높은 첫 번째 결과 선택
            place_name = loc['place_name']
            address_name = loc['address_name']
            place_url = loc['place_url']
            coord_x = loc['x']
            coord_y = loc['y']
            break

        if ref_destn[1] == '': # 시군구 단위 없을 때
            if ref_destn[0] in loc['address_name']: # 시도 단위만 맞으면
                place_name = loc['place_name']
                address_name = loc['address_name']
                place_url = loc['place_url']
                coord_x = loc['x']
                coord_y = loc['y']
                break
        else: # 시군구 단위도 있을 때
            if (ref_destn[0] in loc['address_name']) and (ref_destn[1] in loc['address_name']): # 시도, 시군구 단위 모두 맞을 때
                place_name = loc['place_name']
                address_name = loc['address_name']
                place_url = loc['place_url']
                coord_x = loc['x']
                coord_y = loc['y']
                break
            elif (ref_destn[0] in loc['address_name']): # 시도 단위라도 맞을 때
                place_name = loc['place_name']
                address_name = loc['address_name']
                place_url = loc['place_url']
                coord_x = loc['x']
                coord_y = loc['y']
            else: # 그 외 경우
                place_name = None
                address_name = None
                place_url = None
                coord_x = None
                coord_y = None
    
    if (coord_x == None) & (len(destination.split()) > 1):
        try:
            loc_info = requests.get('https://dapi.kakao.com/v2/local/search/keyword.json?&query=' + destination.split()[0], # 관광지 검색
                                headers=header, params=params).json()
                                
            for loc in loc_info['documents']: # 카카오 결과 목록에서
                if ref_destn[0] == "" and ref_destn[1] == "": # 여기도 동일하게 필터 없으면 바로 통과
                    place_name = loc['place_name']
                    address_name = loc['address_name']
                    place_url = loc['place_url']
                    coord_x = loc['x']
                    coord_y = loc['y']
                    break

                if ref_destn[1] == '': # 시군구 단위 없을 때
                    if ref_destn[0] in loc['address_name']: # 시도 단위만 맞으면
                        place_name = loc['place_name']
                        address_name = loc['address_name']
                        place_url = loc['place_url']
                        coord_x = loc['x']
                        coord_y = loc['y']
                        break
                else: # 시군구 단위도 있을 때
                    if (ref_destn[0] in loc['address_name']) and (ref_destn[1] in loc['address_name']): # 시도, 시군구 단위 모두 맞을 때
                        place_name = loc['place_name']
                        address_name = loc['address_name']
                        place_url = loc['place_url']
                        coord_x = loc['x']
                        coord_y = loc['y']
                        break
                    elif (ref_destn[0] in loc['address_name']): # 시도 단위라도 맞을 때
                        place_name = loc['place_name']
                        address_name = loc['address_name']
                        place_url = loc['place_url']
                        coord_x = loc['x']
                        coord_y = loc['y']

        except: # 그 외 경우
            pass

    return place_name, address_name, place_url, coord_x, coord_y

def make_clickable(url):
    if url:
        # target="_blank"는 링크를 새 탭에서 열게 합니다.
        return f'<a href="{url}" target="_blank">상세보기</a>'
    return "링크 없음"

def Reorder_path(gdf_Point):
    days = gdf_Point['Attr_day'].unique().tolist()
    days.sort()

    # 추천경로
    final_day_list = []

    for day in days:
        tmp_gpt_Pts = gdf_Point[gdf_Point['Attr_day']==day]

        waypoint_candidate = []
        for idx, row in tmp_gpt_Pts.iterrows():
            coord_x = row['geometry'].x
            coord_y = row['geometry'].y
            # waypoint_candidate.append(f"{coord_x},{coord_y},name={row['Attr_name']}")
            waypoint_candidate.append(((coord_x,coord_y),row['Attr_name']))

        nPr = list(itertools.permutations(waypoint_candidate, len(waypoint_candidate)))
        nPr = [list(n) for n in nPr]

        min_dist = 1e9 # 가장 이동시간이 짧은 경우 탐색
        for test_waypoints in nPr:
            tmp_dist = 0
            for i in range(1, len(test_waypoints)):
                tmp_dist += math.dist(test_waypoints[i][0], test_waypoints[i-1][0])
            if tmp_dist < min_dist:
                min_dist = tmp_dist
                final_route = test_waypoints
        final_day_list.append(final_route)


    # final_day_list의 각 요소들 별로 정렬하고 하나의 gdf로 병합
    final_gdf_list = []
    for day_idx, day in enumerate(days):
        temp_gdf = gdf_Point.set_index('Attr_name').loc[[place[1] for place in final_day_list[day_idx]]].reset_index()
        final_gdf_list.append(temp_gdf)
    gdf_Point_re = pd.concat(final_gdf_list).reset_index(drop=True)

    gdf_Line_re = gdf_Point_re.groupby(['Attr_day', 'Attr_theme'])['geometry'].apply(
        lambda x: LineString(x.tolist())
    ).reset_index()
    
    return gdf_Point_re, gdf_Line_re

def Reorder_path_with_accom(gdf_Point):
    days = gdf_Point['Attr_day'].unique().tolist()
    days.sort()

    # 추천경로
    final_day_list = []

    for day in days:
        tmp_gpt_Pts = gdf_Point[gdf_Point['Attr_day']==day]

        waypoint_candidate = []
        for idx, row in tmp_gpt_Pts.iterrows():
            waypoint_candidate.append({
                'idx': idx,              # 고유 인덱스 (나중에 순서 복원용)
                'geom': row['geometry']  # 거리 계산용 좌표
            })

        start_node = waypoint_candidate[0]       # 첫 번째 장소 (고정)
        end_node = waypoint_candidate[-1]        # 마지막 장소 (고정)
        middle_nodes = waypoint_candidate[1:-1]  # 중간 장소들 (재배열 대상)

        nPr = list(itertools.permutations(middle_nodes, len(middle_nodes)))
        best_middle_order = middle_nodes

        min_dist = float('inf')# 무한대로 초기화
        for mid_route in nPr:
            test_waypoints = [start_node] + list(mid_route) + [end_node]
            tmp_dist = 0
            for i in range(1, len(test_waypoints)):
                p1 = test_waypoints[i-1]['geom']
                p2 = test_waypoints[i]['geom']
                tmp_dist += math.dist((p1.x, p1.y), (p2.x, p2.y))
            if tmp_dist < min_dist:
                min_dist = tmp_dist
                best_middle_order = mid_route
        # 최적 경로 완성: [시작] + [최적 중간] + [끝]
        best_route = [start_node] + list(best_middle_order) + [end_node]

        # 결정된 순서대로 인덱스만 추출하여 저장
        final_day_list.extend([item['idx'] for item in best_route])
        
    
    gdf_Point_re = gdf_Point.loc[final_day_list].reset_index(drop=True)

    gdf_Line_re = gdf_Point_re.groupby(['Attr_day', 'Attr_theme'])['geometry'].apply(
            lambda x: LineString(x.tolist()) if len(x) > 1 else None
        ).reset_index()
    gdf_Line_re = gdf_Line_re.dropna(subset=['geometry'])
    
    return gdf_Point_re, gdf_Line_re

#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#

@tool
def recommend_travel_course(region_name: str, period: str) -> str:
    """
    (중요!) 사용자가 숙소를 구체적으로 언급하지 않았을 때 사용하세요.
    특정 지역과 여행 기간을 받아 추천할 여행 코스를 지도에 표시합니다.
    Args:
        region_name (str): '서울특별시', '성북구', '인천광역시 동구', '제주도' 등의 지역명
        period (str): 3일, 이틀, 사흘 등의 기간
    """

    try:
        current_request_key = f"{region_name}_{period}"
        if st.session_state.get("last_request_key") != current_request_key:
            with st.spinner("여행 경로를 생성하고 지도를 그리는 중입니다..."):
                gdf_Point, gdf_Line = Recomm_to_path(region_name, period)
                
                # 계산된 데이터를 세션에 저장 (나중에 지도 그릴 때 씀)
                st.session_state["cached_gdf_point"] = gdf_Point
                st.session_state["cached_gdf_line"] = gdf_Line
                st.session_state["last_request_key"] = current_request_key

            # 지도 표시 플래그 켜기
            st.session_state["show_tour_map"] = True
            st.session_state["region_name"] = region_name
            st.session_state["period"] = period
            # 다른 지도 닫기
            st.session_state["show_tour_map_accom"] = False
            st.session_state["show_reorder_tour_map"] = False
            st.session_state["show_reorder_tour_map_accom"] = False
            return "지도를 출력합니다."

    except Exception as e:
        return f"지도 생성 중 오류 발생: {str(e)}"
    

@tool
def recommend_travel_course_with_accom(region_name: str, accom: str, period: str) -> str:
    """
    (중요!) 사용자가 '숙소' 또는 '호텔'의 이름을 명확히 알려주었을 때만 사용하세요.
    특정 지역과 숙소 명칭, 여행 기간을 받아 추천할 여행 코스를 지도에 표시합니다.
    Args:
        region_name (str): '서울특별시', '성북구', '인천광역시 동구', '제주도' 등의 지역명
        accom (str): '웨스틴조선 서울', '오션스위츠 제주', '라마다 앙코르 부산역' 등 숙소명
        period (str): 3일, 이틀, 사흘 등의 기간
    """

    try:
        current_request_key = f"{region_name}_{accom}_{period}"
        if st.session_state.get("last_request_key") != current_request_key:
            with st.spinner("여행 경로를 생성하고 지도를 그리는 중입니다..."):
                gdf_Point, gdf_Line = Recomm_to_path_with_accom(region_name, accom, period)
                
                # 계산된 데이터를 세션에 저장 (나중에 지도 그릴 때 씀)
                st.session_state["cached_gdf_point_accom"] = gdf_Point
                st.session_state["cached_gdf_line_accom"] = gdf_Line
                st.session_state["last_request_key"] = current_request_key

            # 지도 표시 플래그 켜기
            st.session_state["show_tour_map_accom"] = True
            st.session_state["region_name"] = region_name
            st.session_state["period"] = period
            # 다른 지도 닫기
            st.session_state["show_tour_map"] = False
            st.session_state["show_reorder_tour_map"] = False
            st.session_state["show_reorder_tour_map_accom"] = False
            return "지도를 출력합니다."

    except Exception as e:
        return f"지도 생성 중 오류 발생: {str(e)}"


# @tool
# def reorder_travel_course() -> str:
#     """
#     추천한 여행 경로를 재정렬 및 최적화하여 지도에 표시합니다.
#     """
#     try:
#         if st.session_state.get("cached_gdf_point") is None:
#             return "먼저 여행 코스를 추천받은 후에 경로 최적화를 요청해주세요."
        
#         region_name = st.session_state["region_name"]
#         period = st.session_state["period"]
#         current_request_key = st.session_state["last_request_key"]

#         gdf_Point = st.session_state["cached_gdf_point"]
        
#         with st.spinner("여행 경로를 최적화하는 중입니다..."):
#             gdf_Point_re, gdf_Line_re = Reorder_path(gdf_Point)

#             st.session_state["cached_gdf_point"] = gdf_Point_re
#             st.session_state["cached_gdf_line"] = gdf_Line_re

#         # 지도 표시 플래그 켜기
#         st.session_state["show_reorder_tour_map"] = True
#         # 다른 지도 닫기
#         st.session_state["show_tour_map"] = False
#         st.session_state["show_tour_map_accom"] = False
#         st.session_state["show_reorder_tour_map_accom"] = False
#         return "지도를 출력합니다."

#     except Exception as e:
#         return f"지도 생성 중 오류 발생: {str(e)}"

# 도구 바인딩
tools = [
    recommend_travel_course,
    recommend_travel_course_with_accom,
    # reorder_travel_course,
         ]
tool_dict = {
    "recommend_travel_course": recommend_travel_course,
    "recommend_travel_course_with_accom": recommend_travel_course_with_accom,
    # "reorder_travel_course":reorder_travel_course,
}

# Streamlit 앱
st.set_page_config(page_title="Tourist Recommender",
                   layout="wide",
                   initial_sidebar_state="collapsed"
                   )

# iframe 여백 제거 CSS
st.markdown("""
<style>
    /* 1. iframe(지도)을 감싸는 컨테이너의 하단 여백 제거 */
    .element-container:has(iframe) {
        margin-bottom: 0rem !important;
        padding-bottom: 0rem !important;
    }
    
    /* 2. iframe 자체를 블록 요소로 만들어 하단 미세 공백 제거 */
    iframe {
        display: block;
    }
    
    /* 3. 지도 바로 다음에 오는 표(Dataframe)와의 간격 조정 */
    /* 필요에 따라 -1rem 값을 조절해서 간격을 더 좁히거나 넓힐 수 있습니다 */
    .stDataFrame {
        margin-top: -0.5rem !important; 
    }
    
    /* (옵션) 수직 블록 간의 기본 간격 줄이기 */
    .stVerticalBlock > div {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("관광추천 챗봇")

st.sidebar.title("모델 설정")
model_option = st.sidebar.radio(
    "사용할 모델을 선택하세요:",
    ("Gemini 2.5 Flash (Google)", "GPT-5 Nano (OpenAI)", "Hugging Face (GPT-4)", "Hugging Face (MiniMax)"),
    index=2  # 기본값: 0은 첫 번째(Gemini), 1은 두 번째(GPT)
)
# 선택된 옵션에 따라 모델 초기화
if "Gemini" in model_option:
    # Google Gemini 설정
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite"
    )
    st.sidebar.success("Gemini 모델이 선택되었습니다.")
elif "GPT-5" in model_option:
    # OpenAI GPT 설정
    llm = ChatOpenAI(
        model="gpt-5-nano"
    )
    st.sidebar.info("GPT 모델이 선택되었습니다.")
elif "Hugging" in model_option:
    # Hugging Face 설정
    if "GPT-4" in model_option:
        repo_id = "openai/gpt-oss-120b"
    elif 'Mini' in model_option:
        repo_id = "MiniMaxAI/MiniMax-M2.1"
        
    llm = HuggingFaceEndpoint(
        repo_id = repo_id,  # 모델 저장소 ID를 지정
        max_new_tokens=2048,
        temperature=0.01,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )
    # 2. 채팅 모델 래퍼 씌우기 (ChatHuggingFace)
    llm = ChatHuggingFace(llm=llm)

    st.sidebar.info("Hugging Face 모델이 선택되었습니다.")

llm_with_tools = llm.bind_tools(tools)

# 사용자의 메시지 처리하기 위한 함수
def get_ai_response(messages):
    response = llm_with_tools.stream(messages) # ① llm.stream()을 llm_with_tools.stream()로 변경

    gathered = None # ②
    for chunk in response:
        yield chunk

        if gathered is None: #  ③
            gathered = chunk
        else:
            gathered += chunk
 
    if gathered and gathered.tool_calls:
        st.session_state.messages.append(gathered)
        
        for tool_call in gathered.tool_calls:
            tool_name = tool_call['name']
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call) 
            # print(tool_msg, type(tool_msg))

            st.session_state.messages.append(tool_msg)

            # # (재귀 호출 안 함)
            # if tool_name == "recommend_travel_course":
            #     # 빈 문자열을 yield하여 스트림을 정상 종료 처리 (선택 사항)
            #     yield ""
            #     return  # <--- 여기서 함수 종료! (재귀 호출 안 함)
            # elif tool_name == "reorder_travel_course":
            #     yield ""
            #     return

            # if tool_name in ["recommend_travel_course", "recommend_travel_course_with_accom", "reorder_travel_course"]:
            if tool_name in ["recommend_travel_course", "recommend_travel_course_with_accom"]:
                yield "📍 지도를 준비 중입니다. 아래에 표시됩니다."
                return  # 함수 완전 종료
           
        for chunk in get_ai_response(st.session_state.messages):
            yield chunk
    else:
        # tool 호출 없는 일반 응답
        # yield gathered
        pass



# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다."),  
        AIMessage("How can I help you?")
    ]

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, ToolMessage):
            if "지도를 출력합니다." in msg.content:
                with st.chat_message("tool"):
                    st.write("📍 아래에 지도를 출력했습니다.")
            else:
                st.chat_message("tool").write(msg.content)

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    st.session_state.messages.append(HumanMessage(prompt)) # 사용자 메시지 저장
    response = get_ai_response(st.session_state["messages"])
    result = st.chat_message("assistant").write_stream(response) # AI 메시지 출력
    st.session_state["messages"].append(AIMessage(result)) # AI 메시지 저장


# 추천 여행 경로 지도 표시
if st.session_state.get("show_tour_map"):
    try:
        region_name = st.session_state["region_name"]
        period = st.session_state["period"]
        # current_request_key = f"{region_name}_{period}"

        # # 데이터가 준비되었는지 확인하는 플래그
        # is_data_ready = False

        # # 'cached_gdf_point'가 없거나, 이전 요청과 다르면 새로 계산
        # if ("cached_gdf_point" not in st.session_state) or (st.session_state.get("last_request_key") != current_request_key):
        #     with st.spinner("여행 경로를 생성하고 지도를 그리는 중입니다..."):
        #         gdf_Point, gdf_Line = Recomm_to_path(region_name, period)
        #     st.write("🗺️ 장소 위치를 지도에 변환 중입니다...")

        #     # 결과를 세션 상태에 저장 (캐싱)
        #     st.session_state["cached_gdf_point"] = gdf_Point
        #     st.session_state["cached_gdf_line"] = gdf_Line
        #     st.session_state["last_request_key"] = current_request_key
        #     is_data_ready = True

        # else:
            # 이미 계산된 값이 있으면 그대로 가져옴 (API 호출 안 함)
        gdf_Point = st.session_state["cached_gdf_point"]
        gdf_Line = st.session_state["cached_gdf_line"]
        # is_data_ready = True

        # gdf_Point, gdf_Line = Recomm_to_path(region_name, period)
        # if is_data_ready:
        if gdf_Point is not None:
            m = folium.Map(control_scale=True, tiles=None)
            folium.TileLayer(tiles=tiles, attr="VWorld").add_to(m)

            # GeoJson으로 gdf 추가
            layer = folium.GeoJson(
                gdf_Point, name="추천 장소",
                popup=folium.features.GeoJsonPopup(
                                            fields=['Attr_name', 'Attr_address', 'Attr_theme', 'Attr_URL_html'],
                                            aliases=['장소명', '주소', '여행테마', 'URL']
                )
            ).add_to(m)

            gdf_Line.rename(columns={'Attr_day': '일정'}).explore(
                m=m, column='일정', cmap='tab10', legend=True, style_kwds={"weight":5})

            # 지도가 보여질 범위를 설정
            bounds = layer.get_bounds()
            m.fit_bounds(bounds, padding=[50, 50])

            with st.container():
                st.markdown(f"{period} 간의 {region_name} 여행 지도")
                st_folium(m, use_container_width=True, height=600, returned_objects=[])

            
            st.divider() # 구분선
            st.markdown("### 📋 여행지 상세 목록")

            # 1. 보기 좋게 만들기 위해 'geometry' 컬럼 제거 (좌표값 숨김)
            df_display = gdf_Point.drop(columns=['geometry', 'Attr_URL_html']).copy()

            # 2. 컬럼 이름 한글로 변경
            st.dataframe(
                df_display,
                use_container_width=True, # 가로폭 꽉 채우기
                hide_index=True,          # 인덱스(0,1,2..) 숨기기
                column_config={
                    "Attr_day": st.column_config.TextColumn("일차", width="small"),
                    "Attr_name": st.column_config.TextColumn("장소명", width="medium"),
                    "Attr_address": st.column_config.TextColumn("주소", width="large"),
                    "Attr_theme": st.column_config.TextColumn("테마", width="medium"),
                    "Attr_URL": st.column_config.LinkColumn(
                        "상세보기",             # 컬럼 헤더 이름
                        help="클릭하면 카카오맵으로 이동합니다.", 
                        display_text="바로가기", # URL 대신 보여줄 텍스트 (예: https://... -> 바로가기)
                        width="small"
                    ),
                }
            )
            # st.rerun()

            st.divider() # 구분선
            st.markdown("#### 💡 여행 경로 최적화 제안")
            col_msg, col_btn = st.columns([0.7, 0.3])
            with col_msg:
                st.info("이동 거리가 짧아지도록 여행 코스를 최적화해 드릴까요?")
            with col_btn:
                # 버튼을 클릭하면
                if st.button("경로 최적화 실행", help="이동 거리를 기준으로 경로를 재정렬합니다.", use_container_width=True):
                    try:
                        with st.spinner("최적의 경로를 계산 중입니다..."):
                            # 1. 현재 세션에 저장된 포인트 데이터 가져오기
                            current_gdf = st.session_state["cached_gdf_point"]
                            
                            # 2. 최적화 로직 실행 (Reorder_path 함수 재사용)
                            gdf_Point_re, gdf_Line_re = Reorder_path(current_gdf)

                            # 3. 최적화된 데이터를 세션에 덮어쓰기
                            st.session_state["cached_gdf_point"] = gdf_Point_re
                            st.session_state["cached_gdf_line"] = gdf_Line_re
                        
                        # 4. 화면 전환 플래그 설정
                        st.session_state["show_tour_map"] = False        # 현재 지도 끄기
                        st.session_state["show_reorder_tour_map"] = True # 최적화 지도 켜기
                        
                        # 5. 마지막 요청 키 유지 (데이터가 날아가지 않도록)
                        # (필요하다면 로깅을 위해 메시지 추가 가능)
                        # st.session_state.messages.append(AIMessage("경로를 최적화하여 지도를 다시 그렸습니다."))

                        # 6. 화면 새로고침
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"최적화 중 오류 발생: {e}")

    
    except Exception as e:
            st.error(f"지도를 작성하는 중 오류가 발생했습니다: {e}")

elif st.session_state.get("show_tour_map_accom"):
    try:
        region_name = st.session_state["region_name"]
        period = st.session_state["period"]
        gdf_Point = st.session_state["cached_gdf_point_accom"]
        gdf_Line = st.session_state["cached_gdf_line_accom"]

        if gdf_Point is not None:
            m = folium.Map(control_scale=True, tiles=None)
            folium.TileLayer(tiles=tiles, attr="VWorld").add_to(m)

            # GeoJson으로 gdf 추가
            layer = folium.GeoJson(
                gdf_Point, name="추천 장소",
                popup=folium.features.GeoJsonPopup(
                                            fields=['Attr_name', 'Attr_address', 'Attr_theme', 'Attr_URL_html'],
                                            aliases=['장소명', '주소', '여행테마', 'URL']
                )
            ).add_to(m)

            gdf_Line.rename(columns={'Attr_day': '일정'}).explore(
                m=m, column='일정', cmap='tab10', legend=True, style_kwds={"weight":5})

            # 지도가 보여질 범위를 설정
            bounds = layer.get_bounds()
            m.fit_bounds(bounds, padding=[50, 50])

            with st.container():
                st.markdown(f"{period} 간의 {region_name} 여행 지도")
                st_folium(m, use_container_width=True, height=600, returned_objects=[])

            
            st.divider() # 구분선
            st.markdown("### 📋 여행지 상세 목록")

            # 1. 보기 좋게 만들기 위해 'geometry' 컬럼 제거 (좌표값 숨김)
            df_display = gdf_Point.drop(columns=['geometry', 'Attr_URL_html']).copy()

            # 2. 컬럼 이름 한글로 변경
            st.dataframe(
                df_display,
                use_container_width=True, # 가로폭 꽉 채우기
                hide_index=True,          # 인덱스(0,1,2..) 숨기기
                column_config={
                    "Attr_day": st.column_config.TextColumn("일차", width="small"),
                    "Attr_name": st.column_config.TextColumn("장소명", width="medium"),
                    "Attr_address": st.column_config.TextColumn("주소", width="large"),
                    "Attr_theme": st.column_config.TextColumn("테마", width="medium"),
                    "Attr_URL": st.column_config.LinkColumn(
                        "상세보기",             # 컬럼 헤더 이름
                        help="클릭하면 카카오맵으로 이동합니다.", 
                        display_text="바로가기", # URL 대신 보여줄 텍스트 (예: https://... -> 바로가기)
                        width="small"
                    ),
                }
            )

            st.divider() # 구분선
            st.markdown("#### 💡 여행 경로 최적화 제안")
            col_msg, col_btn = st.columns([0.7, 0.3])
            with col_msg:
                st.info("이동 거리가 짧아지도록 여행 코스를 최적화해 드릴까요?")
            with col_btn:
                # 버튼을 클릭하면
                if st.button("경로 최적화 실행", help="이동 거리를 기준으로 경로를 재정렬합니다.", use_container_width=True):
                    try:
                        with st.spinner("최적의 경로를 계산 중입니다..."):
                            # 1. 현재 세션에 저장된 포인트 데이터 가져오기
                            current_gdf = st.session_state["cached_gdf_point_accom"]
                            
                            # 2. 최적화 로직 실행 (Reorder_path 함수 재사용)
                            gdf_Point_re, gdf_Line_re = Reorder_path_with_accom(current_gdf)

                            # 3. 최적화된 데이터를 세션에 덮어쓰기
                            st.session_state["cached_gdf_point_accom"] = gdf_Point_re
                            st.session_state["cached_gdf_line_accom"] = gdf_Line_re
                        
                        # 4. 화면 전환 플래그 설정
                        st.session_state["show_tour_map_accom"] = False        # 현재 지도 끄기
                        st.session_state["show_reorder_tour_map_accom"] = True # 최적화 지도 켜기

                        # 6. 화면 새로고침
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"최적화 중 오류 발생: {e}")

    
    except Exception as e:
            st.error(f"지도를 작성하는 중 오류가 발생했습니다: {e}")

elif st.session_state.get("show_reorder_tour_map"):
    try:
        region_name = st.session_state["region_name"]
        period = st.session_state["period"]

        gdf_Point = st.session_state["cached_gdf_point"]
        gdf_Line = st.session_state["cached_gdf_line"]

        if gdf_Point is not None:
            m = folium.Map(control_scale=True, tiles=None)
            folium.TileLayer(tiles=tiles, attr="VWorld").add_to(m)

            # GeoJson으로 gdf 추가
            layer = folium.GeoJson(
                gdf_Point, name="추천 장소",
                popup=folium.features.GeoJsonPopup(
                                            fields=['Attr_name', 'Attr_address', 'Attr_theme', 'Attr_URL_html'],
                                            aliases=['장소명', '주소', '여행테마', 'URL']
                )
            ).add_to(m)

            gdf_Line.rename(columns={'Attr_day': '일정'}).explore(
                m=m, column='일정', cmap='tab10', legend=True, style_kwds={"weight":5})

            # 지도가 보여질 범위를 설정
            bounds = layer.get_bounds()
            m.fit_bounds(bounds, padding=[50, 50])

            st.markdown(f"{period} 간의 {region_name} 여행 지도")
            st_folium(m, use_container_width=True, height=600)

            st.divider() # 구분선
            st.markdown("### 📋 여행지 상세 목록")

            # 1. 보기 좋게 만들기 위해 'geometry' 컬럼 제거 (좌표값 숨김)
            df_display = gdf_Point.drop(columns=['geometry', 'Attr_URL_html']).copy()

            # 2. 컬럼 이름 한글로 변경
            st.dataframe(
                df_display,
                use_container_width=True, # 가로폭 꽉 채우기
                hide_index=True,          # 인덱스(0,1,2..) 숨기기
                column_config={
                    "Attr_day": st.column_config.TextColumn("일차", width="small"),
                    "Attr_name": st.column_config.TextColumn("장소명", width="medium"),
                    "Attr_address": st.column_config.TextColumn("주소", width="large"),
                    "Attr_theme": st.column_config.TextColumn("테마", width="medium"),
                    "Attr_URL": st.column_config.LinkColumn(
                        "상세보기",             # 컬럼 헤더 이름
                        help="클릭하면 카카오맵으로 이동합니다.", 
                        display_text="바로가기", # URL 대신 보여줄 텍스트 (예: https://... -> 바로가기)
                        width="small"
                    ),
                }
            )
    
    except Exception as e:
            st.error(f"지도를 작성하는 중 오류가 발생했습니다: {e}")


elif st.session_state.get("show_reorder_tour_map_accom"):
    try:
        region_name = st.session_state["region_name"]
        period = st.session_state["period"]

        gdf_Point = st.session_state["cached_gdf_point_accom"]
        gdf_Line = st.session_state["cached_gdf_line_accom"]

        if gdf_Point is not None:
            m = folium.Map(control_scale=True, tiles=None)
            folium.TileLayer(tiles=tiles, attr="VWorld").add_to(m)

            # GeoJson으로 gdf 추가
            layer = folium.GeoJson(
                gdf_Point, name="추천 장소",
                popup=folium.features.GeoJsonPopup(
                                            fields=['Attr_name', 'Attr_address', 'Attr_theme', 'Attr_URL_html'],
                                            aliases=['장소명', '주소', '여행테마', 'URL']
                )
            ).add_to(m)

            gdf_Line.rename(columns={'Attr_day': '일정'}).explore(
                m=m, column='일정', cmap='tab10', legend=True, style_kwds={"weight":5})

            # 지도가 보여질 범위를 설정
            bounds = layer.get_bounds()
            m.fit_bounds(bounds, padding=[50, 50])

            st.markdown(f"{period} 간의 {region_name} 여행 지도")
            st_folium(m, use_container_width=True, height=600)

            st.divider() # 구분선
            st.markdown("### 📋 여행지 상세 목록")

            # 1. 보기 좋게 만들기 위해 'geometry' 컬럼 제거 (좌표값 숨김)
            df_display = gdf_Point.drop(columns=['geometry', 'Attr_URL_html']).copy()

            # 2. 컬럼 이름 한글로 변경
            st.dataframe(
                df_display,
                use_container_width=True, # 가로폭 꽉 채우기
                hide_index=True,          # 인덱스(0,1,2..) 숨기기
                column_config={
                    "Attr_day": st.column_config.TextColumn("일차", width="small"),
                    "Attr_name": st.column_config.TextColumn("장소명", width="medium"),
                    "Attr_address": st.column_config.TextColumn("주소", width="large"),
                    "Attr_theme": st.column_config.TextColumn("테마", width="medium"),
                    "Attr_URL": st.column_config.LinkColumn(
                        "상세보기",             # 컬럼 헤더 이름
                        help="클릭하면 카카오맵으로 이동합니다.", 
                        display_text="바로가기", # URL 대신 보여줄 텍스트 (예: https://... -> 바로가기)
                        width="small"
                    ),
                }
            )
    
    except Exception as e:
            st.error(f"지도를 작성하는 중 오류가 발생했습니다: {e}")



# ===== 지도 닫기 버튼 =====
if any([st.session_state.get("show_tour_map"),
        st.session_state.get("show_tour_map_accom"),
        st.session_state.get("show_reorder_tour_map"),
        st.session_state.get("show_reorder_tour_map_accom"),
        ]):
    st.divider() # 구분선
    st.info("🗺️ 현재 지도 출력 중")
            
    if st.button("지도 닫기"):
        # 모두 끄고 마지막 지도 기록
        if st.session_state["show_tour_map"]:
            st.session_state["last_shown_map"] = "show_tour_map"
        elif st.session_state["show_tour_map_accom"]:
            st.session_state["last_shown_map"] = "show_tour_map_accom"
        elif st.session_state["show_reorder_tour_map"]:
            st.session_state["last_shown_map"] = "show_reorder_tour_map"
        elif st.session_state["show_reorder_tour_map_accom"]:
            st.session_state["last_shown_map"] = "show_reorder_tour_map_accom"

        st.session_state["show_tour_map"] = False
        st.session_state["show_tour_map_accom"] = False
        st.session_state["show_reorder_tour_map"] = False
        st.session_state["show_reorder_tour_map_accom"] = False

else:
    st.divider() # 구분선
    if st.button("지도 열기"):
        last = st.session_state.get("last_shown_map")

        if last == "show_tour_map":
            st.session_state["show_tour_map"] = True
        elif last == "show_tour_map_accom":
            st.session_state["show_tour_map_accom"] = True
        elif last == "show_reorder_tour_map":
            st.session_state["show_reorder_tour_map"] = True
        elif last == "show_reorder_tour_map_accom":
            st.session_state["show_reorder_tour_map_accom"] = True
            


if st.sidebar.button("🔄 캐시 새로고침"):
    for key in ["cached_gdf_point", "cached_gdf_line", "cached_gdf_point_accom", "cached_gdf_line_accom", "last_request_key"]:
        st.session_state.pop(key, None)
    st.rerun()          





