import requests
import streamlit as st
from streamlit_folium import st_folium
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
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
# 모델 초기화
# llm = ChatOpenAI(model="gpt-5-nano")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
VWORLD_KEY = os.getenv("VWORLD_KEY")
tiles = f"https://api.vworld.kr/req/wmts/1.0.0/{VWORLD_KEY}/Base/{{z}}/{{y}}/{{x}}.png" # Base, white, midnight, Hybrid

# 헬퍼 함수들
def Recomm_to_path(region_name, period):
    print('Recomm')
    system_instructions = (
        "당신은 사용자가 특정 지역과 여행 기간을 입력하면, 그 기간 동안 추천할 여행 코스를 제공하는 도우미입니다.\n"
        f"여행 지역: {region_name}\n"
        f"여행 기간: {period}\n"
        "- 출력은 JSON object 형식이어야 하며, 키는 '1일차', '2일차', ..., 'n일차' 형태이고,"
        "각 키의 값은 다음 두 개의 key를 가진 object입니다:\n"
        "  '테마' (해당 날짜의 여행 테마를 담은 문자열),\n"
        "  '장소들' (해당 날짜에 방문할 3곳에서 7곳 사이의 장소 리스트)\n"
        "- 출력은 반드시 순수 JSON 문자열이어야 합니다.\n"
        "- 마크다운 코드 블록(```json)이나 기타 설명을 절대 포함하지 마세요.\n"
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
            print(place_name, address_name, place_url, coord_x, coord_y)
            
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


def geocode_keyword(region_name, header, params, destination):
    print('geocoding')
    loc_info = requests.get('https://dapi.kakao.com/v2/local/search/address.json?&query=' + region_name,  # 관광지역 검색
                            headers=header, params=params).json()
    ref_destn = [loc_info['documents'][0]['address']['region_1depth_name'], loc_info['documents'][0]['address']['region_2depth_name']] # 관광지역 시도, 시군구 단위

    loc_info = requests.get('https://dapi.kakao.com/v2/local/search/keyword.json?&query=' + destination, # 관광지 검색
                                headers=header, params=params).json()

    place_name = None
    address_name = None
    place_url = None
    coord_x = None
    coord_y = None

    for loc in loc_info['documents']: # 카카오 결과 목록에서
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

    return place_name, address_name, place_url, coord_x, coord_y

def make_clickable(url):
    if url:
        # target="_blank"는 링크를 새 탭에서 열게 합니다.
        return f'<a href="{url}" target="_blank">상세보기</a>'
    return "링크 없음"

@tool
def recommend_travel_course(region_name: str, period: str) -> str:
    """
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
            return "지도를 출력합니다."

    except Exception as e:
        return f"지도 생성 중 오류 발생: {str(e)}"
    

# 도구 바인딩
tools = [recommend_travel_course,
         ]
tool_dict = {
    "recommend_travel_course": recommend_travel_course,
}
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
 
    if gathered.tool_calls:
        st.session_state.messages.append(gathered)
        
        for tool_call in gathered.tool_calls:
            tool_name = tool_call['name']
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call) 
            # print(tool_msg, type(tool_msg))

            st.session_state.messages.append(tool_msg)

            # [핵심] 만약 호출된 도구가 '여행 추천(지도 생성)'이라면?
            if tool_name == "recommend_travel_course":
                # 빈 문자열을 yield하여 스트림을 정상 종료 처리 (선택 사항)
                yield ""
                return  # <--- 여기서 함수 종료! (재귀 호출 안 함)
           
        for chunk in get_ai_response(st.session_state.messages):
            yield chunk

# Streamlit 앱
st.set_page_config(page_title="Tourist Recommender", layout="wide")
st.title("관광추천 챗봇")

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

            gdf_Line.explore(m=m, column='Attr_day', cmap='tab10', legend=True, style_kwds={"weight":5})

            # 지도가 보여질 범위를 설정
            bounds = layer.get_bounds()
            m.fit_bounds(bounds, padding=[50, 50])

            st.markdown(f"{period}간의 {region_name} 여행 지도")
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
        ]):
    st.info("🗺️ 현재 지도 출력 중")
            
    if st.button("지도 닫기"):
        # 모두 끄고 마지막 지도 기록
        if st.session_state["show_tour_map"]:
            st.session_state["last_shown_map"] = "show_tour_map"

        st.session_state["show_tour_map"] = False

else:
    if st.button("지도 열기"):
        last = st.session_state.get("last_shown_map")

        if last == "show_tour_map":
            st.session_state["show_tour_map"] = True


if st.sidebar.button("🔄 캐시 새로고침"):
    for key in ["cached_gdf_point", "cached_gdf_line", "last_request_key"]:
        st.session_state.pop(key, None)
    st.rerun()          