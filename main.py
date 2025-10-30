import streamlit as st # pip install streamlit
import openai # pip install openai
import pandas as pd # pip install pandas
import os
import requests
import json
import cv2
import whisper
import subprocess
import requests

from dotenv import load_dotenv # pip install python-dotenv
from deepface import DeepFace
from collections import Counter
from bs4 import BeautifulSoup

load_dotenv()

# OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")

embedding_api_key = os.getenv("EMBEDDING_OPENAI_API_KEY")
embedding_azure_endpoint = os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT")
embedding_api_type = os.getenv("EMBEDDING_OPENAI_API_TYPE")
embedding_api_version = os.getenv("EMBEDDING_OPENAI_API_VERSION")

embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_index = os.getenv("AZURE_SEARCH_INDEX")

st.set_page_config(page_title="AI 감정 코치", layout="wide")

# -------------------------------
# 답변 감정 추출
# -------------------------------
def return_new_emotion(response_text):
    emotion_extract_prompt = f"""
        다음 어시스턴트의 답변을 읽고, 사용자의 현재 감정을 추론해줘.
        가능한 값: 행복, 평온, 슬픔, 분노, 피로, 불안
        출력은 감정 단어만 반환해.
        답변: "{response_text}"
    """
    emotion_response = openai.chat.completions.create(
        model="dev-gpt-4.1-mini",
        messages=[{"role": "system", "content": emotion_extract_prompt}],
        temperature=0.7
    )
    emotion =  emotion_response.choices[0].message.content.strip()

    emotion_map = {
        "행복": "😊 행복",
        "평온": "😐 평온",
        "슬픔": "😢 슬픔",
        "분노": "😠 분노",
        "피로": "😩 피로",
        "불안": "😨 불안"
    }

    if emotion in emotion_map:
        st.session_state["emotion"] = emotion_map[emotion]
        selected_emotion = emotion_map[emotion]

# -------------------------------
# 추천 콘텐츠 제공
# -------------------------------
def recommend_content_for_emotion(user_input):

    # 임베딩
    embedding_url = f"{embedding_azure_endpoint}/openai/deployments/{embedding_deployment}/embeddings?api-version={embedding_api_version}"
    embedding_headers = {
        "api-key": embedding_api_key,
        "Content-Type": "application/json"
    }
    max_length = 2000
    short_text = user_input[:max_length]
    embedding_data = {"input": short_text}
    embedding_response = requests.post(embedding_url, headers=embedding_headers, json=embedding_data)
    query_vector = embedding_response.json()["data"][0]["embedding"]

    # Azure AI Search에 벡터 검색 요청
    search_url = f"{search_endpoint}/indexes/{search_index}/docs/search?api-version=2023-07-01-Preview"
    search_headers = {
        "api-key": search_api_key,
        "Content-Type": "application/json"
    }

    search_body = {
        "vector": {
            "value": query_vector,
            "fields": "text_vector",
            "k": 6
        },
        "select": "chunk"
    }
    
    vector_response = requests.post(search_url, headers=search_headers, json=search_body)
    search_results = vector_response.json().get("value", [])
    return "\n".join([f"- {doc['chunk']}" for doc in search_results])

# -------------------------------
# GPT 프롬포트 사용
# -------------------------------
def use_openAI(user_input, recommended_texts):
    prompt = (
        f"당신은 따뜻한 감정 코치입니다."
        f"다음 내용은 사용자의 입력값입니다. 해당 내용을 기반으로 감정을 분석하고, 감정 코칭을 해주세요.\n"
        f"또한 응답을 위해 입력에 민감하거나 오해될 수 있는 표현이 있더라도, 그대로 반복하지 말고 안전한 언어로 변환해서 응답하세요.\n"
        f"---\n"
        f"{user_input}\n"
        f"---\n"
        f"아래에는 Azure Search에서 반환된 콘텐츠 목록이 있습니다. 각 콘텐츠에는 emotion 값이 포함되어 있습니다.\n"
        f"반드시 이 emotion 값을 확인하여, 사용자의 현재 감정과 일치하는 콘텐츠만 선별해 응답에 활용하세요.\n"
        f"---\n"
        f"{recommended_texts}\n"
        f"---\n"
        f"응답을 구성할 때는:\n"
        f"1. 사용자의 감정을 먼저 공감해 주세요.\n"
        f"2. 검색된 콘텐츠 중 emotion이 사용자의 감정과 일치하는 것만 골라 간단히 소개하세요.\n"
        f"3. 필요하다면 추가적인 외부 콘텐츠(유튜브 영상, 글귀, 시, 간단한 게임 링크 등)를 간단한 설명과 함께 URL로 추천하세요.\n"
        f"4. 전체 톤은 따뜻하고 코칭하는 듯한 어조로 유지하세요."
    )
 
    messages = [{"role": "system", "content": prompt}] + st.session_state.chat_history
    print(messages)

    response = openai.chat.completions.create(
        model="dev-gpt-4.1-mini",
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content

# -------------------------------
# 화면 전환 상태 관리
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"

# -------------------------------
# 메인 페이지
# -------------------------------
if st.session_state.page == "main":
    st.title("🧠 AI 감정 코치")
    st.subheader("오늘 당신의 감정은 어떤가요?")

    emotion = st.radio("감정을 선택해주세요:", ["😊 행복", "😐 평온", "😢 슬픔", "😠 분노", "😩 피로", "😨 불안"])

    if st.button("코칭 시작하기"):
        st.session_state["emotion"] = emotion
        st.session_state["page"] = "chat"
        st.stop()

# -------------------------------
# 챗봇 페이지
# -------------------------------
elif st.session_state.page == "chat":
    selected_emotion = st.session_state.get("emotion", None)

    if not selected_emotion:
        st.warning("감정이 선택되지 않았습니다. 메인 화면으로 돌아갑니다.")
        st.session_state.page = "main"
        st.rerun()

    st.info(f"🧠 선택한 감정: **{selected_emotion}**")
    st.title("💬 감정 코칭 챗봇")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("지금 어떤 기분이신가요?")

    # -------------------------------
    # 챗봇 영역
    # -------------------------------
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🧠 감정 코칭 분석 중..."):

            recommended_texts = recommend_content_for_emotion(user_input)

            reply = use_openAI(user_input, recommended_texts)

        return_new_emotion(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # 화면에 표시
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -------------------------------
    # 첨부 영역
    # -------------------------------
    if "upload_key" not in st.session_state:
        st.session_state["upload_key"] = "1"

    with st.expander("📎 파일 및 URL 첨부"):
        uploaded_excel = st.file_uploader("엑셀 파일 업로드", type=["xlsx"], key="excel_" + st.session_state["upload_key"])
        uploaded_video = st.file_uploader("동영상 업로드", type=["mp4", "mov", "avi"], key="video_" + st.session_state["upload_key"])
        url_input = st.text_input("🔗 URL 입력 (메일, 문서 등)", key="url_" + st.session_state["upload_key"])

    text = ""
    sources = []

    # -------------------------------
    # 엑셀 첨부 (메신저)
    # -------------------------------
    if uploaded_excel:
        df = pd.read_excel(uploaded_excel)
        sources.append("엑셀")
        text = " ".join(df.astype(str).fillna("").values.flatten())

        st.session_state.chat_history.append({
            "role": "user",
            "content": f"엑셀제목 : {uploaded_excel.name}  \n엑셀내용  \n{text}"
        })

        with st.spinner("🧠 감정 코칭 분석 중..."):

            recommended_texts = recommend_content_for_emotion(text)

            try:
                reply = use_openAI(text, recommended_texts)
                

                st.session_state.chat_history.append({"role": "assistant", "content": reply})

                # 화면에 표시
                with st.chat_message("assistant"):
                    st.success("✅ 분석 완료!")
                    st.markdown("🧠 AI 코멘트:")

                    for line in reply.split("\n"):
                        if "http" in line:
                            parts = line.split("http")
                            description = parts[0].strip("-•: ")
                            url = "http" + parts[1].strip()
                            st.markdown(f"**{description}** 👉 [바로가기]({url})")
                        else:
                            st.markdown(line)
                
                return_new_emotion(reply)

                st.session_state["upload_key"] = str(int(st.session_state["upload_key"]) + 1)

                st.rerun()

            except openai.BadRequestError as e:
            # Azure OpenAI 콘텐츠 필터에 걸린 경우
                st.markdown("⚠️ 입력에 민감한 표현이 포함되어 응답이 제한되었습니다. 표현을 조금 바꿔 다시 시도해주세요.")

            uploaded_excel = None  # 업로드 초기화
            st.session_state["uploaded_excel"] = None

    # -------------------------------
    # 동영상 첨부 (얼굴인식, 음성인식)
    # -------------------------------
    if uploaded_video and "video_processed" not in st.session_state:
        st.session_state.video_processed = True

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        video_path = "temp_video.mp4"

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        emotions = []

        max_frames = 100
        progress_bar = st.progress(0)
        status_text = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break

            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotions.append(result[0]['dominant_emotion'])
            except:
                pass

            frame_count += 1
            progress_bar.progress(frame_count / max_frames)
            status_text.text(f"🔍 감정 분석 중... ({frame_count}/{max_frames} 프레임)")

        cap.release()
        progress_bar.empty()
        status_text.text("✅ 얼굴 감정 분석 완료!")

        # 음성 인식
        model = whisper.load_model("base")

        audio_path = "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path])

        result = model.transcribe(audio_path)
        transcript = result["text"]

        emotion_summary = Counter(emotions).most_common()
        dominant_emotion = emotion_summary[0][0] if emotion_summary else "unknown"

        text = f"영상에서 추정된 감정은 '{dominant_emotion}'입니다.  \n"
        #text += f"사용자의 음성 내용: {transcript}  \n"
        sources.append("동영상(얼굴)")
        #sources.append("동영상(음성)")
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": f"영상제목 : {uploaded_video.name}  \n영상내용  \n{text}"
        })

        with st.spinner("🧠 감정 코칭 분석 중..."):

            recommended_texts = recommend_content_for_emotion(text)

            try:
                reply = use_openAI(text, recommended_texts)

                st.session_state.chat_history.append({"role": "assistant", "content": reply})

                st.markdown(f"📎 분석 대상: {', '.join(sources)}")
                st.markdown("🧠 AI 코멘트:")

                # 화면에 표시
                for line in reply.split("\n"):
                    if "http" in line:
                        parts = line.split("http")
                        description = parts[0].strip("-•: ")
                        url = "http" + parts[1].strip()
                        st.markdown(f"**{description}** 👉 [바로가기]({url})")
                    else:
                        st.markdown(line)

                return_new_emotion(reply)

                # 🔑 첨부 영역 초기화
                st.session_state["upload_key"] = str(int(st.session_state["upload_key"]) + 1)

                st.rerun()
            except openai.BadRequestError as e:
            # Azure OpenAI 콘텐츠 필터에 걸린 경우
                st.error("⚠️ 입력에 민감한 표현이 포함되어 응답이 제한되었습니다. 표현을 조금 바꿔 다시 시도해주세요.")
                
            uploaded_video = None
            if "video_processed" in st.session_state:
                del st.session_state["video_processed"]
    # -------------------------------
    # URL 첨부 (메일함)
    # -------------------------------    
    if url_input:
        try:
            with st.spinner("🧠 감정 코칭 분석 중..."):
                # 1. URL 접속 및 HTML 가져오기
                response = requests.get(url_input)
                response.encoding = "utf-8"
                soup = BeautifulSoup(response.text, "html.parser")

                # 2. 본문 텍스트 추출
                text = soup.get_text(separator=" ", strip=True)

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"URL : {url_input}  \n내용  \n{text}"
                })

                recommended_texts = recommend_content_for_emotion(text)

                reply = use_openAI(text, recommended_texts)

                st.session_state.chat_history.append({"role": "assistant", "content": reply})

                return_new_emotion(reply)

                # 화면에 표시
                with st.chat_message("assistant"):
                    st.success("✅ 분석 완료!")
                    st.markdown("🧠 AI 코멘트:")

                    for line in reply.split("\n"):
                        if "http" in line:
                            parts = line.split("http")
                            description = parts[0].strip("-•: ")
                            url = "http" + parts[1].strip()
                            st.markdown(f"**{description}** 👉 [바로가기]({url})")
                        else:
                            st.markdown(line)

                    # 🔑 첨부 영역 초기화
                    st.session_state["upload_key"] = str(int(st.session_state["upload_key"]) + 1)

                    st.rerun()

        except Exception as e:
            st.error(f"크롤링 중 오류 발생: {e}")
        except openai.BadRequestError as e:
        # Azure OpenAI 콘텐츠 필터에 걸린 경우
            st.markdown("⚠️ 입력에 민감한 표현이 포함되어 응답이 제한되었습니다. 표현을 조금 바꿔 다시 시도해주세요.")

        url_input = ""
        key = "url_" + st.session_state["upload_key"]
        if key in st.session_state:
            del st.session_state[key]


