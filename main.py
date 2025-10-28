import streamlit as st # pip install streamlit
import openai # pip install openai
import pandas as pd # pip install pandas
import os
from dotenv import load_dotenv # pip install python-dotenv

load_dotenv()

# OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_ENDPOINT")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")

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

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        system_prompt = (
            f"당신은 감정 코치입니다. 사용자의 현재 감정은 '{selected_emotion}'입니다. "
            f"사용자의 대화를 분석하고 상대방이 필요로 할때 공감과 코칭을 제공하세요. "
            f"또한 필요하다면 감정에 맞는 외부 콘텐츠(유튜브 영상, 글귀, 시, 간단한 게임 링크 등)를 "
            f"간단한 설명과 함께 URL로 추천하세요."
        )

        messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history
        response = openai.chat.completions.create(
            model="dev-gpt-4.1-mini",
            messages=messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        return_new_emotion(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

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
        st.session_state.chat_history.append({"role": "user", "content": uploaded_excel.name})

        df = pd.read_excel(uploaded_excel)
        sources.append("엑셀")
        text = " ".join(df.astype(str).fillna("").values.flatten())

        prompt = f"""
        사용자의 현재 감정은 '{selected_emotion}'입니다.
        다음 대화 내용을 기반으로 감정을 분석하고, 감정 코칭을 해주세요.
        그리고 이 감정에 맞는 외부 콘텐츠를 추천해주세요.
        YouTube 영상, 글귀, 시, 간단한 게임 링크 등으로 구성해 주세요.
        각 콘텐츠는 간단한 설명과 함께 URL을 포함해주세요.

        대화 내용:
        {text}
        """
        with st.spinner("🧠 감정 코칭 분석 중..."):
            response = openai.chat.completions.create(
                model="dev-gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            reply = response.choices[0].message.content
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

            uploaded_excel = None  # 업로드 초기화
            st.session_state["uploaded_excel"] = None

            st.session_state["upload_key"] = str(int(st.session_state["upload_key"]) + 1)

            st.rerun()

    # -------------------------------
    # 동영상 첨부 (얼굴인식, 음성인식)
    # -------------------------------
    if uploaded_video and "video_processed" not in st.session_state:
        st.session_state.video_processed = True
        st.session_state.chat_history.append({"role": "user", "content": uploaded_video.name})

        import cv2
        from deepface import DeepFace

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
        import whisper
        model = whisper.load_model("base")

        import subprocess
        audio_path = "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-i", uploaded_video.name, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path])

        result = model.transcribe(audio_path)
        transcript = result["text"]

        from collections import Counter
        emotion_summary = Counter(emotions).most_common()

        text = ""
        text += f"영상에서 추정된 감정: {emotion_summary}\n"
        sources.append("동영상(얼굴)")
        text += f"사용자의 음성 내용: {transcript}\n"
        sources.append("동영상(음성)")

        # GPT에게 전달
        prompt = f"""
        사용자의 현재 감정은 '{selected_emotion}'입니다.
        다음 내용을 기반으로 감정을 분석하고, 감정 코칭을 해주세요.
        그리고 이 감정에 맞는 외부 콘텐츠를 추천해주세요.
        YouTube 영상, 글귀, 시, 간단한 게임 링크 등으로 구성해 주세요.
        각 콘텐츠는 간단한 설명과 함께 URL을 포함해주세요.

        분석된 내용:
        {text}
        """

        with st.spinner("🧠 감정 코칭 분석 중..."):
            response = openai.chat.completions.create(
                model="dev-gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            reply = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            st.markdown(f"📎 분석 대상: {', '.join(sources)}")
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

            uploaded_video = None
            if "video_processed" in st.session_state:
                del st.session_state["video_processed"]

            # 🔑 첨부 영역 초기화
            st.session_state["upload_key"] = str(int(st.session_state["upload_key"]) + 1)

            st.rerun()

    # -------------------------------
    # URL 첨부 (메일함)
    # -------------------------------    
    if url_input:
        st.session_state.chat_history.append({"role": "user", "content": url_input})

        import requests
        from bs4 import BeautifulSoup

        try:
            with st.spinner("🧠 감정 코칭 분석 중..."):
                # 1. URL 접속 및 HTML 가져오기
                response = requests.get(url_input)
                soup = BeautifulSoup(response.text, "html.parser")

                # 2. 본문 텍스트 추출
                body_text = soup.get_text(separator=" ", strip=True)

                # 3. GPT에게 감정 분석 + 콘텐츠 추천 요청
                prompt = f"""
                사용자의 현재 감정은 '{selected_emotion}'입니다.
                다음 웹페이지 내용을 기반으로 감정을 분석하고, 감정 코칭을 해주세요.
                그리고 이 감정에 맞는 외부 콘텐츠를 추천해주세요.
                YouTube 영상, 글귀, 시, 간단한 게임 링크 등으로 구성해 주세요.
                각 콘텐츠는 간단한 설명과 함께 URL을 포함해주세요.

                웹페이지 내용:
                {body_text}
                """

                response = openai.chat.completions.create(
                    model="dev-gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                reply = response.choices[0].message.content
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

        except Exception as e:
            st.error(f"크롤링 중 오류 발생: {e}")

        url_input = ""
        key = "url_" + st.session_state["upload_key"]
        if key in st.session_state:
            del st.session_state[key]

        # 🔑 첨부 영역 초기화
        st.session_state["upload_key"] = str(int(st.session_state["upload_key"]) + 1)

        st.rerun()
