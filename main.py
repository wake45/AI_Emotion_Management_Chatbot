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
        st.experimental_rerun()

    st.info(f"🧠 선택한 감정: **{selected_emotion}**")
    st.title("💬 감정 코칭 챗봇")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("지금 어떤 기분이신가요?")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        system_prompt = f"당신은 감정 코치입니다. 사용자의 현재 감정은 '{selected_emotion}'입니다. 이에 맞춰 공감하고 코칭하세요."
        messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history
        response = openai.chat.completions.create(
            model="dev-gpt-4.1-mini",
            messages=messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -------------------------------
    # 첨부 영역
    # -------------------------------
    with st.expander("📎 파일 및 URL 첨부"):
        uploaded_excel = st.file_uploader("엑셀 파일 업로드", type=["xlsx"])
        uploaded_video = st.file_uploader("동영상 업로드", type=["mp4", "mov"])
        url_input = st.text_input("🔗 URL 입력 (메일, 문서 등)")

    text = ""
    sources = []
    if uploaded_excel:
        st.session_state.chat_history.append({"role": "user", "content": uploaded_excel.name})

        df = pd.read_excel(uploaded_excel)
        sources.append("엑셀")
        text = " ".join(df.astype(str).fillna("").values.flatten())

        prompt = f"사용자의 현재 감정은 '{selected_emotion}'입니다. 다음 내용을 기반으로 감정을 분석하고, 감정 코칭을 해줘: {text}"

        with st.spinner("🧠 감정 코칭 분석 중..."):
            prompt = f"사용자의 현재 감정은 '{selected_emotion}'입니다. 다음 내용을 기반으로 감정을 분석하고, 감정 코칭을 해줘: {text}"
            response = openai.chat.completions.create(
                model="dev-gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )


            reply = response.choices[0].message.content

            # 대화 흐름에 추가
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            # 화면에 표시
            with st.chat_message("assistant"):
                st.success("✅ 분석 완료!")
                st.markdown("🧠 AI 코멘트:")
                st.markdown(reply)

            uploaded_excel = None  # 업로드 초기화

    if uploaded_video:
        st.session_state.chat_history.append({"role": "user", "content": uploaded_video.name})
        # 동영상 얼굴인식
        import cv2 # pip install opencv-python
        from deepface import DeepFace # pip install deepface

        video_path = uploaded_video.name
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        emotions = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 100:  # 100프레임까지만 분석
                break

            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotions.append(result[0]['dominant_emotion'])
            except:
                pass

            frame_count += 1

        cap.release()

        # 동영상 음성인식
        import whisper # pip install openai-whisper

        # Whisper 모델 로딩
        model = whisper.load_model("base")

        # 동영상에서 오디오 추출 (ffmpeg 필요)
        import subprocess # pip install ffmpeg-python
        audio_path = "audio.wav"
        subprocess.run(["ffmpeg", "-i", uploaded_video.name, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path])

        # 음성 인식
        result = model.transcribe(audio_path)
        transcript = result["text"]

        # 감정 요약
        from collections import Counter 
        emotion_summary = Counter(emotions).most_common()

        # 얼굴 감정 분석

        # 얼굴 감정 분석 결과 요약
        text += f"영상에서 추정된 감정: {emotion_summary}\n"
        sources.append("동영상(얼굴)")

        # 음성 인식 결과
        text += f"사용자의 음성 내용: {transcript}\n"
        sources.append("동영상(음성)")

        # GPT에게 전달
        prompt = f"사용자의 현재 감정은 '{selected_emotion}'입니다. 다음 내용을 기반으로 감정을 분석하고, 감정 코칭을 해줘: {text}"
        response = openai.chat.completions.create(
            model="dev-gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        st.markdown(f"📎 분석 대상: {', '.join(sources)}")
        st.markdown("🧠 AI 코멘트:")
        st.markdown(response.choices[0].message.content)

        uploaded_video = None  # 업로드 초기화
    if url_input:
        st.session_state.chat_history.append({"role": "user", "content": url_input})

        import requests # pip install requests  
        from bs4 import BeautifulSoup # pip install beautifulsoup4  
        
        try:
            # 1. URL 접속 및 HTML 가져오기
            with st.spinner("🧠 감정 코칭 분석 중..."):
                response = requests.get(url_input)
                soup = BeautifulSoup(response.text, "html.parser")

                # 2. 본문 텍스트 추출 (메일 내용으로 가정)
                # 실제 메일 구조에 따라 태그 조정 필요
                body_text = soup.get_text(separator=" ", strip=True)

                # 3. GPT에게 감정 분석 요청
                prompt = f"다음 메일 내용을 읽고, 보낸 사람의 감정을 분석해줘:\n\n{body_text}"
                response = openai.chat.completions.create(
                    model="dev-gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                reply = response.choices[0].message.content

                # 대화 흐름에 추가
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

                # 화면에 표시
                with st.chat_message("assistant"):
                    st.markdown(reply)
        except Exception as e:
            st.error(f"크롤링 중 오류 발생: {e}")

        url_input = ""  # 입력 초기화