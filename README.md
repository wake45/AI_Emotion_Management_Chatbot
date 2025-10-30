# AI_Emotion_Management_Chatbot
AI 감정 관리 챗봇 프로젝트 (업무 스트레스 완화 및 관리)

# 개요
업무에 지친 직장인들을 위한 감정관리 및 대화용 챗봇
대화를 통해 감정 분석을 진행하고 그 감정을 해소하기 위한 콘텐츠 제공

업무시 캠/마이크를 이용한 인식기능 -> 동영상 업로드로 대체
메신저 인식기능 -> 엑셀 업로드로 대체
메일함 인식기능 -> URL 업로드로 대체

# AZURE URL
https://lsu-web-1030-g0fcg9c9fzhneug6.swedencentral-01.azurewebsites.net/

# 기능
## 채팅 기능
## 엑셀 업로드 (streamlit, pandas)
### 1. streamlit의 file_uploader를 사용해서 파일 객체 저장
### 2. pandas의 read_excel을 사용해서 DataFrame으로 변환 (엑셀 내용 문자열 변환)
## 동영상 업로드 (얼굴&감정 인식 : OpenCV, DeepFace / 음성인식 : ffmpeg, whisper)
### 1. streamlit의 file_uploader를 사용해서 파일 객체 저장
### 2. OpenCV 로 동영상을 프레임 단위로 read
### 3. DeepFace 로 각 프레임을 분석하여 감정 추출 {happy:50 , anger:30...}
### 4. ffmpeg를 사용해서 동영상의 오디오만 추출
### 5. whisper를 사용해서 음성을 텍스트로 변환
## URL 업로드 (HTML 파싱 : BeautifulSoup)
### 1. request.get 으로 웹페이지의 HTML 가져오기
### 2. BeautifulSoup 으로 HTML 파싱 (텍스트만 추출)
## Semantic Serach (recommend_content_for_emotion function)
### 1. Azure OpenAI 임베딩 모델을 사용해 입력된 텍스트를 벡터로 변환
### 2. Azure AI Search에 전달하여 유사한 문서 검색
### - Blob Storage에 콘텐츠 제공을 위한 데이터셋(word) 저장
### - Azure AI Search에 Import 하여 Indexer 생성
## Azure OpenAI (use_openAI function)
### 1. 프롬프트 설계 (당신은 따뜻한 감정코치 입니다 / 민감 표현 처리 / Semantic Serarch 콘텐츠 활용 & 추천 / 응답 구조 설계)
### 2. system 및 user 메시지 구성 
### 3. OpenAI API 호출
### 4. 결과 출력

- BeautifulSoup → 웹페이지 텍스트 추출
- Whisper → 음성 → 텍스트 변환
- FFmpeg → 오디오/비디오 변환·추출 툴
- DeepFace → 얼굴 인식·감정 분석
- OpenCV → 이미지/영상 처리 전반

# history
<2025-10-24>
1. 채팅을 통해 감정파악
2. URL(메일), 액셀(메신저), 동영상(얼굴인식,음성인식)을 통해 감정파악 
3. 대화를 통해 감정 파악 잘되는지 확인

<2025-10-27>
1. 동영상 편집 후 동영상 감정인식 확인
2. 감정 파악 후 AI 코칭 및 컨텐츠 제공 기능 개발

<2025-10-28>
1. 대화를 통해서도 감정 분석할 수 있도록 개발
2. 입력 후 디자인 변경
3. 답변시 현재 사용자 감정 값 변경

<2025-10-29>
1. 데이터셋 인덱스 생성하기
2. 감정별 콘텐츠 추천하는 기능 제작
3. 학습할 데이터 찾기

<2025-10-30>
1. AZURE 환경에 업로드 하기(파일 설치 필요)
2. 동영상, 엑셀, url 감정 파악 잘되는지 확인하기

