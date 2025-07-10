# 🧪 CLIA 분석 성능 평가 자동화 툴

본 프로젝트는 체외진단 제품의 분석 성능 평가를 자동화하기 위해 제작된 **Streamlit 기반 웹 애플리케이션**입니다.

## 🚀 주요 기능

- CSV 또는 Excel 데이터 업로드
- 민감도, 특이도, 정밀도, 정확도, F1-score 자동 계산
- Confusion Matrix 및 ROC Curve 시각화
- 📄 PDF 보고서 자동 생성 및 다운로드 기능

## 📁 입력 파일 형식

| Sample_ID | True_Label | Test_Result |
|-----------|------------|-------------|
| 001       | 1          | 1           |
| 002       | 0          | 0           |

 <샘플 번호>   <실제 결과>  <기기 테스트 결과>

 1 : 양성
 0 : 음성
 
## 🛠 사용 방법

```streamlit 웹에서 앱 실행
-> (https://clia-eval-tool-kdh-forsdbiosensor.streamlit.app/)
```

## 📎 샘플 데이터

`assets/sample_data.csv` 파일 포함 (예시 샘플 데이터 다운 받아서 업로드 후 결과 확인)

---
