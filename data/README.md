## 의약품안전나라 의약품 낱알식별 데이터셋

- [pills_data.csv](https://nedrug.mfds.go.kr/pbp/CCBGA01/getItem?totalPages=4&limit=10&page=2&&openDataInfoSeq=11)

1. 의약품 낱알식품.csv의 url로부터 알약 이미지 다운로드

```python
$ cd data
$ python download_url_to_img.py
```

2. 다운로드한 원본 이미지 배경 전처리

```python
$ python grabcut.py
```

3. 이미지 전경 전처리 custom 기능 동작을 원할 경우

- custom할 알약 품목명 리스트를 입력 후 실행
- 마우스 왼쪽 버튼 드래그: 전경 복구
- 마우스 오른쪽 버튼 드래그: 전경 제거
- Enter: 전처리 후 이미지 보기 및 저장
- q : quit

```python
$ python grabcut_custom.py
```

4. 전처리 된 이미지 augmentation

```python
$ python
```
