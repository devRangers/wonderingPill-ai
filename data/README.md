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

3. 전처리 된 이미지 augmentation

```python
$ python
```
