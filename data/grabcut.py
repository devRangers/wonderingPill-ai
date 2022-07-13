import cv2
import numpy as np
import pandas as pd

def grabcut(preprocess_path):
    src = cv2.imread(f'./img/{preprocess_path}.jpg')

    mask = np.zeros(src.shape[:2], np.uint8)  # 마스크
    bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델
    fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델

    rc = cv2.selectROI(src)

    cv2.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
    dst = src * mask2[:, :, np.newaxis]

    # 초기 분할 결과 출력
    cv2.imshow('dst', dst)

    # 마우스 이벤트 처리 함수 등록
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(dst, (x, y), 3, (255, 0, 0), -1)
            cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)
            cv2.imshow('dst', dst)
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(dst, (x, y), 3, (0, 0, 255), -1)
            cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
            cv2.imshow('dst', dst)
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                cv2.circle(dst, (x, y), 3, (255, 0, 0), -1)
                cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)
                cv2.imshow('dst', dst)
            elif flags & cv2.EVENT_FLAG_RBUTTON:
                cv2.circle(dst, (x, y), 3, (0, 0, 255), -1)
                cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
                cv2.imshow('dst', dst)

    cv2.setMouseCallback('dst', on_mouse)

    while True:
        key = cv2.waitKey()
        if key == 13:  # ENTER
            # 사용자가 지정한 전경/배경 정보를 활용하여 분할
            cv2.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            dst = src * mask2[:, :, np.newaxis]
            cv2.imshow('dst', dst)
            cv2.imwrite(f'./img/{preprocess_path}_0.jpg', dst)
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    data = pd.read_csv("./pills_data.csv")
    pill_code = pd.DataFrame(data['품목일련번호'])

    for i,row in pill_code.iterrows():
        # test 중 임시 5개
        if i == 5:
            break

        try:
            code = str(row['품목일련번호'])
            preprocess_path = f'{code}/{code}'

            grabcut(preprocess_path)
            # download_image(row['큰제품이미지'], pill_name, save_path)
        except:
            pass