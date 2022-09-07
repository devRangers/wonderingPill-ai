import cv2
import numpy as np
import pandas as pd

def grabcut_custom(preprocess_path):
    src = cv2.imread(f'./data/img_color/{preprocess_path}.jpg')

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
        if key == 13:
            cv2.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK) # 마스크 초기화
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            dst = src * mask2[:, :, np.newaxis]
            cv2.imshow('dst', dst)
            cv2.imwrite(f'./data/img_color/{preprocess_path}_0.jpg', dst)

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    data = pd.read_csv("./data/color_csv/available_in_api.origin.renewal_for_download.csv")
    pill_code = pd.DataFrame(data['code'])

    for i,row in pill_code.iterrows():
        try:
            code = str(row['code'])
            preprocess_path = f'{code}/{code}'

            grabcut_custom(preprocess_path)
        except:
            pass
    
    # grabcut individualy
    # pill_code = [197600345, 200400925, 200403808, 201107450, 201307930, 201403877, 201507425, 201700138, 201700539, 201701023, 201701138, 201701161, 201701380, 201701774, 201803063]

    # for code in pill_code:
    #     try:
    #         preprocess_path = f'{code}/{code}'

    #         grabcut_custom(preprocess_path)
    #     except:
    #         pass
