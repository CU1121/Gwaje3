import cv2
import numpy as np

global drawing, mask, brush_size, ix, iy, image, temp_image, selected_mask
drawing = False
brush_size = 20
selected_mask = None

# 마우스 콜백 함수
def draw_circle(event, x, y, flags, param):
    global drawing, ix, iy, mask, temp_image
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(temp_image, (x, y), brush_size, (0, 0, 255), -1)
            cv2.circle(mask, (x, y), brush_size, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(temp_image, (x, y), brush_size, (0, 0, 255), -1)
        cv2.circle(mask, (x, y), brush_size, (255, 255, 255), -1)

# AI 추천 함수 (선택된 영역의 평균 밝기 계산 후 추천)
def ai_recommendation(img, mask):
    selected_area = cv2.bitwise_and(img, img, mask=mask)
    gray_area = cv2.cvtColor(selected_area, cv2.COLOR_BGR2GRAY)
    mean_brightness = cv2.mean(gray_area, mask=mask)[0]

    if mean_brightness < 50:
        return 40  # 매우 어두움 → +40% 추천
    elif mean_brightness < 100:
        return 25  # 어두움 → +25% 추천
    elif mean_brightness < 150:
        return 10  # 약간 어두움 → +10% 추천
    else:
        return -10  # 이미 충분히 밝으면 소폭 감광 추천

# 이미지 밝기 조정 함수
def adjust_brightness(img, mask, percent):
    adjusted = img.copy().astype(np.float32)
    factor = 1 + percent / 100.0

    for c in range(3):
        adjusted[..., c] = np.where(mask > 0, adjusted[..., c] * factor, adjusted[..., c])

    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted

# 메인 함수
def main():
    global image, temp_image, mask

    image_path = input("수정할 이미지 경로를 입력하세요: ")
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 찾을 수 없습니다.")
        return

    mask = np.zeros(image.shape[:2], np.uint8)
    temp_image = image.copy()

    cv2.namedWindow('이미지를 드래그해서 수정 영역 선택')
    cv2.setMouseCallback('이미지를 드래그해서 수정 영역 선택', draw_circle)

    while True:
        cv2.imshow('이미지를 드래그해서 수정 영역 선택', temp_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            temp_image = image.copy()
            mask = np.zeros(image.shape[:2], np.uint8)
            print("영역 선택을 초기화했습니다.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    recommended_value = ai_recommendation(image, mask)
    print(f"AI 추천: 선택한 영역은 평균 밝기 {recommended_value}% 조정이 적절합니다.")

    user_value = input(f"추천 값({recommended_value}%)을 사용하시겠습니까? 아니면 다른 값을 입력하세요: ")
    if user_value.strip() == '':
        brightness_change = recommended_value
    else:
        brightness_change = float(user_value)

    result = adjust_brightness(image, mask, brightness_change)

    cv2.imshow('결과 이미지', result)
    cv2.waitKey(0)
    save_path = input("저장할 파일 이름을 입력하세요 (예: result.jpg): ")
    cv2.imwrite(save_path, result)
    print(f"이미지가 {save_path} 로 저장되었습니다.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
