import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch import nn

# 최종 학습된 조건부 모델 정의 (intent_code 없이 brightness, color_shift만 사용)
class EnhancedConditionalModel(nn.Module):
    def __init__(self, condition_dim=4):  # intent_code 제거
        super(EnhancedConditionalModel, self).__init__()
        self.condition_fc = nn.Linear(condition_dim, 256 * 256)

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, condition):
        batch_size = x.size(0)
        cond_map = self.condition_fc(condition).view(batch_size, 1, 256, 256)
        x = torch.cat([x, cond_map], dim=1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 마우스로 영역 선택 기능 구현
def draw_circle(event, x, y, flags, param):
    global drawing, mask, temp_image
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(temp_image, (x, y), 20, (0, 255, 0), -1)
        cv2.circle(mask, (x, y), 20, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(temp_image, (x, y), 20, (0, 255, 0), -1)
        cv2.circle(mask, (x, y), 20, 255, -1)

# AI를 통해 보정 수행 함수 (intent_code 제거 버전)
def apply_ai_enhancement(image_path, brightness_change, color_shift, mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EnhancedConditionalModel()
    model.load_state_dict(torch.load("final_conditional_model.pth", map_location=device))
    model.to(device)
    model.eval()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    condition_vector = torch.tensor([[brightness_change] + color_shift], dtype=torch.float32).to(device)

    with torch.no_grad():
        enhanced_tensor = model(input_tensor, condition_vector).squeeze(0).cpu()

    enhanced_image = enhanced_tensor.permute(1, 2, 0).numpy()
    enhanced_image = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)
    enhanced_image_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)

    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_3ch = cv2.merge([mask_resized, mask_resized, mask_resized])

    final_result = np.where(mask_3ch == 255, enhanced_image_bgr, image)

    cv2.imshow("AI 보정 결과", final_result)
    cv2.waitKey(0)
    save_path = input("저장할 파일명을 입력하세요 (예: final_output.jpg): ")
    cv2.imwrite(save_path, final_result)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = input("보정할 이미지 경로를 입력하세요: ")
    image = cv2.imread(image_path)
    temp_image = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.namedWindow("마우스로 영역 선택 (q로 완료)")
    cv2.setMouseCallback("마우스로 영역 선택 (q로 완료)", draw_circle)

    drawing = False
    while True:
        cv2.imshow("마우스로 영역 선택 (q로 완료)", temp_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    brightness_change = float(input("밝기 조정값 입력 (예: 20, -10): "))
    r_shift = float(input("R 채널 조정값 (예: 5): "))
    g_shift = float(input("G 채널 조정값 (예: -2): "))
    b_shift = float(input("B 채널 조정값 (예: 3): "))

    apply_ai_enhancement(image_path, brightness_change, [r_shift, g_shift, b_shift], mask)
