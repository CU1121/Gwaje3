
import cv2
import torch
import numpy as np
import torchvision.transforms as T
import kornia.color as KC

from model import UNetConditionalModel, SimpleEdgeExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

draw_flag = False
mask_sel = None
temp_sel = None

def draw_sel(event, x, y, flags, param):
    global draw_flag, mask_sel, temp_sel
    if event == cv2.EVENT_LBUTTONDOWN:
        draw_flag = True
    elif event == cv2.EVENT_MOUSEMOVE and draw_flag:
        cv2.circle(temp_sel, (x, y), 20, (0, 255, 0), -1)
        cv2.circle(mask_sel, (x, y), 20, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        draw_flag = False

def inference(image_path, brightness, shifts):
    global temp_sel, mask_sel, draw_flag

    image = cv2.imread(image_path)
    temp_sel = image.copy()
    mask_sel = np.zeros(image.shape[:2], dtype=np.uint8)
    draw_flag = False

    cv2.namedWindow("영역 선택 (q: 완료)")
    cv2.setMouseCallback("영역 선택 (q: 완료)", draw_sel)
    while True:
        cv2.imshow("영역 선택 (q: 완료)", temp_sel)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    model = UNetConditionalModel(cond_dim=3).to(device)
    structure_model = SimpleEdgeExtractor().to(device)
    model.load_state_dict(torch.load("final.pth", map_location=device))
    model.eval()
    structure_model.eval()

    transform = T.Compose([T.ToPILImage(), T.Resize((256,256)), T.ToTensor()])
    input_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    cond = [brightness/255.0] + [s/255.0 for s in shifts]
    condition_tensor = torch.tensor([cond], dtype=torch.float32).to(device)

    mask_resized = cv2.resize(mask_sel, (input_tensor.shape[3], input_tensor.shape[2]))
    mask_tensor = (torch.from_numpy(mask_resized.astype(np.float32) / 255.0)
                   .unsqueeze(0).unsqueeze(0).to(device))

    b = condition_tensor[:, :1]
    cs = condition_tensor[:, 1:]
    lo_hsv = KC.rgb_to_hsv(input_tensor)
    lo_hsv[:,2:3,:,:] = torch.clamp(lo_hsv[:,2:3,:,:] + b.view(-1,1,1,1) * mask_tensor, 0.0, 1.0)
    lo_b = KC.hsv_to_rgb(lo_hsv)
    lo_bc = torch.clamp(lo_b + cs.view(-1,3,1,1) * mask_tensor, 0.0, 1.0)

    with torch.no_grad():
        struct_map = structure_model(lo_bc)
        residual = model(lo_bc, cs, struct_map)
        out_tensor = torch.clamp(lo_bc + residual, 0.0, 1.0)[0]

    output_img = (out_tensor.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    mask_full  = cv2.resize(mask_sel, (image.shape[1], image.shape[0]))
    mask_3ch   = np.stack([mask_full]*3, axis=2)
    result     = np.where(mask_3ch==255, output_bgr, image)

    cv2.imshow("AI 보정 결과", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
