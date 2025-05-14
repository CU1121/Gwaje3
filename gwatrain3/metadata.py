
import os
import cv2
import numpy as np
import json

def analyze_and_generate_metadata(low_dir, enh_dir, save_name="metadata.json"):
    metadata = {}
    low_files = sorted(os.listdir(low_dir))
    enh_files = sorted([
        f for f in os.listdir(enh_dir)
        if not f.startswith('mask_') and f.lower().endswith(('.jpg', '.png'))
    ])

    for low_f, enh_f in zip(low_files, enh_files):
        low_bgr = cv2.imread(os.path.join(low_dir, low_f))
        enh_bgr = cv2.imread(os.path.join(enh_dir, enh_f))
        if low_bgr is None or enh_bgr is None:
            continue

        low_hsv = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        enh_hsv = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        diff_rgb = cv2.absdiff(low_bgr, enh_bgr)
        mask = (cv2.cvtColor(diff_rgb, cv2.COLOR_BGR2GRAY) > 15).astype(np.uint8)*255

        V_low_px = low_hsv[...,2][mask>0]
        V_enh_px = enh_hsv[...,2][mask>0]
        if len(V_low_px) > 0:
            v_diff = float(np.mean(V_enh_px) - np.mean(V_low_px))
        else:
            v_diff = 0.0

        lo_hsv_adj = low_hsv.copy()
        lo_hsv_adj[...,2] = np.clip(lo_hsv_adj[...,2] + v_diff, 0, 255)
        lo_rgb_adj = cv2.cvtColor(lo_hsv_adj.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        color_diff = np.mean(enh_bgr.astype(np.float32) - lo_rgb_adj, axis=(0,1)).tolist()

        metadata[enh_f] = {
            "brightness": v_diff,
            "color_shift": color_diff
        }
        print(v_diff, color_diff)
        cv2.imwrite(os.path.join(enh_dir, f"mask_{enh_f}"), mask)

    with open(os.path.join(enh_dir, save_name), 'w') as f:
        json.dump(metadata, f, indent=4)
    print("✅ 메타데이터 생성 완료.")
