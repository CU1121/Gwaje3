
import os
import torch

def safe_save(model, path):
    tmp = path + '.tmp'
    try:
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)
        print(f"✅ 저장 완료: {path}")
    except Exception as e:
        if os.path.exists(tmp): os.remove(tmp)
        print(f"❌ 저장 오류: {e}")
