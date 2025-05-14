
from metadata import analyze_and_generate_metadata
from train import train
from infer import inference

if __name__ == "__main__":
    mode = input("Mode(train/infer): ").strip().lower()
    if mode == "train":
        low = input("원본 이미지 폴더 경로: ")
        enh = input("보정된 이미지 폴더 경로: ")
        analyze_and_generate_metadata(low, enh)
        train(low, enh, f"{enh}/metadata.json")
    elif mode == "infer":
        path = input("이미지 경로: ")
        b = float(input("밝기 조정값 (0~255): "))
        r = float(input("R shift: "))
        g = float(input("G shift: "))
        b2 = float(input("B shift: "))
        inference(path, b, [r, g, b2])
    else:
        print("Unknown mode. 'train' 또는 'infer'를 입력하세요.")
