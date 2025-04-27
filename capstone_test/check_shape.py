import os
from PIL import Image


def check_image_sizes(folder_path):
    sizes = {}

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    size = img.size  # (width, height)
                    if size not in sizes:
                        sizes[size] = 1
                    else:
                        sizes[size] += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

    print("\n📸 统计完毕！图片尺寸情况如下：")
    for size, count in sizes.items():
        print(f"Size {size}: {count} images")


# 用法示例
check_image_sizes('/Users/yihaoniu/Desktop/data/train/2')

