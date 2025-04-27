import os
import shutil

def flatten_patient_folders(src_parent_folder, dst_parent_folder):
    for class_folder in os.listdir(src_parent_folder):
        class_path = os.path.join(src_parent_folder, class_folder)
        if not os.path.isdir(class_path):
            continue  # 不是文件夹跳过
        print(f"Processing class folder: {class_folder}")

        # 在目标路径创建对应的class文件夹
        dst_class_path = os.path.join(dst_parent_folder, class_folder)
        os.makedirs(dst_class_path, exist_ok=True)

        for patient_folder in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient_folder)
            if not os.path.isdir(patient_path):
                continue  # 不是文件夹跳过

            for file_name in os.listdir(patient_path):
                src_file = os.path.join(patient_path, file_name)
                dst_file = os.path.join(dst_class_path, file_name)

                # 防止重名
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(file_name)
                    count = 1
                    while os.path.exists(dst_file):
                        dst_file = os.path.join(dst_class_path, f"{base}_{count}{ext}")
                        count += 1

                shutil.copy2(src_file, dst_file)  # 注意：这里是copy，不是move！
                print(f"Copied {src_file} -> {dst_file}")

if __name__ == "__main__":
    src_train_path = "/Users/yihaoniu/Desktop/v3_data/val"  # 原来的train路径
    dst_train_path = "/Users/yihaoniu/Desktop/data/val"  # 新的目标路径
    flatten_patient_folders(src_train_path, dst_train_path)
