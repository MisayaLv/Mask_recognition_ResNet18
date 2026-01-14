import os
import random
import shutil


def split_mask_data(src_dir, dest_dir, val_ratio=0.2):
    # 你的三个类别名，必须和文件夹名完全一致
    categories = ['with_mask', 'without_mask', 'mask_weared_incorrect']

    for cat in categories:
        # 创建目标目录
        train_path = os.path.join(dest_dir, 'train', cat)
        val_path = os.path.join(dest_dir, 'val', cat)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        # 获取该类别下所有图片
        src_cat_path = os.path.join(src_dir, cat)
        images = [f for f in os.listdir(src_cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 随机打乱
        random.shuffle(images)

        # 计算划分数量
        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]
        train_images = images[val_count:]

        # 复制文件（用copy而非move，防止出错后原数据丢失）
        for img in train_images:
            shutil.copy(os.path.join(src_cat_path, img), os.path.join(train_path, img))
        for img in val_images:
            shutil.copy(os.path.join(src_cat_path, img), os.path.join(val_path, img))

        print(f"类别 {cat}: 训练集 {len(train_images)} 张, 验证集 {len(val_images)} 张")


if __name__ == "__main__":
    # src_dir: 你刚才裁剪出来的 3 个分类文件夹所在的目录
    # dest_dir: 你想要生成的最终训练目录
    split_mask_data(src_dir='./data', dest_dir='../data', val_ratio=0.2)