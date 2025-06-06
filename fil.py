import os
import shutil

src_mask_dir = "/home/grace/Documents/inf/lars/semantic_masks"
dst_mask_dir = "/home/grace/Documents/inf/lars-new/semantic_masks"
img_dir = "/home/grace/Documents/inf/lars-new/images"

# Lấy danh sách tên file (không extension) trong images
img_names = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))}

# Copy mask nếu tên trùng
for mask_file in os.listdir(src_mask_dir):
    name, ext = os.path.splitext(mask_file)
    if name in img_names:
        src_path = os.path.join(src_mask_dir, mask_file)
        dst_path = os.path.join(dst_mask_dir, mask_file)
        shutil.copy2(src_path, dst_path)