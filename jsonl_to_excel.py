"""
JSONL 转 Excel 可视化脚本

使用方法:
    python jsonl_to_excel.py \
        --input <JSONL文件路径> \
        [--output_dir <输出目录路径>] \
        [--frame_size <截图宽度, 默认320>] \
        [--limit <限制处理记录数, 默认无限制>]

示例:
python data_process/jsonl_to_excel.py \
    --input ./examples/v1-20260423-122223_checkpoint-300.jsonl \
    --output_dir ./examples/output

字段与媒体识别规则:
     1. JSONL 基本要求:
         - 文件每一行必须是一个合法 JSON 对象（dict）。
         - Excel 列名默认使用第一条记录（第 1 行 JSON）的 key 集合。
         - 若后续记录缺少某个 key，则该单元格留空；若后续记录新增 key，不会自动新增列。

     2. 哪些字段会被识别为“媒体字段”:
         - 字段值是字符串，且以支持的媒体后缀结尾；或
         - 字段值是非空列表，且第一个元素是字符串并以支持的媒体后缀结尾。
         - 支持的视频后缀:
            .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .m4v
         - 支持的图片后缀:
            .jpg, .jpeg, .png, .bmp, .gif, .tiff, .webp

     3. 媒体字段的路径要求:
         - 必须是当前机器可访问的本地文件路径（绝对路径或相对路径均可）。
         - 路径需真实存在，否则预览列显示 [文件不存在]。
         - 若为列表字段，当前仅使用第一个路径生成预览图。

     4. 导出行为:
         - 所有原始字段都会保留为文本列（列表会转为逗号拼接字符串）。
         - 每个媒体字段会额外新增一个 <字段名>_preview 列。
         - 视频取首帧，图片按 frame_size 等比缩放后嵌入 Excel。
         - 预览生成失败时，对应单元格显示 [预览生成失败]。

     5. 输出目录规则:
         - 指定 --output_dir 时输出到该目录。
         - 不指定时，默认在输入文件同级目录创建 _excel 后缀目录。
"""

import os
import json
import shutil
import tempfile
import argparse
from datetime import datetime

try:
    import cv2
except ImportError:
    print("请安装 opencv-python: pip install opencv-python")
    exit(1)

try:
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XlImage
    from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    print("请安装 openpyxl: pip install openpyxl")
    exit(1)

try:
    from PIL import Image as PILImage
except ImportError:
    print("请安装 Pillow: pip install Pillow")
    exit(1)

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS


def get_args():
    parser = argparse.ArgumentParser(description="JSONL 转 Excel 可视化脚本")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录路径 (可选)")
    parser.add_argument("--frame_size", type=int, default=320, help="截图/图片宽度 (默认 320 像素)")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少条数据 (可选，用于快速测试)")
    return parser.parse_args()


def is_media_field(value):
    """判断字段值是否为媒体文件路径，或包含媒体路径的列表"""
    if isinstance(value, str):
        return value.lower().endswith(MEDIA_EXTENSIONS)
    elif isinstance(value, list) and len(value) > 0:
        # 如果是列表，检查第一个元素是否是字符串且以媒体后缀结尾
        first_item = value[0]
        if isinstance(first_item, str):
            return first_item.lower().endswith(MEDIA_EXTENSIONS)
    return False


def get_media_type(path):
    """获取媒体类型"""
    lower = path.lower()
    if lower.endswith(VIDEO_EXTENSIONS):
        return "video"
    elif lower.endswith(IMAGE_EXTENSIONS):
        return "image"
    return None


def extract_video_frame(video_path, output_path, target_width=320):
    """从视频中提取首帧并保存为图片"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  警告: 无法打开视频 {video_path}")
            return False

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print(f"  警告: 无法读取视频帧 {video_path}")
            return False

        h, w = frame.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (target_width, new_h))
        cv2.imwrite(output_path, frame_resized)
        return True
    except Exception as e:
        print(f"  警告: 提取视频帧失败 {video_path}: {e}")
        return False


def resize_image(image_path, output_path, target_width=320):
    """缩放图片到指定宽度"""
    try:
        img = PILImage.open(image_path)
        w, h = img.size
        scale = target_width / w
        new_h = int(h * scale)
        img_resized = img.resize((target_width, new_h), PILImage.LANCZOS)
        img_resized.save(output_path)
        return True
    except Exception as e:
        print(f"  警告: 处理图片失败 {image_path}: {e}")
        return False


def main():
    args = get_args()
    input_abs = os.path.abspath(args.input)

    # 1. 确定输出目录
    if args.output_dir is None:
        input_dir = os.path.dirname(input_abs)
        if 'test_data' in input_dir:
            args.output_dir = input_dir.replace('test_data', 'data_to_excel')
        else:
            args.output_dir = input_dir.rstrip('/') + '_excel'
        print(f"自动生成输出目录: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    output_excel = os.path.join(args.output_dir, f"{input_basename}.xlsx")

    print(f"输入: {args.input}")
    print(f"输出 Excel: {output_excel}")

    # 2. 读取 JSONL
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if args.limit and len(records) >= args.limit:
                break
                
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  警告: 第 {line_no} 行 JSON 解析失败: {e}")

    if not records:
        print("错误: JSONL 文件为空或无有效记录")
        return

    print(f"共读取 {len(records)} 条记录")

    # 3. 分析字段，识别媒体字段
    all_keys = list(records[0].keys())
    media_fields = set()
    for record in records:
        for key in all_keys:
            if key in record and is_media_field(record.get(key, "")):
                media_fields.add(key)

    print(f"检测到媒体字段: {media_fields if media_fields else '无'}")

    # 4. 构建列头
    columns = []
    for key in all_keys:
        columns.append(key)
        if key in media_fields:
            columns.append(f"{key}_preview")

    # 5. 创建 Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "推理结果"

    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell_alignment = Alignment(vertical="top", wrap_text=True)
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    for col_idx, col_name in enumerate(columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border

    ws.freeze_panes = "A2"

    # 6. 创建临时目录存放截图
    tmp_dir = tempfile.mkdtemp(prefix="jsonl_to_excel_")

    try:
        preview_count = 0
        fail_count = 0

        for row_idx, record in enumerate(records, 2):
            col_idx = 1
            row_height = 15

            for key in all_keys:
                value = record.get(key, "")

                # 写入原始字段值
                # 如果是列表，转换为逗号分隔的字符串以便在Excel中显示
                display_value = ", ".join(map(str, value)) if isinstance(value, list) else str(value) if value else ""
                cell = ws.cell(row=row_idx, column=col_idx, value=display_value)
                cell.alignment = cell_alignment
                cell.border = thin_border
                col_idx += 1

                # 媒体字段：生成预览图并嵌入
                if key in media_fields:
                    preview_cell = ws.cell(row=row_idx, column=col_idx)
                    preview_cell.border = thin_border

                    # 如果是列表，我们目前只取第一个路径进行预览展示
                    media_path = str(value[0]) if isinstance(value, list) and len(value) > 0 else str(value)
                    media_type = get_media_type(media_path) if media_path else None

                    if media_type and os.path.exists(media_path):
                        preview_filename = f"r{row_idx}_c{col_idx}.png"
                        preview_path = os.path.join(tmp_dir, preview_filename)

                        success = False
                        if media_type == "video":
                            success = extract_video_frame(media_path, preview_path, args.frame_size)
                        elif media_type == "image":
                            success = resize_image(media_path, preview_path, args.frame_size)

                        if success and os.path.exists(preview_path):
                            img = XlImage(preview_path)
                            pil_img = PILImage.open(preview_path)
                            img_w, img_h = pil_img.size
                            pil_img.close()

                            # 调整列宽
                            col_letter = get_column_letter(col_idx)
                            needed_col_width = img_w / 7 + 2
                            current_width = ws.column_dimensions[col_letter].width or 8
                            if needed_col_width > current_width:
                                ws.column_dimensions[col_letter].width = needed_col_width

                            # 调整行高
                            needed_row_height = img_h * 0.75 + 5
                            if needed_row_height > row_height:
                                row_height = needed_row_height

                            cell_ref = f"{col_letter}{row_idx}"
                            ws.add_image(img, cell_ref)
                            preview_count += 1
                        else:
                            preview_cell.value = "[预览生成失败]"
                            fail_count += 1
                    elif media_path and media_type:
                        preview_cell.value = f"[文件不存在]"
                        fail_count += 1

                    col_idx += 1

            ws.row_dimensions[row_idx].height = row_height

            if (row_idx - 1) % 50 == 0:
                print(f"  已处理 {row_idx - 1}/{len(records)} 条...")

        # 7. 调整非预览列的列宽
        for col_idx, col_name in enumerate(columns, 1):
            if col_name.endswith("_preview"):
                continue
            col_letter = get_column_letter(col_idx)
            if col_name == "prompts":
                ws.column_dimensions[col_letter].width = 50
            elif col_name == "response":
                ws.column_dimensions[col_letter].width = 60
            elif "path" in col_name:
                ws.column_dimensions[col_letter].width = 40
            else:
                ws.column_dimensions[col_letter].width = 30

        # 8. 保存
        wb.save(output_excel)

        print(f"\n完成!")
        print(f"  Excel: {output_excel}")
        print(f"  预览图: {preview_count} 个已嵌入")
        if fail_count > 0:
            print(f"  失败: {fail_count} 个")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("临时文件已清理")


if __name__ == "__main__":
    main()