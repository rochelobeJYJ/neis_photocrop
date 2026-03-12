# -*- coding: utf-8 -*-
"""
사진 출석부 PDF → 개별 학생 사진 크롭 & 파일명 변환 자동화 도구
=================================================================
- PDF 페이지를 고해상도 이미지로 렌더링
- OpenCV 윤곽선 검출로 검은 테두리 상자(사진 영역)를 자동 인식
- 각 상자 아래 텍스트(학생 정보)를 추출하여 파일명에 매핑
- tkinter GUI (100% 한글)
"""

import os
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────
# 상수 / 기본값
# ──────────────────────────────────────────────
DEFAULT_FORMAT = "{학년}_{반}_{번호}_{이름}"
RENDER_DPI = 200            # PDF → 이미지 변환 해상도
MIN_BOX_AREA_RATIO = 0.002  # 페이지 면적 대비 최소 상자 크기 비율
MAX_BOX_AREA_RATIO = 0.15   # 페이지 면적 대비 최대 상자 크기 비율
ASPECT_RATIO_MIN = 0.5      # 세로/가로 비율 최소
ASPECT_RATIO_MAX = 2.5      # 세로/가로 비율 최대
TEXT_SEARCH_MARGIN_Y = 80    # 상자 아래 텍스트 검색 y 여유(px, 렌더 해상도 기준)
BORDER_TRIM_PX = 4           # 크롭 시 테두리 제거 여백(px)


# ──────────────────────────────────────────────
# 핵심 로직: PDF 분석 → 사진 크롭 & 정보 추출
# ──────────────────────────────────────────────

def render_page_to_image(page: fitz.Page, dpi: int = RENDER_DPI) -> np.ndarray:
    """PDF 페이지를 numpy 이미지(BGR)로 렌더링."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    # fitz는 RGB → OpenCV用 BGR 변환
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def detect_photo_boxes(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    이미지에서 검은 테두리 사각형 상자를 검출.
    반환값: [(x, y, w, h), ...] 좌표 목록 (좌상단 기준, 왼→오, 위→아래 정렬)
    """
    h_img, w_img = img_bgr.shape[:2]
    page_area = h_img * w_img
    min_area = page_area * MIN_BOX_AREA_RATIO
    max_area = page_area * MAX_BOX_AREA_RATIO

    # 그레이스케일 → 이진화
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 적응형 이진화로 테두리 선을 강조
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # 모폴로지 연산으로 노이즈 제거 & 선 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        # 다각형 근사 → 사각형에 가까운 것만 선택
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) < 4:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < min_area or area > max_area:
            continue

        aspect = h / w if w > 0 else 0
        if not (ASPECT_RATIO_MIN <= aspect <= ASPECT_RATIO_MAX):
            continue

        boxes.append((x, y, w, h))

    # 중복/겹침 제거 (IoU 기반)
    boxes = _remove_overlapping_boxes(boxes)

    # 정렬: 위→아래(행), 왼→오(열)
    if boxes:
        boxes = _sort_boxes_grid(boxes)

    return boxes


def _remove_overlapping_boxes(boxes: list, iou_threshold: float = 0.3) -> list:
    """겹치는 상자 중 큰 것만 남김."""
    if not boxes:
        return boxes
    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    for box in boxes_sorted:
        x1, y1, w1, h1 = box
        is_dup = False
        for kx, ky, kw, kh in keep:
            # 교차 영역 계산
            ix = max(x1, kx)
            iy = max(y1, ky)
            ix2 = min(x1 + w1, kx + kw)
            iy2 = min(y1 + h1, ky + kh)
            inter = max(0, ix2 - ix) * max(0, iy2 - iy)
            union = w1 * h1 + kw * kh - inter
            if union > 0 and inter / union > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(box)
    return keep


def _sort_boxes_grid(boxes: list) -> list:
    """상자들을 행(row) 기준으로 그룹핑 후 왼→오 정렬."""
    boxes_sorted = sorted(boxes, key=lambda b: b[1])  # y 기준 정렬
    rows = []
    current_row = [boxes_sorted[0]]
    for box in boxes_sorted[1:]:
        # 같은 행 판정: y 좌표 차이가 상자 높이의 50% 이내
        ref_y = current_row[0][1]
        ref_h = current_row[0][3]
        if abs(box[1] - ref_y) < ref_h * 0.5:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)

    result = []
    for row in rows:
        row.sort(key=lambda b: b[0])  # x 기준 정렬
        result.extend(row)
    return result


def _find_photo_bottom_edge(img_bgr: np.ndarray,
                            cy0: int, cy1: int,
                            cx0: int, cx1: int,
                            padding: int = 2) -> int:
    """
    크롭 영역 하단의 흰 여백과 테두리 선을 건너뛰고,
    실제 사진 콘텐츠가 끝나는 지점(y좌표)을 반환.
    - 연속 4줄 이상 어두운 픽셀이 많으면 → 사진 콘텐츠
    - 1~3줄만 어두우면 → 테두리 선으로 간주하고 건너뜀
    """
    region = img_bgr[cy0:cy1, cx0:cx1]
    if region.size == 0:
        return cy1

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 하단 35%만 스캔 (나머지는 확실히 사진이므로 건드리지 않음)
    scan_limit = max(0, int(h * 0.65))
    consecutive_content = 0

    for y in range(h - 1, scan_limit, -1):
        dark_ratio = np.sum(gray[y, :] < 200) / w
        if dark_ratio > 0.15:
            consecutive_content += 1
            if consecutive_content >= 4:
                return cy0 + y + consecutive_content + padding
        else:
            consecutive_content = 0

    return cy1  # 못 찾으면 원래 값 유지


def _text_to_pixel_coords(page: fitz.Page, text_rect: tuple,
                           dpi: int = RENDER_DPI) -> tuple:
    """
    PDF 텍스트 좌표(mediabox 기준) → 렌더링 이미지의 픽셀 좌표 변환.
    페이지 회전(rotation=0, 90, 180, 270)을 자동으로 보정.
    Returns: (px_x0, px_y0, px_x1, px_y1) — 정규화된 픽셀 좌표
    """
    zoom = dpi / 72.0
    mx0, my0, mx1, my1 = text_rect
    rotation = page.rotation % 360
    mb_w = page.mediabox.width
    mb_h = page.mediabox.height

    if rotation == 0:
        px_x0 = mx0 * zoom
        px_y0 = my0 * zoom
        px_x1 = mx1 * zoom
        px_y1 = my1 * zoom
    elif rotation == 90:
        # 90° 회전: px = (mb_h - my) * zoom, py = mx * zoom
        px_x0 = (mb_h - my1) * zoom
        px_y0 = mx0 * zoom
        px_x1 = (mb_h - my0) * zoom
        px_y1 = mx1 * zoom
    elif rotation == 180:
        px_x0 = (mb_w - mx1) * zoom
        px_y0 = (mb_h - my1) * zoom
        px_x1 = (mb_w - mx0) * zoom
        px_y1 = (mb_h - my0) * zoom
    elif rotation == 270:
        px_x0 = my0 * zoom
        px_y0 = (mb_w - mx1) * zoom
        px_x1 = my1 * zoom
        px_y1 = (mb_w - mx0) * zoom
    else:
        # 알 수 없는 회전 — 회전 없음으로 처리
        px_x0 = mx0 * zoom
        px_y0 = my0 * zoom
        px_x1 = mx1 * zoom
        px_y1 = my1 * zoom

    # 좌표 정규화 (x0 < x1, y0 < y1 보장)
    if px_x0 > px_x1:
        px_x0, px_x1 = px_x1, px_x0
    if px_y0 > px_y1:
        px_y0, px_y1 = px_y1, px_y0

    return (px_x0, px_y0, px_x1, px_y1)


def _get_pixel_words(page: fitz.Page, dpi: int = RENDER_DPI) -> list[dict]:
    """
    페이지의 모든 단어를 추출하고 픽셀 좌표로 변환.
    Returns: [{'x0': float, 'y0': float, 'x1': float, 'y1': float, 'text': str}, ...]
    """
    raw_words = page.get_text("words")
    pixel_words = []
    for w in raw_words:
        px = _text_to_pixel_coords(page, w[:4], dpi)
        pixel_words.append({
            'x0': px[0], 'y0': px[1],
            'x1': px[2], 'y1': px[3],
            'text': w[4],
        })
    return pixel_words


def _find_words_in_box(pixel_words: list, bx: int, by: int,
                       bw: int, bh: int, margin: int = 5) -> list[dict]:
    """
    지정된 상자 영역 안에 있는 단어 목록 반환.
    단어의 중심점이 상자 안(+ margin)에 위치하면 포함.
    """
    result = []
    for w in pixel_words:
        cx = (w['x0'] + w['x1']) / 2
        cy = (w['y0'] + w['y1']) / 2
        if (bx - margin <= cx <= bx + bw + margin and
                by - margin <= cy <= by + bh + margin):
            result.append(w)
    return result


def _group_words_to_lines(words: list[dict], y_tolerance: float = 15) -> list[str]:
    """
    단어 목록을 y좌표 기준으로 줄(line)로 그룹핑 후 텍스트 합치기.
    Returns: 줄별 텍스트 리스트 (위→아래 순)
    """
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w['y0'], w['x0']))
    lines = []
    current_line = [sorted_words[0]]

    for w in sorted_words[1:]:
        if abs(w['y0'] - current_line[0]['y0']) < y_tolerance:
            current_line.append(w)
        else:
            current_line.sort(key=lambda w: w['x0'])
            lines.append(' '.join(w['text'] for w in current_line))
            current_line = [w]

    current_line.sort(key=lambda w: w['x0'])
    lines.append(' '.join(w['text'] for w in current_line))
    return lines


def _extract_header_info(pixel_words: list, img_height: int) -> dict:
    """
    페이지 상단 영역(상위 20%)에서 학년/반 공통 정보를 추출.
    개별 상자에 학년/반 정보가 없는 양식에 대비.
    """
    header_words = [w for w in pixel_words if w['y1'] < img_height * 0.20]
    if not header_words:
        return {}
    header_text = ' '.join(
        w['text'] for w in sorted(header_words, key=lambda w: (w['y0'], w['x0']))
    )
    info = {}
    m = re.search(r'(\d+)\s*학년', header_text)
    if m:
        info['학년'] = m.group(1)
    m = re.search(r'(\d+)\s*반', header_text)
    if m:
        info['반'] = m.group(1)
    return info


def parse_student_info(text: str) -> dict:
    """
    추출된 텍스트에서 학년, 반, 번호, 이름 파싱.
    예: '3학년 1반 2번\n김○○' → {'학년': '3', '반': '1', '번호': '2', '이름': '김○○'}
         '3학년 1반 2번 김○○'  → 같은 결과
    """
    info = {}
    # 학년 추출
    m = re.search(r'(\d+)\s*학년', text)
    if m:
        info['학년'] = m.group(1)
    # 반 추출
    m = re.search(r'(\d+)\s*반', text)
    if m:
        info['반'] = m.group(1)
    # 번호 추출
    m = re.search(r'(\d+)\s*번', text)
    if m:
        info['번호'] = m.group(1)
    # 이름: 학년/반/번호 정보 제거 후 남은 텍스트
    name_text = re.sub(r'\d+\s*학년', '', text)
    name_text = re.sub(r'\d+\s*반', '', name_text)
    name_text = re.sub(r'\d+\s*번', '', name_text)
    name_text = name_text.strip()
    # 여러 줄일 수 있으므로 줄바꿈 분리 후 비어있지 않은 마지막 항목
    lines = [l.strip() for l in name_text.split('\n') if l.strip()]
    if lines:
        info['이름'] = lines[-1]

    return info


def build_filename(fmt: str, info: dict, fallback_index: int) -> str:
    """
    사용자 포맷 문자열에 학생 정보를 대입하여 파일명 생성.
    정보가 부족하면 None 반환 → 미확인 처리.
    """
    required_keys = re.findall(r'\{(\w+)\}', fmt)
    if not required_keys:
        return None

    # 모든 필수 키가 존재하는지 확인
    for key in required_keys:
        if key not in info or not info[key]:
            return None

    try:
        filename = fmt.format(**info)
        # 파일명에 사용 불가 문자 제거
        filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
        return filename
    except (KeyError, ValueError):
        return None


def process_pdf(pdf_path: str, output_dir: str, name_format: str,
                marg_top: int = 0, marg_bottom: int = 0,
                marg_left: int = 0, marg_right: int = 0,
                progress_callback=None, status_callback=None) -> dict:
    """
    메인 처리 함수.
    페이지 회전(rotation=0/90/180/270)을 자동 보정하여
    다양한 PDF 양식(나이스 등)을 안정적으로 처리.
    Returns: {'성공': int, '미확인': int, '총_상자': int, '페이지': int}
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    success_count = 0
    unknown_count = 0
    total_boxes = 0

    # 미확인 폴더 생성
    unknown_dir = os.path.join(output_dir, "미확인_사진")

    if status_callback:
        status_callback(f"PDF 로드 완료 — 총 {total_pages}페이지")

    for page_idx in range(total_pages):
        page = doc[page_idx]
        if status_callback:
            status_callback(f"[{page_idx + 1}/{total_pages}] 페이지 분석 중...")

        # 1) 페이지를 이미지로 렌더링
        img_bgr = render_page_to_image(page, RENDER_DPI)
        h_img, w_img = img_bgr.shape[:2]

        # 2) 페이지의 모든 텍스트 단어를 픽셀 좌표로 변환
        #    (페이지 회전을 자동 보정 — 핵심 개선점)
        pixel_words = _get_pixel_words(page, RENDER_DPI)

        # 3) 검은 테두리 상자 검출
        boxes = detect_photo_boxes(img_bgr)
        total_boxes += len(boxes)

        if not boxes:
            if status_callback:
                status_callback(f"[{page_idx + 1}/{total_pages}] 사진 상자 없음, 건너뜀")
            if progress_callback:
                progress_callback((page_idx + 1) / total_pages * 100)
            continue

        # 4) 페이지 헤더에서 학년/반 공통 정보 추출 (보충용)
        header_info = _extract_header_info(pixel_words, h_img)

        # 5) 각 상자 처리
        for box_idx, (bx, by, bw, bh) in enumerate(boxes):
            try:
                # 상자 영역 안의 단어 찾기 (픽셀 좌표 기반 매칭)
                box_words = _find_words_in_box(pixel_words, bx, by, bw, bh)

                # 상자 하위 55%를 텍스트(정보) 영역으로 간주
                half_y = by + bh * 0.45
                bottom_words = [w for w in box_words if w['y0'] > half_y]

                if bottom_words:
                    # 정보 텍스트의 가장 위쪽 y좌표 (픽셀)
                    info_top_y_px = min(w['y0'] for w in bottom_words)
                    # 단어들을 줄 단위로 그룹핑
                    text_lines = _group_words_to_lines(bottom_words)
                    text = '\n'.join(text_lines)
                else:
                    # 하단에 텍스트가 없으면 상자 전체 텍스트 사용
                    info_top_y_px = by + bh * 0.80
                    text_lines = _group_words_to_lines(box_words)
                    text = '\n'.join(text_lines)

                # 크롭 영역 설정 (테두리 제거 + 텍스트 영역 제외)
                trim = BORDER_TRIM_PX
                cx0 = max(0, bx + trim + marg_left)
                cy0 = max(0, by + trim + marg_top)
                cx1 = min(w_img, bx + bw - trim - marg_right)
                cy1 = min(h_img, int(info_top_y_px) - 2)

                # cy1이 cy0보다 작으면 최소 영역 보장 (상자의 75% 지점)
                if cy1 <= cy0:
                    cy1 = by + int(bh * 0.75)

                # 하단 흰 여백 + 테두리 선 자동 제거
                cy1 = _find_photo_bottom_edge(img_bgr, cy0, cy1, cx0, cx1)

                cy1 = max(cy0 + 1, cy1 - marg_bottom)

                cropped = img_bgr[cy0:cy1, cx0:cx1]

                if cropped.size == 0:
                    raise ValueError("크롭 영역이 비어 있습니다.")

                # PIL Image로 변환 (저장용)
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cropped_rgb)

                # 학생 정보 파싱 & 파일명 생성
                info = parse_student_info(text)

                # 헤더 정보로 보충 (개별 상자에 학년/반 정보가 없는 경우)
                if '학년' not in info and '학년' in header_info:
                    info['학년'] = header_info['학년']
                if '반' not in info and '반' in header_info:
                    info['반'] = header_info['반']

                filename = build_filename(name_format, info, unknown_count + 1)

                if filename:
                    save_path = os.path.join(output_dir, f"{filename}.png")
                    # 동일 파일명 존재 시 넘버링
                    if os.path.exists(save_path):
                        base, ext = os.path.splitext(save_path)
                        counter = 2
                        while os.path.exists(f"{base}_{counter}{ext}"):
                            counter += 1
                        save_path = f"{base}_{counter}{ext}"
                    pil_img.save(save_path, "PNG")
                    success_count += 1
                else:
                    # 미확인 처리
                    os.makedirs(unknown_dir, exist_ok=True)
                    unknown_count += 1
                    save_path = os.path.join(unknown_dir, f"알수없음_{unknown_count}.png")
                    pil_img.save(save_path, "PNG")

            except Exception:
                # 어떤 오류든 작업 중단하지 않고 미확인으로 저장
                try:
                    os.makedirs(unknown_dir, exist_ok=True)
                    unknown_count += 1
                    trim = BORDER_TRIM_PX
                    cx0 = max(0, bx + trim + marg_left)
                    cy0 = max(0, by + trim + marg_top)
                    cx1 = min(w_img, bx + bw - trim - marg_right)
                    cy1 = min(h_img, by + int(bh * 0.75) - marg_bottom)
                    cy1 = max(cy0 + 1, cy1)
                    cropped = img_bgr[cy0:cy1, cx0:cx1]
                    if cropped.size > 0:
                        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(cropped_rgb)
                        save_path = os.path.join(unknown_dir, f"알수없음_{unknown_count}.png")
                        pil_img.save(save_path, "PNG")
                except Exception:
                    pass  # 최종 실패 시 그냥 건너뜀

        if progress_callback:
            progress_callback((page_idx + 1) / total_pages * 100)

    doc.close()
    return {
        '성공': success_count,
        '미확인': unknown_count,
        '총_상자': total_boxes,
        '페이지': total_pages,
    }


# ──────────────────────────────────────────────
# GUI (tkinter)
# ──────────────────────────────────────────────

class PhotoCropApp:
    """사진 출석부 PDF 크롭 도구 GUI."""

    WINDOW_TITLE = "📸 사진 출석부 크롭 도구"
    WINDOW_SIZE = "620x600"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.resizable(False, False)
        
        # 아이콘 설정
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception:
                pass

        # 스타일
        self.root.configure(bg="#F5F5F5")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("맑은 고딕", 10), padding=6)
        style.configure("Accent.TButton", font=("맑은 고딕", 11, "bold"),
                        padding=10, background="#4CAF50", foreground="white")
        style.map("Accent.TButton",
                  background=[("active", "#45A049"), ("disabled", "#A5D6A7")])
        style.configure("TLabel", font=("맑은 고딕", 10), background="#F5F5F5")
        style.configure("Header.TLabel", font=("맑은 고딕", 14, "bold"),
                        background="#F5F5F5", foreground="#333333")
        style.configure("TEntry", font=("맑은 고딕", 10), padding=4)
        style.configure("TLabelframe", font=("맑은 고딕", 10),
                        background="#F5F5F5")
        style.configure("TLabelframe.Label", font=("맑은 고딕", 10, "bold"),
                        background="#F5F5F5", foreground="#555555")

        self._build_ui()

    def _build_ui(self):
        root = self.root
        pad = {"padx": 15, "pady": 5}

        # ─── 제목 ───
        header_frame = tk.Frame(root, bg="#3F51B5", height=56)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="📸 사진 출석부 크롭 도구",
                 font=("맑은 고딕", 15, "bold"), fg="white",
                 bg="#3F51B5").pack(expand=True)

        spacer = tk.Frame(root, height=10, bg="#F5F5F5")
        spacer.pack(fill="x")

        # ─── PDF 파일 선택 ───
        frame_pdf = ttk.LabelFrame(root, text=" 1️⃣  PDF 파일 선택 ")
        frame_pdf.pack(fill="x", **pad)

        row_pdf = tk.Frame(frame_pdf, bg="#F5F5F5")
        row_pdf.pack(fill="x", padx=10, pady=8)

        self.var_pdf = tk.StringVar(value="선택된 파일 없음")
        ttk.Button(row_pdf, text="📂 파일 찾기",
                   command=self._select_pdf).pack(side="left")
        ttk.Label(row_pdf, textvariable=self.var_pdf,
                  foreground="#666666").pack(side="left", padx=10)

        # ─── 저장 폴더 지정 ───
        frame_out = ttk.LabelFrame(root, text=" 2️⃣  저장 폴더 지정 ")
        frame_out.pack(fill="x", **pad)

        row_out = tk.Frame(frame_out, bg="#F5F5F5")
        row_out.pack(fill="x", padx=10, pady=8)

        self.var_output = tk.StringVar(value="지정된 폴더 없음")
        ttk.Button(row_out, text="📁 폴더 선택",
                   command=self._select_output).pack(side="left")
        ttk.Label(row_out, textvariable=self.var_output,
                  foreground="#666666").pack(side="left", padx=10)

        # ─── 파일명 포맷 ───
        frame_fmt = ttk.LabelFrame(root, text=" 3️⃣  파일명 변환 포맷 ")
        frame_fmt.pack(fill="x", **pad)

        row_fmt = tk.Frame(frame_fmt, bg="#F5F5F5")
        row_fmt.pack(fill="x", padx=10, pady=8)

        self.var_format = tk.StringVar(value=DEFAULT_FORMAT)
        ttk.Entry(row_fmt, textvariable=self.var_format,
                  width=40).pack(side="left")
        ttk.Label(row_fmt, text="  사용 가능: {학년} {반} {번호} {이름}",
                  foreground="#999999").pack(side="left", padx=5)

        # ─── 크롭 미세조정 옵션 ───
        frame_margin = ttk.LabelFrame(root, text=" 4️⃣  크롭 추가 제거 (미세 조정) ")
        frame_margin.pack(fill="x", **pad)

        row_margin = tk.Frame(frame_margin, bg="#F5F5F5")
        row_margin.pack(fill="x", padx=10, pady=8)

        self.var_mt = tk.IntVar(value=0)
        self.var_mb = tk.IntVar(value=0)
        self.var_ml = tk.IntVar(value=0)
        self.var_mr = tk.IntVar(value=0)

        ttk.Label(row_margin, text="상:").pack(side="left")
        ttk.Entry(row_margin, textvariable=self.var_mt, width=4).pack(side="left", padx=(2, 10))
        ttk.Label(row_margin, text="하:").pack(side="left")
        ttk.Entry(row_margin, textvariable=self.var_mb, width=4).pack(side="left", padx=(2, 10))
        ttk.Label(row_margin, text="좌:").pack(side="left")
        ttk.Entry(row_margin, textvariable=self.var_ml, width=4).pack(side="left", padx=(2, 10))
        ttk.Label(row_margin, text="우:").pack(side="left")
        ttk.Entry(row_margin, textvariable=self.var_mr, width=4).pack(side="left", padx=(2, 10))
        ttk.Label(row_margin, text="(기본값 0픽셀, 양수 입력시 해당 방향 안쪽을 더 자름)", foreground="#999999").pack(side="left", padx=5)

        # ─── 실행 버튼 ───
        spacer2 = tk.Frame(root, height=8, bg="#F5F5F5")
        spacer2.pack(fill="x")

        self.btn_run = ttk.Button(root, text="▶  작업 실행",
                                  style="Accent.TButton",
                                  command=self._run_task)
        self.btn_run.pack(pady=8)

        # ─── 진행 상태 ───
        frame_prog = ttk.LabelFrame(root, text=" 진행 상태 ")
        frame_prog.pack(fill="x", **pad)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame_prog, variable=self.progress_var,
            maximum=100, length=560, mode="determinate"
        )
        self.progress_bar.pack(padx=10, pady=(8, 4))

        self.var_status = tk.StringVar(value="대기 중...")
        ttk.Label(frame_prog, textvariable=self.var_status,
                  foreground="#555555").pack(padx=10, pady=(0, 8))

        # ─── 결과 요약 ───
        self.var_result = tk.StringVar(value="")
        self.lbl_result = ttk.Label(root, textvariable=self.var_result,
                                    foreground="#1B5E20",
                                    font=("맑은 고딕", 10))
        self.lbl_result.pack(pady=(5, 10))

        # 내부 상태
        self._pdf_path = None
        self._output_dir = None

    # ─── 이벤트 핸들러 ───

    def _select_pdf(self):
        path = filedialog.askopenfilename(
            title="사진 출석부 PDF 파일을 선택하세요",
            filetypes=[("PDF 파일", "*.pdf"), ("모든 파일", "*.*")]
        )
        if path:
            self._pdf_path = path
            # 긴 경로 축약
            display = path if len(path) < 55 else "..." + path[-52:]
            self.var_pdf.set(display)

    def _select_output(self):
        path = filedialog.askdirectory(title="저장할 폴더를 선택하세요")
        if path:
            self._output_dir = path
            display = path if len(path) < 55 else "..." + path[-52:]
            self.var_output.set(display)

    def _run_task(self):
        # 입력 검증
        if not self._pdf_path or not os.path.isfile(self._pdf_path):
            messagebox.showwarning("알림", "PDF 파일을 먼저 선택해 주세요.")
            return
        if not self._output_dir or not os.path.isdir(self._output_dir):
            messagebox.showwarning("알림", "저장 폴더를 먼저 지정해 주세요.")
            return
        fmt = self.var_format.get().strip()
        if not fmt:
            messagebox.showwarning("알림", "파일명 변환 포맷을 입력해 주세요.")
            return

        try:
            mt = self.var_mt.get()
            mb = self.var_mb.get()
            ml = self.var_ml.get()
            mr = self.var_mr.get()
        except tk.TclError:
            messagebox.showwarning("알림", "크롭 미세 조정 값은 숫자로 입력해 주세요.")
            return

        # UI 잠금
        self.btn_run.configure(state="disabled")
        self.progress_var.set(0)
        self.var_status.set("작업 시작 중...")
        self.var_result.set("")

        # 별도 스레드에서 실행 (GUI 블로킹 방지)
        thread = threading.Thread(
            target=self._worker, args=(self._pdf_path, self._output_dir, fmt, mt, mb, ml, mr),
            daemon=True
        )
        thread.start()

    def _worker(self, pdf_path, output_dir, fmt, mt, mb, ml, mr):
        """백그라운드 작업 스레드."""
        try:
            result = process_pdf(
                pdf_path, output_dir, fmt,
                marg_top=mt, marg_bottom=mb, marg_left=ml, marg_right=mr,
                progress_callback=self._on_progress,
                status_callback=self._on_status,
            )
            self.root.after(0, self._on_done, result)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _on_progress(self, value: float):
        self.root.after(0, self.progress_var.set, min(value, 100))

    def _on_status(self, text: str):
        self.root.after(0, self.var_status.set, text)

    def _on_done(self, result: dict):
        self.progress_var.set(100)
        self.var_status.set("✅ 작업 완료!")
        summary = (
            f"총 {result['페이지']}페이지 / "
            f"검출 상자 {result['총_상자']}개 / "
            f"성공 {result['성공']}개 / "
            f"미확인 {result['미확인']}개"
        )
        self.var_result.set(summary)
        self.btn_run.configure(state="normal")

        if result['미확인'] > 0:
            messagebox.showinfo(
                "작업 완료",
                f"{summary}\n\n"
                f"⚠ 인식 실패 사진 {result['미확인']}건은\n"
                f"'미확인_사진' 폴더에 저장되었습니다."
            )
        else:
            messagebox.showinfo("작업 완료", f"{summary}\n\n모든 사진이 정상 처리되었습니다! 🎉")

    def _on_error(self, msg: str):
        self.var_status.set("❌ 오류 발생")
        self.btn_run.configure(state="normal")
        messagebox.showerror("오류", f"작업 중 오류가 발생했습니다.\n\n{msg}")

    def run(self):
        self.root.mainloop()


# ──────────────────────────────────────────────
# 엔트리 포인트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = PhotoCropApp()
    app.run()
