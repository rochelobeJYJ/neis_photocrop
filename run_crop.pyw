# -*- coding: utf-8 -*-
"""
콘솔 창(검은 화면) 없이 백그라운드에서 GUI 애플리케이션을 실행하기 위한 파일입니다.
"""
import os
import sys

# main.py가 있는 폴더를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import PhotoCropApp

if __name__ == "__main__":
    app = PhotoCropApp()
    app.run()
