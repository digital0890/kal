# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pytz
from pykalman import KalmanFilter
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import os
import pywt
import streamlit.components.v1 as components

# جایگزینی ایمن برای import pad
try:
    from pywt._dwt import pad
except ImportError:
    from pywt._doc_utils import pad

from sklearn.metrics import mean_squared_error
import warnings
import streamlit as st
from datetime import datetime, timedelta
import time

# تنظیمات اولیه
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# تزریق استایل حرفه‌ای
def inject_pro_style():
    pro_css = """
    <style>
        /* Modern Glassmorphism & Vibrant Gradient Theme */
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #10b981;
            --accent: #8b5cf6;
            --dark: #181c25;
            --darker: #10131c;
            --light: #f1f5f9;
            --gray: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --card-bg: rgba(24, 28, 37, 0.85);
            --card-border: rgba(255, 255, 255, 0.10);
            --glass-blur: 18px;
            --transition-smooth: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-soft: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            --transition-bounce: all 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        }
        html, body, .stApp {
            min-height: 100vh;
            color: var(--light);
            /* Animated gradient background */
            background: linear-gradient(270deg, #232946, #6366f1, #8b5cf6, #10b981, #232946);
            background-size: 400% 400%;
            animation: animatedGradientBG 18s ease infinite;
        }
        @keyframes animatedGradientBG {
            0% {background-position: 0% 50%;}
            25% {background-position: 50% 100%;}
            50% {background-position: 100% 50%;}
            75% {background-position: 50% 0%;}
            100% {background-position: 0% 50%;}
        }
        .stApp {
            font-family: 'Inter', 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: none;
        }
        header.modern-header {
            width: 100vw;
            background: linear-gradient(90deg, #232946 0%, #6366f1 100%);
            box-shadow: 0 4px 24px rgba(99,102,241,0.10);
            padding: 1.2rem 0;
            position: sticky;
            top: 0;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition-smooth);
        }
        .modern-header-content {
            width: 100%;
            max-width: 1400px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
        }
        .modern-logo {
            display: flex;
            align-items: center;
            gap: 14px;
            transition: var(--transition-soft);
        }
        .modern-logo:hover {
            transform: scale(1.02);
        }
        .modern-logo-icon {
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
            width: 48px;
            height: 48px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 18px rgba(99,102,241,0.18);
            transition: var(--transition-bounce);
        }
        .modern-logo-icon:hover {
            transform: rotate(5deg) scale(1.1);
            box-shadow: 0 8px 32px rgba(99,102,241,0.3);
        }
        .modern-logo-text {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            transition: var(--transition-soft);
        }
        .main-container {
            max-width: 1400px;
            margin: 2.5rem auto 2rem auto;
            padding: 0 2.5rem;
            animation: fadeInUp 0.8s ease-out;
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .card {
            background: var(--card-bg);
            border: 1.5px solid var(--card-border);
            border-radius: 24px;
            box-shadow: 0 8px 32px rgba(99,102,241,0.10);
            backdrop-filter: blur(var(--glass-blur));
            padding: 2rem 2rem 1.5rem 2rem;
            margin-bottom: 2rem;
            transition: var(--transition-smooth);
            animation: slideInFromLeft 0.6s ease-out;
        }
        @keyframes slideInFromLeft {
            from {
                opacity: 0;
                transform: translateX(-50px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        .card:hover {
            box-shadow: 0 16px 48px rgba(99,102,241,0.18);
            border-color: var(--primary);
            transform: translateY(-4px) scale(1.01);
        }
        .stSidebar {
            background: linear-gradient(180deg, #232946 0%, #232946 100%);
            border-right: 1.5px solid var(--card-border) !important;
            box-shadow: 2px 0 24px rgba(99,102,241,0.08);
            backdrop-filter: blur(var(--glass-blur));
            transition: var(--transition-soft);
        }
        .stSidebar .sidebar-content {
            padding: 1.5rem 1rem 1rem 1rem;
            animation: slideInFromRight 0.8s ease-out;
        }
        @keyframes slideInFromRight {
            from {
                opacity: 0;
                transform: translateX(50px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        .stSidebar .sidebar-section {
            margin-bottom: 1rem;
            transition: var(--transition-soft);
        }
        .stSidebar .sidebar-section:hover {
            transform: translateX(5px);
        }
        .stSidebar .section-title {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.8rem;
            padding-bottom: 0.5rem;
            border-bottom: 1.5px solid var(--card-border);
            transition: var(--transition-soft);
        }
        .stSidebar .section-title:hover {
            color: var(--accent);
            transform: translateX(3px);
        }
        .stSidebar .section-title svg {
            transition: var(--transition-bounce);
        }
        .stSidebar .section-title:hover svg {
            transform: scale(1.2) rotate(5deg);
        }
        .stButton>button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            border: none;
            border-radius: 14px;
            color: white;
            padding: 1rem 2rem;
            font-weight: 700;
            font-size: 1.1rem;
            cursor: pointer;
            transition: var(--transition-bounce);
            box-shadow: 0 4px 18px rgba(99,102,241,0.18);
            width: 100%;
            margin-top: 0.8rem;
            position: relative;
            overflow: hidden;
        }
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: var(--transition-smooth);
        }
        .stButton>button:hover::before {
            left: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 32px rgba(99,102,241,0.25);
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--accent) 100%);
        }
        .stButton>button:active {
            transform: translateY(0) scale(0.98);
        }
        .stTextInput>div>div>input, .stSelectbox>div>div>select, 
        .stDateInput>div>div>input, .stTimeInput>div>div>input,
        .stNumberInput>div>div>input {
            background: rgba(30, 41, 59, 0.85) !important;
            border: 1.5px solid var(--card-border) !important;
            color: var(--light) !important;
            border-radius: 14px !important;
            padding: 1rem 1.2rem !important;
            font-size: 1.05rem;
            transition: var(--transition-soft);
        }
        .stTextInput>div>div>input:hover, .stSelectbox>div>div>select:hover, 
        .stDateInput>div>div>input:hover, .stTimeInput>div>div>input:hover,
        .stNumberInput>div>div>input:hover {
            border-color: rgba(99,102,241,0.5) !important;
            transform: translateY(-1px);
        }
        .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, 
        .stDateInput>div>div>input:focus, .stTimeInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
            transform: translateY(-2px);
        }
        .stRadio>div {
            flex-direction: row !important;
            gap: 2.2rem;
        }
        .stRadio>div>label {
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(30, 41, 59, 0.7);
            padding: 1.5rem 3rem;
            border-radius: 16px;
            border: 1.5px solid var(--card-border);
            transition: var(--transition-bounce);
            min-width: 180px;
            min-height: 10px;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        .stRadio>div>label::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(99,102,241,0.1), transparent);
            transition: var(--transition-smooth);
        }
        .stRadio>div>label:hover::before {
            left: 100%;
        }
        .stRadio>div>label:hover {
            border-color: var(--primary);
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 24px rgba(99,102,241,0.15);
        }
        .stRadio>div>label[data-baseweb="radio"]>div:first-child {
            background: rgba(30, 41, 59, 0.85) !important;
            border-color: var(--card-border) !important;
            transition: var(--transition-soft);
        }
        .stRadio>div>label[data-baseweb="radio"]>div:first-child>div {
            background: var(--primary) !important;
            transition: var(--transition-bounce);
        }
        .stExpander {
            background: var(--card-bg) !important;
            border-radius: 18px !important;
            border: 1.5px solid var(--card-border) !important;
            box-shadow: 0 4px 18px rgba(99,102,241,0.10);
            margin-bottom: 1.5rem;
            transition: var(--transition-soft);
            animation: fadeInScale 0.5s ease-out;
        }
        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.95);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        .stExpander:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(99,102,241,0.15);
        }
        .stExpanderHeader {
            font-weight: 700;
            color: var(--primary);
            transition: var(--transition-soft);
        }
        .stExpanderHeader:hover {
            color: var(--accent);
        }
        footer.modern-footer {
            width: 100vw;
            background: linear-gradient(90deg, #232946 0%, #6366f1 100%);
            box-shadow: 0 -4px 24px rgba(99,102,241,0.10);
            padding: 2rem 0 1rem 0;
            margin-top: 3rem;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition-smooth);
        }
        .modern-footer-content {
            width: 100%;
            max-width: 1400px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            color: var(--gray);
            font-size: 1rem;
        }
        
        /* استایل جدید برای دکمه‌های ANALYSIS METHOD و VISUALIZATION */
        .method-btn {
            display: inline-block;
            width: 100%;
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 14px;
            color: var(--light);
            font-weight: 700;
            text-align: center;
            cursor: pointer;
            transition: var(--transition-bounce);
            box-shadow: 0 4px 12px rgba(99,102,241,0.1);
            border: 1.5px solid var(--card-border);
            position: relative;
            overflow: hidden;
        }
        
        .method-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: var(--transition-smooth);
        }
        
        .method-btn:hover::before {
            left: 100%;
        }
        
        /* استایل دکمه‌های غیرفعال */
        .method-btn.inactive {
            background: rgba(30, 41, 59, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: var(--gray) !important;
            opacity: 0.7;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        .method-btn.inactive:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }
        
        /* استایل دکمه‌های فعال */
        .method-btn.active {
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
            border: 1.5px solid var(--primary) !important;
            color: white !important;
            opacity: 1;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25) !important;
            font-weight: 700;
        }
        
        .method-btn.active:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
            background: linear-gradient(135deg, var(--primary-dark) 0%, #7c3aed 100%) !important;
        }
        
        /* افکت پالس برای دکمه‌های فعال */
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
            70% { box-shadow: 0 0 0 8px rgba(99, 102, 241, 0); }
            100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
        }
        
        .method-btn.active {
            animation: pulse 2s infinite;
        }
        
        .section-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 1rem;
        }
        
        /* استایل جدید برای دکمه‌های INTERVAL */
        .interval-btn {
            display: inline-block;
            width: 100%;
            padding: 0.7rem 0.5rem;
            margin-bottom: 0.5rem;
            border-radius: 12px;
            font-weight: 600;
            text-align: center;
            cursor: pointer;
            transition: var(--transition-bounce);
            box-shadow: 0 4px 10px rgba(99,102,241,0.1);
            font-size: 0.9rem;
            border: 1.5px solid var(--card-border);
            position: relative;
            overflow: hidden;
        }
        
        .interval-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: var(--transition-smooth);
        }
        
        .interval-btn:hover::before {
            left: 100%;
        }
        
        /* استایل دکمه‌های غیرفعال */
        .interval-btn.inactive {
            background: rgba(30, 41, 59, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: var(--gray) !important;
            opacity: 0.7;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        .interval-btn.inactive:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }
        
        /* استایل دکمه‌های فعال */
        .interval-btn.active {
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
            border: 1.5px solid var(--primary) !important;
            color: white !important;
            opacity: 1;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25) !important;
            font-weight: 700;
        }
        
        .interval-btn.active:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
            background: linear-gradient(135deg, var(--primary-dark) 0%, #7c3aed 100%) !important;
        }
        
        .interval-btn.active {
            animation: pulse 2s infinite;
        }
        
        .interval-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 8px;
            margin-bottom: 1rem;
        }
        
        /* استایل جدول تحلیل */
        .stDataFrame {
            border-radius: 16px;
            border: 1px solid var(--card-border);
            margin-top: 1.5rem;
            transition: var(--transition-soft);
            animation: fadeInUp 0.6s ease-out;
        }
        
        .stDataFrame:hover {
            box-shadow: 0 8px 24px rgba(99,102,241,0.15);
            transform: translateY(-2px);
        }
        
        .stDataFrame th {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
            color: white !important;
            font-weight: 700;
            transition: var(--transition-soft);
        }
        
        .stDataFrame tr:nth-child(even) {
            background-color: rgba(30, 41, 59, 0.5) !important;
            transition: var(--transition-soft);
        }
        
        .stDataFrame tr:hover {
            background-color: rgba(99, 102, 241, 0.2) !important;
            transform: scale(1.01);
        }
        
        /* استایل برای وسط چین کردن لیبل‌ها */
        .stDateInput > label, .stSelectbox > label, .stNumberInput > label {
            text-align: center !important;
            display: block !important;
            margin-bottom: 0.5rem !important;
            font-weight: 600 !important;
            color: var(--light) !important;
            transition: var(--transition-soft);
        }
        
        .stDateInput > label:hover, .stSelectbox > label:hover, .stNumberInput > label:hover {
            color: var(--primary) !important;
        }
        
        /* استایل برای وسط چین کردن متن در selectbox */
        .stSelectbox > div > div > select {
            text-align: center !important;
        }
        
        /* استایل برای وسط چین کردن متن در input ها */
        .stDateInput > div > div > input, .stNumberInput > div > div > input {
            text-align: center !important;
        }
        
        /* انیمیشن برای checkbox */
        .stCheckbox > div > label {
            transition: var(--transition-soft);
        }
        
        .stCheckbox > div > label:hover {
            transform: translateX(3px);
        }
        
        /* انیمیشن برای file uploader */
        .stFileUploader > div {
            transition: var(--transition-soft);
        }
        
        .stFileUploader > div:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(99,102,241,0.15);
        }
        
        /* انیمیشن برای info box */
        .stAlert {
            transition: var(--transition-soft);
            animation: slideInFromTop 0.5s ease-out;
        }
        
        @keyframes slideInFromTop {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .stAlert:hover {
            transform: translateY(-1px);
        }
        
        /* انیمیشن برای spinner */
        .stSpinner > div {
            /* Removed spinning animation */
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        /* انیمیشن برای success/error messages */
        .stSuccess, .stError, .stWarning {
            animation: bounceIn 0.6s ease-out;
        }
        
        @keyframes bounceIn {
            0% {
                opacity: 0;
                transform: scale(0.3);
            }
            50% {
                opacity: 1;
                transform: scale(1.05);
            }
            70% {
                transform: scale(0.9);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        /* انیمیشن برای plotly charts */
        .js-plotly-plot {
            transition: var(--transition-soft);
        }
        
        .js-plotly-plot:hover {
            transform: scale(1.01);
        }
        
        @media (max-width: 900px) {
            .main-container {
                padding: 0 1rem;
            }
            .modern-header-content, .modern-footer-content {
                padding: 0 1rem;
            }
            .section-buttons, .interval-buttons {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 600px) {
            .main-container {
                padding: 0 0.2rem;
            }
            .modern-header-content, .modern-footer-content {
                flex-direction: column;
                gap: 1rem;
                padding: 0 0.2rem;
            }
        }
    </style>
    """
    st.markdown(pro_css, unsafe_allow_html=True)
    # Main container (no hero section)
    st.markdown("""
    <div class="main-container">
    """, unsafe_allow_html=True)

# تبدیل تاریخ به فرمت آگاه از منطقه زمانی
def make_timezone_aware(dt, timezone_str):
    tz = pytz.timezone(timezone_str)
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return tz.localize(dt)
    else:
        return dt.astimezone(tz)

# دانلود داده‌ها
def download_filtered_data(symbol, start_datetime, end_datetime, interval, timezone=None):
    start_dt = pd.to_datetime(start_datetime)
    end_dt = pd.to_datetime(end_datetime)

    if timezone:
        start_dt = make_timezone_aware(start_dt, timezone)
        end_dt = make_timezone_aware(end_dt, timezone)

    data = yf.download(
        symbol,
        start=start_dt.date().isoformat(),
        end=(end_dt + pd.Timedelta(days=1)).date().isoformat(),
        interval=interval
    )

    data.index = pd.to_datetime(data.index)

    if timezone:
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert(timezone)
    else:
        data.index = data.index.tz_localize(None)

    data = data.loc[(data.index >= start_dt) & (data.index <= end_dt)]

    data.rename(columns={
        'Open': f'Open_{symbol}',
        'High': f'High_{symbol}',
        'Low': f'Low_{symbol}',
        'Close': f'Close_{symbol}',
        'Volume': f'Volume_{symbol}'
    }, inplace=True)

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]

    # Find the first column that starts with 'Close' as the close_col
    close_candidates = [col for col in data.columns if str(col).startswith('Close')]
    if not close_candidates:
        return data, None
    close_col = close_candidates[0]

    if 'Adj Close' in data.columns:
        data.drop(columns=['Adj Close'], inplace=True)

    return data, close_col

# یافتن قله‌ها و دره‌ها
def find_peaks_valleys(residuals, window=5):
    peaks = []
    valleys = []
    
    for i in range(window, len(residuals) - window):
        if all(residuals[i] > residuals[i-j] for j in range(1, window+1)) and \
           all(residuals[i] > residuals[i+j] for j in range(1, window+1)):
            peaks.append(i)
        
        if all(residuals[i] < residuals[i-j] for j in range(1, window+1)) and \
           all(residuals[i] < residuals[i+j] for j in range(1, window+1)):
            valleys.append(i)
    
    return peaks, valleys

# محاسبه روند با ویولت
def compute_wavelet_trend(signal):
    wavelets = ['db4', 'sym5', 'coif3', 'bior3.3', 'haar']
    results = {}
    
    if len(signal) > 10:
        wavelet_temp = pywt.Wavelet('db4')
        max_lvl = pywt.dwt_max_level(len(signal), wavelet_temp.dec_len)
        level = max(1, min(max_lvl - 1, 5))
    else:
        level = 1

    for wavelet_name in wavelets:
        try:
            padded_length = 2**level - len(signal) % 2**level if len(signal) % 2**level != 0 else 0
            signal_padded = pad(signal, (0, padded_length), 'symmetric')
            
            coeffs = pywt.wavedec(signal_padded, wavelet_name, mode='periodization', level=level)
            
            uthresh_coeffs = []
            for c in coeffs[1:]:
                if len(c) > 0:
                    sigma = np.median(np.abs(c)) / 0.6745
                    uthresh = sigma * np.sqrt(2 * np.log(len(c)))
                    uthresh_coeffs.append(uthresh)
                else:
                    uthresh_coeffs.append(0)
            
            coeffs_thresh = [coeffs[0]]
            for i in range(1, len(coeffs)):
                coeffs_thresh.append(pywt.threshold(coeffs[i], uthresh_coeffs[i-1], mode='soft'))
            
            trend_padded = pywt.waverec(coeffs_thresh, wavelet_name, mode='periodization')
            trend = trend_padded[:len(signal)]
            
            mse = mean_squared_error(signal, trend)
            results[wavelet_name] = mse
            
        except Exception:
            results[wavelet_name] = float('inf')
    
    best_wavelet = min(results, key=results.get) if results else 'db4'
    
    try:
        padded_length = 2**level - len(signal) % 2**level if len(signal) % 2**level != 0 else 0
        signal_padded = pad(signal, (0, padded_length), 'symmetric')
        
        coeffs = pywt.wavedec(signal_padded, best_wavelet, mode='periodization', level=level)
        
        uthresh_coeffs = []
        for c in coeffs[1:]:
            if len(c) > 0:
                sigma = np.median(np.abs(c)) / 0.6745
                uthresh = sigma * np.sqrt(2 * np.log(len(c)))
                uthresh_coeffs.append(uthresh)
            else:
                uthresh_coeffs.append(0)
        
        coeffs_thresh = [coeffs[0]]
        for i in range(1, len(coeffs)):
            coeffs_thresh.append(pywt.threshold(coeffs[i], uthresh_coeffs[i-1], mode='soft'))
        
        trend_padded = pywt.waverec(coeffs_thresh, best_wavelet, mode='periodization')
        trend = trend_padded[:len(signal)]
        
        return trend, best_wavelet, level
    except Exception:
        return signal, 'db4', 1

# محاسبه اکسترمم‌ها و میانگین‌ها
def compute_extrema_and_averages(residuals, method_type):
    if method_type == 'kalman':
        peaks_idx = argrelextrema(residuals, np.greater, order=3)[0]
        valleys_idx = argrelextrema(residuals, np.less, order=3)[0]
    else:
        peaks_idx, valleys_idx = find_peaks_valleys(residuals, window=5)

    peaks = residuals[peaks_idx]
    valleys = residuals[valleys_idx]

    mean_peak = np.mean(peaks) if len(peaks) > 0 else 0
    mean_valley = np.mean(valleys) if len(valleys) > 0 else 0

    high_peaks = [p for p in peaks if p > mean_peak] if len(peaks) > 0 else []
    low_valleys = [v for v in valleys if v < mean_valley] if len(valleys) > 0 else []

    mean_high_peak = np.mean(high_peaks) if len(high_peaks) > 0 else mean_peak
    mean_low_valley = np.mean(low_valleys) if len(low_valleys) > 0 else mean_valley
    
    filtered_peaks_idx = [i for i in peaks_idx if residuals[i] > mean_peak]
    filtered_valleys_idx = [i for i in valleys_idx if residuals[i] < mean_valley]
    filtered_peaks = residuals[filtered_peaks_idx]
    filtered_valleys = residuals[filtered_valleys_idx]

    return {
        'peaks_idx': peaks_idx,
        'valleys_idx': valleys_idx,
        'peaks': peaks,
        'valleys': valleys,
        'mean_peak': mean_peak,
        'mean_valley': mean_valley,
        'mean_high_peak': mean_high_peak,
        'mean_low_valley': mean_low_valley,
        'filtered_peaks_idx': filtered_peaks_idx,
        'filtered_valleys_idx': filtered_valleys_idx,
        'filtered_peaks': filtered_peaks,
        'filtered_valleys': filtered_valleys
    }

# تابع تشخیص روند بر اساس تایم فریم
def determine_trend(interval, avg_slope, current_price):
    # تنظیم آستانه‌ها بر اساس تایم فریم و قیمت فعلی
    base_price = 2000.0  # قیمت پایه برای طلا
    scale_factor = current_price / base_price
    
    # تعریف آستانه‌ها برای تایم فریم‌های مختلف
    threshold_map = {
        '1m': 0.1 * scale_factor,
        '5m': 0.3 * scale_factor,
        '15m': 0.5 * scale_factor,
        '30m': 1.0 * scale_factor,
        '1h': 2.0 * scale_factor,
        '4h': 5.0 * scale_factor
    }
    
    threshold = threshold_map.get(interval, 1.0 * scale_factor)
    
    # تشخیص روند بر اساس شیب
    if avg_slope > threshold:
        return "صعودی 📈", "#10b981"  # سبز
    elif avg_slope < -threshold:
        return "نزولی 📉", "#ef4444"  # قرمز
    else:
        return "خنثی ↔️", "#94a3b8"  # خاکستری

# تابع اصلی تحلیل
def run_analysis(symbol, start_date, start_hour, start_minute, end_date, end_hour, end_minute, interval, 
                 initial_state_mean, auto_initial_state, show_residual,
                 methods, uploaded_file=None):
    
    # تعریف اینتروال برای تشخیص روند
    interval_for_trend = interval
    
    if uploaded_file is not None:
        try:
            data, close_col = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            data.index = pd.to_datetime(data.index, errors='coerce')
            data = data[~data.index.isna()]
            
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data.dropna(subset=[close_col], inplace=True)
            
            # Ensure close_col is a string and data[close_col] is a Series
            if isinstance(close_col, list):
                close_col = close_col[0]
            if isinstance(data[close_col], pd.DataFrame):
                if data[close_col].shape[1] == 1:
                    only_col = data[close_col].columns[0]
                    # تغییر close_col به نام واقعی ستون
                    close_col = only_col
                    data[close_col] = data[close_col][only_col]
                else:
                    st.error(
                        f"""
                        ❌ Multiple columns found for {close_col}: {list(data[close_col].columns)}
                        \nType: {type(data[close_col])}
                        \nShape: {data[close_col].shape}
                        \nColumns: {data[close_col].columns}
                        \nSample data:\n{data[close_col].head(3).to_string()}
                        """
                    )
                    return
            
            # Check if data is empty after processing
            if data is None or data.empty:
                st.error("❌ Uploaded file is empty or invalid.")
                return
            
            # Calculate initial value from first data point if auto mode is enabled
            if auto_initial_state and len(data) > 0:
                calculated_initial_value = float(data[close_col].iloc[0])
                st.success(f"✅ Data processed successfully | Auto-calculated Initial Value: {calculated_initial_value:.4f}")
                initial_state_mean = calculated_initial_value
            else:
                st.success("✅ Data processed successfully")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return
    else:
        start_time = f"{start_hour.zfill(2)}:{start_minute.zfill(2)}"
        end_time = f"{end_hour.zfill(2)}:{end_minute.zfill(2)}"
        start_datetime = f"{start_date} {start_time}"
        end_datetime = f"{end_date} {end_time}"
        
        timezone = "Asia/Tehran"

        try:
            data, close_col = download_filtered_data(symbol, start_datetime, end_datetime, interval, timezone)
            # Check if data is empty after download
            if data is None or data.empty or close_col is None:
                st.error("❌ No data was downloaded or no Close column found. Please check the symbol, date range, or your internet connection.")
                return
            # Ensure close_col is a string and data[close_col] is a Series
            if isinstance(close_col, list):
                close_col = close_col[0]
            if isinstance(data[close_col], pd.DataFrame):
                if data[close_col].shape[1] == 1:
                    only_col = data[close_col].columns[0]
                    close_col = only_col
                    data[close_col] = data[close_col][only_col]
                else:
                    st.error(
                        f"""
                        ❌ Multiple columns found for {close_col}: {list(data[close_col].columns)}
                        \nType: {type(data[close_col])}
                        \nShape: {data[close_col].shape}
                        \nColumns: {data[close_col].columns}
                        \nSample data:\n{data[close_col].head(3).to_string()}
                        """
                    )
                    return
            
            # Calculate initial value from first data point if auto mode is enabled
            if auto_initial_state and len(data) > 0:
                calculated_initial_value = float(data[close_col].iloc[0])
                st.success(f"✅ Data for {symbol} downloaded successfully | Auto-calculated Initial Value: {calculated_initial_value:.4f}")
                initial_state_mean = calculated_initial_value
            else:
                st.success(f"✅ Data for {symbol} downloaded successfully")
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            return

    # مقداردهی اولیه initial_state_mean برای هر دو حالت Yahoo و آپلود فایل
    if (methods[0] == 'Kalman' or methods[0] == 'Kalman+Wavelet') and len(data) > 0:
        if auto_initial_state:
            initial_state_mean = data[close_col].iloc[0]
        # else: مقدار initial_state_mean همان مقدار ورودی کاربر باقی می‌ماند
        # Ensure initial_state_mean is always a float and not a Series
        if isinstance(initial_state_mean, pd.Series):
            initial_state_mean = float(initial_state_mean.iloc[0])
        else:
            initial_state_mean = float(initial_state_mean)

    results_by_method = {}
    for method in methods:
        analysis_method = method
        if method == 'Kalman':
            try:
                observations = data[close_col].values.reshape(-1, 1)
                kf = KalmanFilter(
                    transition_matrices=[[1.0, 1.0], [0.0, 1.0]],
                    observation_matrices=[[1.0, 0.0]],
                    initial_state_mean=[initial_state_mean, 0.0],
                    n_dim_state=2,
                    n_dim_obs=1
                )
                kf = kf.em(observations, n_iter=70)
                state_means, _ = kf.filter(observations)
                filtered_close = state_means[:, 0]
                filtered_col = f'Filtered_Close_{method}'
                residual_col = f'Residual_{method}'
                data[filtered_col] = filtered_close
                data[residual_col] = data[close_col] - data[filtered_col]
                method_name = 'Kalman'
                
                # محاسبه شیب روند
                slopes = np.diff(data[filtered_col])
                if len(slopes) > 0:
                    if len(slopes) >= 5:
                        avg_slope = np.mean(slopes[-5:])
                    else:
                        avg_slope = np.mean(slopes)
                else:
                    avg_slope = 0
                    
                # تشخیص روند
                current_price = data[close_col].iloc[-1]
                trend_direction, trend_color = determine_trend(interval_for_trend, avg_slope, current_price)
                
                results_by_method[method] = {
                    'method_name': method_name,
                    'filtered_col': filtered_col,
                    'residual_col': residual_col,
                    'trend_direction': trend_direction,
                    'trend_color': trend_color,
                    'avg_slope': avg_slope
                }
            except Exception as e:
                st.error(f"Error applying Kalman filter: {e}")
                return
        elif method == 'Wavelet':
            try:
                signal = data[close_col].values.flatten()
                trend, best_wavelet, level = compute_wavelet_trend(signal)
                filtered_col = f'Filtered_Close_{method}'
                residual_col = f'Residual_{method}'
                data[filtered_col] = trend
                data[residual_col] = data[close_col] - data[filtered_col]
                method_name = f'Wavelet ({best_wavelet}, level: {level})'
                
                # محاسبه شیب روند
                slopes = np.diff(data[filtered_col])
                if len(slopes) > 0:
                    if len(slopes) >= 5:
                        avg_slope = np.mean(slopes[-5:])
                    else:
                        avg_slope = np.mean(slopes)
                else:
                    avg_slope = 0
                    
                # تشخیص روند
                current_price = data[close_col].iloc[-1]
                trend_direction, trend_color = determine_trend(interval_for_trend, avg_slope, current_price)
                
                results_by_method[method] = {
                    'method_name': method_name,
                    'filtered_col': filtered_col,
                    'residual_col': residual_col,
                    'trend_direction': trend_direction,
                    'trend_color': trend_color,
                    'avg_slope': avg_slope
                }
            except Exception as e:
                st.error(f"Error in wavelet analysis: {e}")
                return

    # نمایش نمودارها
    for method in methods:
        method_data = results_by_method[method]
        method_name = method_data['method_name']
        
        # نمایش وضعیت روند
        st.markdown(f"### تحلیل روند ({method_name})")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**تایم فریم:** `{interval_for_trend}`")
        with col2:
            st.markdown(f"**وضعیت فعلی:** <span style='color:{method_data['trend_color']};font-weight:bold;'>{method_data['trend_direction']}</span>", 
                        unsafe_allow_html=True)
        st.markdown(f"**میانگین شیب:** `{method_data['avg_slope']:.6f}`")
        st.markdown("---")
        
        if show_residual:
            residuals = data[method_data['residual_col']].values
            results = compute_extrema_and_averages(residuals, method.lower())

            # --- Resistance and Support Points Table in a separate box ---
            analysis_points = []
            for idx in results['filtered_peaks_idx']:
                dt = data.index[idx]
                analysis_points.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Time": dt.strftime("%H:%M"),
                    "Value": f"{results['filtered_peaks'][results['filtered_peaks_idx'].index(idx)]:.4f}",
                    "Type": "Support"
                })
            for idx in results['filtered_valleys_idx']:
                dt = data.index[idx]
                analysis_points.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Time": dt.strftime("%H:%M"),
                    "Value": f"{results['filtered_valleys'][results['filtered_valleys_idx'].index(idx)]:.4f}",
                    "Type": "Resistance"
                })
            analysis_points.sort(key=lambda x: (x["Date"], x["Time"]))

            with st.expander(f"📌 Resistance and Support Points Analysis - {method}", expanded=True):
                if analysis_points:
                    df_analysis = pd.DataFrame(analysis_points)
                    st.dataframe(
                        df_analysis,
                        column_order=["Date", "Time", "Value", "Type"],
                        hide_index=True,
                        use_container_width=True,
                        height=min(len(analysis_points) * 35 + 35, 500)
                    )
                else:
                    st.warning("No resistance or support points identified in this time period")

            # --- Residual Chart ---
            with st.expander(f"📊 Residual Analysis - {method}", expanded=False):
                RESIDUAL_COLOR = '#6366f1'
                FILTERED_PEAK_COLOR = '#10b981'
                FILTERED_VALLEY_COLOR = '#ec4899'
                MEAN_PEAK_COLOR = '#10b981'
                MEAN_VALLEY_COLOR = '#ec4899'
                ZERO_LINE_COLOR = '#ffffff'
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=data.index,
                    y=data[method_data['residual_col']],
                    mode='lines',
                    name='Residual',
                    line=dict(color=RESIDUAL_COLOR, width=2)
                ))
                fig_res.add_shape(type='line', x0=data.index[0], x1=data.index[-1], y0=0, y1=0, 
                                line=dict(color=ZERO_LINE_COLOR, dash='dot', width=1.5))
                if len(results['filtered_peaks_idx']) > 0:
                    fig_res.add_trace(go.Scatter(
                        x=data.index[results['filtered_peaks_idx']],
                        y=results['filtered_peaks'],
                        mode='markers',
                        name='Filtered Peak',
                        marker=dict(color=FILTERED_PEAK_COLOR, size=10, symbol='triangle-up')
                    ))
                if len(results['filtered_valleys_idx']) > 0:
                    fig_res.add_trace(go.Scatter(
                        x=data.index[results['filtered_valleys_idx']],
                        y=results['filtered_valleys'],
                        mode='markers',
                        name='Filtered Valley',
                        marker=dict(color=FILTERED_VALLEY_COLOR, size=10, symbol='triangle-down')
                    ))
                fig_res.add_hline(
                    y=results['mean_peak'], 
                    line=dict(color=MEAN_PEAK_COLOR, width=2.5, dash='dash'),
                    annotation_text=f"Primary Peak Avg: {results['mean_peak']:.4f}"
                )
                fig_res.add_hline(
                    y=results['mean_valley'], 
                    line=dict(color=MEAN_VALLEY_COLOR, width=2.5, dash='dash'),
                    annotation_text=f"Primary Valley Avg: {results['mean_valley']:.4f}"
                )
                fig_res.update_layout(
                    title=f'Residual Analysis ({method_name})',
                    xaxis_title='Date',
                    yaxis_title='Residual Value',
                    height=600,
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_res, use_container_width=True)
            # END of chart expander

    # نمایش نقاط مشترک بین Kalman و Wavelet
    # (این بخش حذف شد)

# تابع نمایش ویجت‌های TradingView
def show_tradingview_widgets():
    tradingview_html = '''
    <div class="tradingview-widget-container">
      <div class="charts-grid">
        <div class="chart-cell" id="tradingview_30m"><div class="resize-handle">&#8690;</div></div>
        <div class="chart-cell" id="tradingview_15m"><div class="resize-handle">&#8690;</div></div>
        <div class="chart-cell" id="tradingview_3m"><div class="resize-handle">&#8690;</div></div>
        <div class="chart-cell" id="tradingview_1m"><div class="resize-handle">&#8690;</div></div>
      </div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
        const widgetConfig = {
          width: "100%",
          height: "100%",
          autosize: true,
          symbol: "OANDA:XAUUSD",
          timezone: "Asia/Tehran",
          theme: "dark",
          style: "1",
          locale: "fa_IR",
          toolbar_bg: "#131722",
          enable_publishing: true,
          allow_symbol_change: true,
          withdateranges: true,
          hide_side_toolbar: false,
          details: true,
          hotlist: true,
          calendar: true,
          show_popup_button: true,
          popup_width: "1000",
          popup_height: "650",
          save_image: true,
          show_chart_property_settings: true,
          show_symbol_logo: true,
          hideideas: false,
          hide_volume: true,
          watchlist: [
            "OANDA:XAUUSD",
            "OANDA:EURUSD",
            "OANDA:GBPUSD",
            "OANDA:USDJPY",
            "OANDA:USDCAD",
            "OANDA:AUDUSD",
            "OANDA:USDCHF"
          ],
          supported_resolutions: [
            "1",   // 1m
            "3",   // 3m
            "15",  // 15m
            "30"   // 30m
          ]
        };

        function createWidget(containerId, interval) {
          new TradingView.widget({
            ...widgetConfig,
            interval,
            container_id: containerId
          });
        }

        [
          {id: "tradingview_30m", interval: "30"},
          {id: "tradingview_15m", interval: "15"},
          {id: "tradingview_3m", interval: "3"},
          {id: "tradingview_1m", interval: "1"}
        ].forEach(cfg => createWidget(cfg.id, cfg.interval));

        // Manual resizing logic for chart cells + double click to reset
        document.querySelectorAll('.chart-cell').forEach(cell => {
          const handle = cell.querySelector('.resize-handle');
          if (!handle) return;
          let isResizing = false, lastX = 0, lastY = 0, startW = 0, startH = 0;

          // ذخیره اندازه اولیه هر سلول
          if (!cell.dataset.initialWidth || !cell.dataset.initialHeight) {
            const computed = window.getComputedStyle(cell);
            cell.dataset.initialWidth = computed.width;
            cell.dataset.initialHeight = computed.height;
          }

          handle.addEventListener('mousedown', e => {
            e.preventDefault();
            isResizing = true;
            lastX = e.clientX;
            lastY = e.clientY;
            startW = cell.offsetWidth;
            startH = cell.offsetHeight;
            document.body.style.userSelect = 'none';
          });

          const mouseMove = e => {
            if (!isResizing) return;
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            cell.style.width = Math.max(200, startW + dx) + 'px';
            cell.style.height = Math.max(150, startH + dy) + 'px';
            cell.style.minWidth = '100px';
            cell.style.minHeight = '100px';
            cell.style.maxWidth = '100vw';
            cell.style.maxHeight = '100vh';
            cell.style.flex = 'none';
            cell.style.position = 'relative';
            cell.style.zIndex = 100;
          };

          const mouseUp = () => {
            if (isResizing) {
              isResizing = false;
              document.body.style.userSelect = '';
            }
          };

          window.addEventListener('mousemove', mouseMove);
          window.addEventListener('mouseup', mouseUp);

          // Double click to reset size
          handle.addEventListener('dblclick', e => {
            e.preventDefault();
            cell.style.width = cell.dataset.initialWidth;
            cell.style.height = cell.dataset.initialHeight;
            cell.style.minWidth = '';
            cell.style.minHeight = '';
            cell.style.maxWidth = '';
            cell.style.maxHeight = '';
            cell.style.flex = '';
            cell.style.position = '';
            cell.style.zIndex = '';
          });
        });
      </script>
      <style>
        html, body {
          height: 100%;
          margin: 0;
          padding: 0;
          background: #181c25;
        }
        .tradingview-widget-container {
          width: 100vw;
          min-height: 100vh;
          margin: 0;
          padding: 0;
          background: #181c25;
          display: flex;
          align-items: center;
          justify-content: center;
          height: 100vh;
        }
        .charts-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          grid-template-rows: repeat(2, 1fr);
          width: 100vw;
          height: 90vh;
          max-width: 100vw;
          max-height: 100vh;
          margin: 0;
          padding: 0;
          background: #181c25;
          align-items: stretch;
          justify-items: stretch;
        }
        .chart-cell {
          background: #181c25;
          overflow: hidden;
          min-width: 100px;
          min-height: 100px;
          width: 100%;
          height: 100%;
          display: flex;
          position: relative;
          resize: both;
          transition: width 0.2s, height 0.2s;
        }
        .chart-cell::-webkit-resizer {
          background: #444;
        }
        .resize-handle {
          position: absolute;
          width: 18px;
          height: 18px;
          right: 2px;
          bottom: 2px;
          background: rgba(255,255,255,0.15);
          border-radius: 3px;
          cursor: se-resize;
          z-index: 10;
          display: flex;
          align-items: flex-end;
          justify-content: flex-end;
          font-size: 16px;
          color: #aaa;
          user-select: none;
        }
        @media (max-width: 1100px) {
          .charts-grid {
            grid-template-columns: 1fr;
            grid-template-rows: repeat(4, 1fr);
            width: 100vw;
            height: 100vh;
          }
        }
        .chart-cell > div,
        .chart-cell iframe,
        .chart-cell .tradingview-widget-container__widget {
          width: 100% !important;
          height: 100% !important;
          min-width: 0 !important;
          min-height: 0 !important;
          max-width: 100% !important;
          max-height: 100% !important;
          display: block;
        }
      </style>
    </div>
    '''
    components.html(tradingview_html, height=800, scrolling=True)

# --- Trading Sessions (Tehran Time) ---
TRADING_SESSIONS = [
    {"name": "Sydney", "start": "02:30", "end": "11:30"},
    {"name": "Tokyo",  "start": "04:30", "end": "13:30"},
    {"name": "London", "start": "11:30", "end": "20:30"},
    {"name": "New York", "start": "16:00", "end": "00:30"},
]

def get_current_tehran_time():
    tz = pytz.timezone("Asia/Tehran")
    return datetime.now(tz)

def get_current_session(now=None):
    if now is None:
        now = get_current_tehran_time()
    now_time = now.time()
    for session in TRADING_SESSIONS:
        start = datetime.strptime(session["start"], "%H:%M").time()
        end = datetime.strptime(session["end"], "%H:%M").time()
        # Handle sessions that pass midnight
        if start < end:
            if start <= now_time < end:
                return session["name"]
        else:
            if now_time >= start or now_time < end:
                return session["name"]
    return None

# رابط کاربری Streamlit
def main():
    st.set_page_config(layout="wide")
    inject_pro_style()
    
    # Initialize session state
    if "kalman_selected" not in st.session_state:
        st.session_state.kalman_selected = True
    if "wavelet_selected" not in st.session_state:
        st.session_state.wavelet_selected = False
    if "residual_selected" not in st.session_state:
        st.session_state.residual_selected = True
    if "tradingview_selected" not in st.session_state:
        st.session_state.tradingview_selected = False
    if "selected_interval" not in st.session_state:
        st.session_state.selected_interval = "30m"
    if "auto_initial_state" not in st.session_state:
        st.session_state.auto_initial_state = True
    if "initial_value" not in st.session_state:
        st.session_state.initial_value = 0.0
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
        """, unsafe_allow_html=True)
        
        # 1. RUN BUTTON (at the top)
        st.markdown("""
        <div class="sidebar-section">
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                RUN ANALYSIS
            </div>
        """, unsafe_allow_html=True)
        
        run_button = st.button("RUN", type="primary", use_container_width=True)
        
        # --- Digital Clock & Trading Sessions ---
        with st.expander("🕒 Trading Sessions (Tehran Time)", expanded=False):
            def digital_clock():
                tehran = pytz.timezone("Asia/Tehran")
                now = datetime.now(tehran)
                st.markdown(f"""
                <div style="text-align:center; margin-bottom:0.5em;">
                    <span style="font-size:2.5em;font-weight:bold;color:#6366f1;letter-spacing:2px;">
                        {now.strftime('%H:%M:%S')}
                    </span>
                    <div style="font-size:1.1em;color:#10b981;margin-top:0.2em;">
                        Tehran Time
                    </div>
                </div>
                """, unsafe_allow_html=True)

            def show_sessions_analog_clock():
                import plotly.graph_objects as go
                import numpy as np
                sessions = [
                    {"name": "Sydney", "start": "02:30", "end": "11:30", "color": "#f59e42"},
                    {"name": "Tokyo",  "start": "04:30", "end": "13:30", "color": "#10b981"},
                    {"name": "London", "start": "11:30", "end": "20:30", "color": "#6366f1"},
                    {"name": "New York", "start": "16:00", "end": "00:30", "color": "#ef4444"},
                ]
                def to_minutes(t):
                    h, m = map(int, t.split(":"))
                    return h*60 + m
                tehran = pytz.timezone("Asia/Tehran")
                now = datetime.now(tehran)
                now_minutes = now.hour * 60 + now.minute
                fig = go.Figure()
                for s in sessions:
                    start = to_minutes(s["start"])
                    end = to_minutes(s["end"])
                    if end < start:
                        end += 24*60
                    theta = np.linspace(start/4, end/4, 100)
                    r = np.ones_like(theta)
                    fig.add_trace(go.Scatterpolar(
                        r=r,
                        theta=theta,
                        mode='lines',
                        line=dict(color=s["color"], width=14),
                        name=s["name"],
                        hoverinfo='text',
                        text=f"{s['name']}: {s['start']} - {s['end']}"
                    ))
                fig.add_trace(go.Scatterpolar(
                    r=[0, 1.1],
                    theta=[0, now_minutes/4],
                    mode='lines+markers',
                    line=dict(color='#10b981', width=4),
                    marker=dict(size=12, color='#10b981'),
                    name='Now',
                    hoverinfo='skip'
                ))
                fig.update_layout(
                    showlegend=True,
                    polar=dict(
                        radialaxis=dict(visible=False, range=[0, 1.2]),
                        angularaxis=dict(
                            tickmode='array',
                            tickvals=[i*60/4 for i in range(0, 24, 3)],
                            ticktext=[f'{str(i).zfill(2)}:00' for i in range(0, 24, 3)],
                            rotation=90,
                            direction='clockwise'
                        ),
                    ),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=350,
                    width=350,
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False})
                # لیست متنی سشن‌ها و نقاط اشتراک حذف شد
            digital_clock()
            show_sessions_analog_clock()
        
        # 2. SYMBOL (moved here from below)
        st.markdown("""
        <div class="sidebar-section">
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                    <path d="M2 17l10 5 10-5"></path>
                    <path d="M2 12l10 5 10-5"></path>
                </svg>
                SYMBOL
            </div>
        """, unsafe_allow_html=True)
        
        symbols = [
            'GC=F', 'SI=F', '============','PEPE24478-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD'  # Added PEPE cryptocurrency
        ]
        symbol = st.selectbox('Select Symbol', symbols, index=0, label_visibility="collapsed")
        
        # 3. TIME RANGE
        st.markdown("""
        <div class="sidebar-section">
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="16" y1="2" x2="16" y2="6"></line>
                    <line x1="8" y1="2" x2="8" y2="6"></line>
                    <line x1="3" y1="10" x2="21" y2="10"></line>
                </svg>
                TIME RANGE
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', value=datetime.today())
            start_hour = st.selectbox('Start Hour', [str(i).zfill(2) for i in range(24)], index=0, label_visibility="collapsed")
            start_minute = st.selectbox('Start Minute', ['00', '15', '30', '45'], index=0, label_visibility="collapsed")
        with col2:
            end_date = st.date_input('End Date', value=datetime.today())
            end_hour = st.selectbox('End Hour', [str(i).zfill(2) for i in range(24)], index=datetime.now().hour % 24, label_visibility="collapsed")
            end_minute = st.selectbox('End Minute', ['00', '15', '30', '45'], index=1, label_visibility="collapsed")
        
        # 4. INTERVAL
        st.markdown("""
        <div class="sidebar-section">
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                INTERVAL
            </div>
            <div class="interval-buttons">
        """, unsafe_allow_html=True)
        
        # ایجاد دکمه‌های بازه زمانی
        intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '4h']
        
        btn1m = st.button("1m", key="btn_1m", use_container_width=True)
        btn3m = st.button("3m", key="btn_3m", use_container_width=True)
        btn5m = st.button("5m", key="btn_5m", use_container_width=True)
        btn15m = st.button("15m", key="btn_15m", use_container_width=True)
        btn30m = st.button("30m", key="btn_30m", use_container_width=True)
        btn1h = st.button("1h", key="btn_1h", use_container_width=True)
        btn4h = st.button("4h", key="btn_4h", use_container_width=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # بروزرسانی انتخاب اینتروال
        if btn1m:
            st.session_state.selected_interval = "1m"
        if btn3m:
            st.session_state.selected_interval = "3m"
        if btn5m:
            st.session_state.selected_interval = "5m"
        if btn15m:
            st.session_state.selected_interval = "15m"
        if btn30m:
            st.session_state.selected_interval = "30m"
        if btn1h:
            st.session_state.selected_interval = "1h"
        if btn4h:
            st.session_state.selected_interval = "4h"
        
        # نمایش وضعیت فعلی دکمه‌ها
        interval_classes = {
            "1m": "active" if st.session_state.selected_interval == "1m" else "inactive",
            "3m": "active" if st.session_state.selected_interval == "3m" else "inactive",
            "5m": "active" if st.session_state.selected_interval == "5m" else "inactive",
            "15m": "active" if st.session_state.selected_interval == "15m" else "inactive",
            "30m": "active" if st.session_state.selected_interval == "30m" else "inactive",
            "1h": "active" if st.session_state.selected_interval == "1h" else "inactive",
            "4h": "active" if st.session_state.selected_interval == "4h" else "inactive"
        }
        
        st.markdown(f"""
        <script>
            document.querySelector('[data-testid="baseButton-btn_1m"]').className = 'interval-btn {interval_classes["1m"]}';
            document.querySelector('[data-testid="baseButton-btn_3m"]').className = 'interval-btn {interval_classes["3m"]}';
            document.querySelector('[data-testid="baseButton-btn_5m"]').className = 'interval-btn {interval_classes["5m"]}';
            document.querySelector('[data-testid="baseButton-btn_15m"]').className = 'interval-btn {interval_classes["15m"]}';
            document.querySelector('[data-testid="baseButton-btn_30m"]').className = 'interval-btn {interval_classes["30m"]}';
            document.querySelector('[data-testid="baseButton-btn_1h"]').className = 'interval-btn {interval_classes["1h"]}';
            document.querySelector('[data-testid="baseButton-btn_4h"]').className = 'interval-btn {interval_classes["4h"]}';
        </script>
        """, unsafe_allow_html=True)
        
        # 5. ANALYSIS METHOD
        st.markdown("""
        <div class="sidebar-section">
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                    <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
                ANALYSIS METHOD
            </div>
            <div class="section-buttons">
        """, unsafe_allow_html=True)
        
        # ایجاد دکمه‌های تحلیل
        kalman_btn = st.button("Kalman", key="kalman_btn", use_container_width=True)
        wavelet_btn = st.button("Wavelet", key="wavelet_btn", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # مدیریت وضعیت دکمه‌ها
        if kalman_btn:
            st.session_state.kalman_selected = not st.session_state.kalman_selected
        if wavelet_btn:
            st.session_state.wavelet_selected = not st.session_state.wavelet_selected
        
        # نمایش وضعیت فعلی دکمه‌ها
        kalman_class = "active" if st.session_state.kalman_selected else "inactive"
        wavelet_class = "active" if st.session_state.wavelet_selected else "inactive"
        
        st.markdown(f"""
        <script>
            document.querySelector('[data-testid="baseButton-kalman_btn"]').className = 'method-btn {kalman_class}';
            document.querySelector('[data-testid="baseButton-wavelet_btn"]').className = 'method-btn {wavelet_class}';
        </script>
        """, unsafe_allow_html=True)
        
        # 6. VISUALIZATION
        st.markdown("""
        <div class="sidebar-section">
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
                VISUALIZATION
            </div>
            <div class="section-buttons">
        """, unsafe_allow_html=True)
        
        # ایجاد دکمه‌های بصری‌سازی
        residual_btn = st.button("Residual Chart", key="residual_btn", use_container_width=True)
        tradingview_btn = st.button("TradingView", key="tradingview_btn", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # مدیریت وضعیت دکمه‌ها
        if residual_btn:
            st.session_state.residual_selected = not st.session_state.residual_selected
        if tradingview_btn:
            st.session_state.tradingview_selected = not st.session_state.tradingview_selected
        
        # نمایش وضعیت فعلی دکمه‌ها
        residual_class = "active" if st.session_state.residual_selected else "inactive"
        tradingview_class = "active" if st.session_state.tradingview_selected else "inactive"
        
        st.markdown(f"""
        <script>
            document.querySelector('[data-testid="baseButton-residual_btn"]').className = 'method-btn {residual_class}';
            document.querySelector('[data-testid="baseButton-tradingview_btn"]').className = 'method-btn {tradingview_class}';
        </script>
        """, unsafe_allow_html=True)
        
        # 7. DATA SOURCE
        st.markdown("""
        <div class="sidebar-section">
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                DATA SOURCE
            </div>
        """, unsafe_allow_html=True)
        
        data_source = st.radio("Select Data Source", ["Yahoo Finance", "Upload CSV"], index=0, label_visibility="collapsed")
        
        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        # 8. FILTER SETTINGS
        st.markdown("""
        <div class="sidebar-section">
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                    <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
                FILTER SETTINGS
            </div>
        """, unsafe_allow_html=True)
        
        # Get current auto initial state setting
        auto_initial = st.checkbox('Auto Initial State', value=st.session_state.auto_initial_state)
        st.session_state.auto_initial_state = auto_initial
        
        # Show initial value input
        if auto_initial:
            # Show placeholder for auto-calculated value
            st.info("📊 Initial Value will be calculated from the first data point when you click RUN")
            initial_value = 0.0  # This will be calculated when RUN is pressed
        else:
            # Allow manual input
            initial_value = st.number_input('Initial Value', value=st.session_state.initial_value, format="%.4f")
            st.session_state.initial_value = initial_value
    
    # Main content
    if run_button:
        if data_source == "Upload CSV" and uploaded_file is None:
            st.error("Please upload a CSV file")
            return
            
        # جمع‌آوری روش‌های انتخاب شده
        methods = []
        if st.session_state.kalman_selected:
            methods.append('Kalman')
        if st.session_state.wavelet_selected:
            methods.append('Wavelet')
            
        if not methods:
            st.error("Please select at least one analysis method")
            return
            
        if data_source == "Yahoo Finance":
            run_analysis(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                start_hour,
                start_minute,
                end_date.strftime('%Y-%m-%d'),
                end_hour,
                end_minute,
                st.session_state.selected_interval,
                initial_value,
                auto_initial,
                st.session_state.residual_selected,
                methods,
                uploaded_file=None
            )
        else:
            run_analysis(
                "UPLOADED",
                None, None, None, None, None, None, None,
                initial_value,
                auto_initial,
                st.session_state.residual_selected,
                methods,
                uploaded_file=uploaded_file
            )
        
        # نمایش ویجت TradingView اگر انتخاب شده باشد
        if st.session_state.tradingview_selected:
            with st.expander("📊 TradingView Multi-Chart", expanded=True):
                show_tradingview_widgets()

if __name__ == "__main__":
    main()
