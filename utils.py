import sqlite3
import os
import pandas as pd
import streamlit as st
from datetime import datetime

# ===========================
# SQLite Database Functions
# ===========================

def get_db_connection():
    """Возвращает подключение к базе, безопасно для Streamlit Cloud"""
    db_path = os.path.join("/tmp", "signals.db")  # безопасная временная директория
    conn = sqlite3.connect(db_path)
    return conn

def save_to_db(df):
    """Сохраняет DataFrame в SQLite"""
    if df.empty:
        return
    conn = get_db_connection()
    df_to_save = df.copy()
    df_to_save['saved_at'] = datetime.now()
    df_to_save.to_sql("signals", conn, if_exists='append', index=False)
    conn.close()

def load_from_db(limit=50):
    """Загружает последние записи из базы"""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(f"SELECT * FROM signals ORDER BY saved_at DESC LIMIT {limit}", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

# ===========================
# Helpers
# ===========================

def format_display_time(timestamp):
    """Форматирование времени для отображения"""
    return timestamp.strftime('%H:%M - %d %b') if timestamp else ""
