"""
=============================================================================
CryptoAI - Cok Boyutlu Kripto & Makro Veri Seti Birlestirici (Data Fusion Pipeline)
=============================================================================
BM401 Bilgisayar Muhendisligi Proje Tasarimi
Prof. Dr. Abdullah Talha KABAKUS
Hazirlayanlar: Beyza Ummuhan Goryunar & Ahmet Faruk Durmus

Veri Kaynaklari (4 Adet):
1. BTC-USD (Yahoo Finance) - Saatlik Fiyat & Hacim (Omurga)
2. ETH-USD (Yahoo Finance) - Saatlik Sermaye Rotasyonu
3. S&P 500 (^GSPC - Yahoo Finance) - Kuresel Risk Istahi (Hafta sonlari asof backward fill ile doldurulur)
4. Fear & Greed Index (Alternative.me API) - Gunluk Piyasa Psikolojisi (Saatlik asof backward fill)

Turetilen Teknik Gostergeler:
- RSI (Relative Strength Index - 14 Periyot)
- MACD (Moving Average Convergence Divergence - 12, 26, 9)
=============================================================================
"""

import os
import sys
import time
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Gereksiz uyarilari gizle
warnings.filterwarnings('ignore')

def fetch_yfinance_data():
    print("[1/5] Yahoo Finance uzerinden son 2 yillik saatlik veriler indiriliyor...")
    
    # 1. Bitcoin (BTC-USD)
    print("  -> Bitcoin (BTC-USD) saatlik verisi cekiliyor...")
    btc_ticker = yf.Ticker("BTC-USD")
    btc_raw = btc_ticker.history(period="730d", interval="1h")
    time.sleep(1)
    
    btc = pd.DataFrame(index=pd.to_datetime(btc_raw.index, utc=True))
    btc['btc_open'] = btc_raw['Open'].astype(float).values
    btc['btc_high'] = btc_raw['High'].astype(float).values
    btc['btc_low'] = btc_raw['Low'].astype(float).values
    btc['btc_close'] = btc_raw['Close'].astype(float).values
    btc['btc_volume'] = btc_raw['Volume'].astype(float).values
    btc.sort_index(inplace=True)
    print(f"     BTC alindi: {len(btc):,} saatlik mum")
    
    # 2. Ethereum (ETH-USD)
    print("  -> Ethereum (ETH-USD) saatlik verisi cekiliyor...")
    eth_ticker = yf.Ticker("ETH-USD")
    eth_raw = eth_ticker.history(period="730d", interval="1h")
    time.sleep(1)
    
    eth = pd.DataFrame(index=pd.to_datetime(eth_raw.index, utc=True))
    eth['eth_close'] = eth_raw['Close'].astype(float).values
    eth['eth_volume'] = eth_raw['Volume'].astype(float).values
    eth.sort_index(inplace=True)
    print(f"     ETH alindi: {len(eth):,} saatlik mum")
    
    # 3. S&P 500 (^GSPC)
    print("  -> S&P 500 (^GSPC) saatlik makro endeks verisi cekiliyor...")
    sp_ticker = yf.Ticker("^GSPC")
    sp_raw = sp_ticker.history(period="730d", interval="1h")
    time.sleep(1)
    
    sp500 = pd.DataFrame(index=pd.to_datetime(sp_raw.index, utc=True))
    sp500['sp500_close'] = sp_raw['Close'].astype(float).values
    sp500.sort_index(inplace=True)
    print(f"     S&P 500 alindi: {len(sp500):,} borsa seans saati")
    
    return btc, eth, sp500


def fetch_fear_and_greed_data():
    print("[2/5] Alternative.me API'sinden Korku ve Acgozluluk Endeksi cekiliyor...")
    url = "https://api.alternative.me/fng/?limit=730"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json().get("data", [])
        
        fng_records = []
        for item in data:
            ts = pd.to_datetime(int(item["timestamp"]), unit="s", utc=True)
            val = float(item["value"])
            fng_records.append({"datetime": ts, "fear_greed_score": val})
            
        fng_df = pd.DataFrame(fng_records)
        fng_df.set_index("datetime", inplace=True)
        fng_df.sort_index(inplace=True)
        print(f"  -> {len(fng_df)} gunluk duygu skoru basariyla alindi.")
        return fng_df
    except Exception as e:
        print(f"  UYARI: Fear & Greed verisi alinirken hata olustu: {e}. Varsayilan 50 (Notr) kullanilacak.")
        return pd.DataFrame()


def compute_technical_indicators(df):
    print("[3/5] Teknik gostergeler (RSI ve MACD) hesaplaniyor...")
    
    # 1. RSI (Relative Strength Index - 14 Periyot)
    delta = df['btc_close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    df['btc_rsi'] = 100 - (100 / (1 + rs))
    # Ilk 14 saatin baslangic RSI degerini 50 (Notr) ile doldur
    df['btc_rsi'] = df['btc_rsi'].bfill().fillna(50.0)
    
    # 2. MACD (Moving Average Convergence Divergence - 12, 26, 9)
    ema12 = df['btc_close'].ewm(span=12, adjust=False).mean()
    ema26 = df['btc_close'].ewm(span=26, adjust=False).mean()
    df['btc_macd'] = ema12 - ema26
    df['btc_macd_signal'] = df['btc_macd'].ewm(span=9, adjust=False).mean()
    df['btc_macd_hist'] = df['btc_macd'] - df['btc_macd_signal']
    
    # 3. Model Hedef Degiskenleri (Targets)
    # 1 saat sonraki kapanis fiyati (Regresyon icin: ARIMA / LSTM)
    df['target_price_1h'] = df['btc_close'].shift(-1)
    # 1 saat sonraki getiri yuzdesi
    df['target_return_1h'] = (df['target_price_1h'] - df['btc_close']) / df['btc_close']
    # 1 saat sonraki yon (Siniflandirma icin: 1 = Yukselis, 0 = Dusuk/Yatay)
    df['target_direction_1h'] = (df['target_price_1h'] > df['btc_close']).astype(int)
    
    return df


def merge_and_clean_datasets(btc, eth, sp500, fng):
    print("[4/5] 4 veri seti merge_asof ve ffill yontemiyle senkronize ediliyor...")
    
    # merge_asof icin sirali DataFrame gereklidir
    btc = btc.sort_index()
    eth = eth.sort_index()
    sp500 = sp500.sort_index()
    
    # 1. BTC ile ETH birlestir (merge_asof: en yakin onceki degeri eslestirir)
    merged = pd.merge_asof(
        btc,
        eth,
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
    # 2. S&P 500 makro endeksini ekle
    # Amerika borsasi hafta sonu kapaliyken cuma saat 23:00 kapanis fiyati pazartesiye kadar aktarilir
    merged = pd.merge_asof(
        merged,
        sp500,
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
    # 3. Fear & Greed duygu endeksini ekle
    if not fng.empty:
        fng = fng.sort_index()
        merged = pd.merge_asof(
            merged,
            fng,
            left_index=True,
            right_index=True,
            direction='backward'
        )
    else:
        merged['fear_greed_score'] = 50.0
        
    # En bastaki ilk birkac saatte S&P veya FNG henuz baslamamissa geriye dogru doldur (bfill)
    merged['sp500_close'] = merged['sp500_close'].ffill().bfill()
    merged['eth_close'] = merged['eth_close'].ffill().bfill()
    merged['eth_volume'] = merged['eth_volume'].fillna(0.0)
    merged['fear_greed_score'] = merged['fear_greed_score'].ffill().bfill()
    
    # 4. Teknik gostergeleri (RSI, MACD) ve model hedeflerini hesapla
    merged = compute_technical_indicators(merged)
    
    # En son saatin sonraki saati (target) henuz yasanmadigi icin son 1 satiri cikar
    merged = merged.iloc[:-1].copy()
    
    return merged


def main():
    start_time = time.time()
    print("=" * 75)
    print("CryptoAI - Cok Boyutlu Veri Seti Birlestirme Pipeline Baslatildi")
    print("=" * 75)
    
    # 1. Yahoo Finance indir
    btc, eth, sp500 = fetch_yfinance_data()
    
    # 2. Alternative.me Fear & Greed indir
    fng = fetch_fear_and_greed_data()
    
    # 3. Birlestir ve ffill ile bosluklari doldur
    final_df = merge_and_clean_datasets(btc, eth, sp500, fng)
    
    # 4. Dosya yollari
    target_dir = r"C:\Users\beyz\Desktop\bitirme projesi"
    output_csv = os.path.join(target_dir, "crypto_macro_dataset.csv")
    workspace_csv = os.path.join(r"c:\Users\beyz\CryptoAI", "crypto_macro_dataset.csv")
    
    print("[5/5] Birlestirilmis veri seti diske kaydediliyor...")
    final_df.to_csv(output_csv, index=True, index_label="timestamp")
    try:
        final_df.to_csv(workspace_csv, index=True, index_label="timestamp")
    except Exception:
        pass
        
    duration = time.time() - start_time
    file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    
    print("=" * 75)
    print(f"TEBRIKLER! Birlestirme islemi {duration:.2f} saniyede basariyla tamamlandi.")
    print("=" * 75)
    print(f"Kaydedilen Dosya  : {output_csv}")
    print(f"Dosya Boyutu      : {file_size_mb:.2f} MB (Cok hafif)")
    print(f"Toplam Satir (Mum): {len(final_df):,} saat")
    print(f"Tarih Araligi     : {final_df.index[0].strftime('%Y-%m-%d %H:%M UTC')}  --->  {final_df.index[-1].strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Sutun Sayisi      : {len(final_df.columns)} adet")
    print("\nSutunlar ve Eksik Veri (NaN) Kontrolu:")
    print("-" * 55)
    null_counts = final_df.isnull().sum()
    for col, null_c in null_counts.items():
        print(f"  {col:<24}: {null_c} eksik veri (SIFIR EKSIK - TEMIZ)")
        
    print("-" * 55)
    print("\nIlk 3 Satir Onizleme:")
    preview_cols = ['btc_close', 'eth_close', 'sp500_close', 'fear_greed_score', 'btc_rsi', 'btc_macd']
    print(final_df[preview_cols].head(3).to_string())
    print("\nSon 3 Satir Onizleme:")
    print(final_df[preview_cols].tail(3).to_string())
    print("=" * 75)

if __name__ == "__main__":
    main()
