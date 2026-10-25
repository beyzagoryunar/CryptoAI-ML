"""
=============================================================================
CryptoAI - ARIMA Geleneksel Referans (Baseline) Model Egitimi ve Degerlendirmesi
=============================================================================
BM401 Bilgisayar Muhendisligi Proje Tasarimi
Prof. Dr. Abdullah Talha KABAKUS
Hazirlayanlar: Beyza Ummuhan Goryunar & Ahmet Faruk Durmus

Model Tanimi:
- ARIMA(1, 1, 1) (AutoRegressive Integrated Moving Average)
- p = 1: 1 saat onceki gecmis fiyat iliskisi (AutoRegressive)
- d = 1: 1. dereceden fark alma ile seriyi duraganlastirma (Integrated)
- q = 1: 1 saat onceki tahmin hatasini duzeltme (Moving Average)

Egitim / Test Protokolu:
- Egitim Kumesi : Ilk 17,118 saat (%99)
- Test Kumesi   : Son 200 saat (Guncel test haftasi, Hold-out Out-of-Sample)
- Metrikler     : RMSE, MAE, MAPE, R2, Yon Dogrulugu (Directional Accuracy %)
=============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

def run_arima_experiment():
    print("=" * 75)
    print("CryptoAI - ARIMA(1,1,1) Zaman Serisi Model Egitimi Baslatildi")
    print("=" * 75)
    
    # 1. Veri Setini Oku
    csv_path = "crypto_macro_dataset.csv"
    if not os.path.exists(csv_path):
        csv_path = r"C:\Users\beyz\Desktop\bitirme projesi\crypto_macro_dataset.csv"
        
    print(f"[1/4] Veri seti yukleniyor: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Zaman damgasini datetime yap ve btc_close serisini al
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    elif 'Datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['Datetime'])
    else:
        df['datetime'] = pd.date_range(start='2024-10-26', periods=len(df), freq='h')
        
    df.set_index('datetime', inplace=True)
    prices = df['btc_close'].values
    dates = df.index
    
    print(f"  -> Toplam Veri Sayisi : {len(prices):,} saat")
    
    # 2. Egitim ve Test Kumesi Ayrilmasi (Zamansal Walk-Forward Split)
    # Son 168 saat (Tam 1 Hafta) test kumesi
    test_size = 168
    train_prices = prices[:-test_size]
    test_prices = prices[-test_size:]
    test_dates = dates[-test_size:]
    
    print(f"  -> Egitim Kumesi      : {len(train_prices):,} saat ({dates[0].strftime('%Y-%m-%d')} - {dates[-test_size-1].strftime('%Y-%m-%d')})")
    print(f"  -> Test Kumesi        : {len(test_prices):,} saat (Son 7 Gun: {test_dates[0].strftime('%Y-%m-%d %H:%M')} - {test_dates[-1].strftime('%Y-%m-%d %H:%M')})")
    
    # 3. ARIMA Model Egitimi ve 1-Adim Ileri (1-Step Ahead) Tahmin
    print("\n[2/4] ARIMA(1, 1, 1) modeli egitiliyor ve 1 saatlik ileri tahminler uretiliyor...")
    start_time = time.time()
    
    # Modeli egitim verisiyle fit et
    history = list(train_prices)
    predictions = []
    
    # Hizli ve stabil 1-adim rolling forecast
    model = ARIMA(history, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Test haftasinda adim adim tahmin et
    for t in range(len(test_prices)):
        yhat = model_fit.forecast(steps=1)[0]
        predictions.append(yhat)
        # Gerceklesen gozlemi modele ekle (Rolling update)
        history.append(test_prices[t])
        model_fit = model_fit.append([test_prices[t]], refit=False)
        
    duration = time.time() - start_time
    predictions = np.array(predictions)
    print(f"  -> Tahmin islemi {duration:.2f} saniyede tamamlandi.")
    
    # 4. Hata ve Basari Metrikleri Hesaplama
    print("\n[3/4] Akademik basari metrikleri hesaplaniyor...")
    mae = mean_absolute_error(test_prices, predictions)
    rmse = np.sqrt(mean_squared_error(test_prices, predictions))
    mape = np.mean(np.abs((test_prices - predictions) / test_prices)) * 100
    r2 = r2_score(test_prices, predictions)
    
    # Yon Dogrulugu (Directional Accuracy): Fiyatin artis/azalis yonu bilindi mi?
    actual_direction = np.sign(np.diff(np.append([train_prices[-1]], test_prices)))
    pred_direction = np.sign(predictions - np.append([train_prices[-1]], test_prices[:-1]))
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    print("=" * 75)
    print("ARIMA(1, 1, 1) ILK SONUC RAPORU:")
    print("=" * 75)
    print(f"  * Ortalama Mutlak Hata (MAE)     : ${mae:,.2f}")
    print(f"  * Hata Kareler Koku (RMSE)       : ${rmse:,.2f}")
    print(f"  * Yuzdesel Hata Orani (MAPE)     : %{mape:.2f}")
    print(f"  * Belirleme Katsayisi (R2)       : {r2:.4f}")
    print(f"  * Yon Tahmin Dogrulugu (DA)      : %{direction_accuracy:.2f}")
    print("=" * 75)
    
    # 5. Profesyonel Grafik Olusturma
    print("\n[4/4] Karsilastirma grafigi ciziliyor ve kaydediliyor...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Ust Panel: Gercek vs ARIMA Tahmini
    ax1.plot(test_dates, test_prices, label='Gercek Bitcoin Fiyati ($)', color='#00B2FE', linewidth=2.0)
    ax1.plot(test_dates, predictions, label='ARIMA(1,1,1) Tahmini ($)', color='#FF3B30', linestyle='--', linewidth=1.8, alpha=0.9)
    ax1.set_title(f'Bitcoin Saatlik Fiyat Tahmini: Gercek vs ARIMA(1,1,1)\n(MAE: ${mae:.1f} | MAPE: %{mape:.2f} | Yon Dogrulugu: %{direction_accuracy:.1f})', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel('Fiyat ($ USD)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Alt Panel: Kalinti / Tahmin Hatasi (Residuals)
    residuals = test_prices - predictions
    ax2.plot(test_dates, residuals, color='#26A17B', linewidth=1.2, label='Tahmin Hatasi (Gercek - Tahmin)')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    ax2.set_ylabel('Hata ($)', fontsize=10)
    ax2.set_xlabel('Zaman (Saat)', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # Grafikleri kaydet
    out_graph_local = "arima_tahmin_grafigi.png"
    out_graph_desktop = r"C:\Users\beyz\Desktop\bitirme projesi\arima_tahmin_grafigi.png"
    plt.savefig(out_graph_local, dpi=300)
    try:
        plt.savefig(out_graph_desktop, dpi=300)
    except Exception:
        pass
    plt.close()
    
    print(f"  -> Grafik kaydedildi: {out_graph_local}")
    if os.path.exists(out_graph_desktop):
        print(f"  -> Masaustune de aktarildi: {out_graph_desktop}")
        
    print("\nARIMA Egitimi ve Degerlendirmesi Basariyla Tamamlandi!")

if __name__ == "__main__":
    run_arima_experiment()
