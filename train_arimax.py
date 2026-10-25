"""
=============================================================================
CryptoAI - ARIMAX Cok Degiskenli Geleneksel Model Egitimi ve Karsilastirmasi
=============================================================================
BM401 Bilgisayar Muhendisligi Proje Tasarimi
Prof. Dr. Abdullah Talha KABAKUS
Hazirlayanlar: Beyza Ummuhan Goryunar & Ahmet Faruk Durmus

Model Tanimi:
- ARIMAX(1, 1, 1) (ARIMA with eXogenous variables)
- Hedef Degisken (Endogenous) : BTC Kapanis Fiyati (btc_close)
- Dis Degiskenler (Exogenous):
    1. eth_close (Ethereum Fiyati)
    2. eth_volume (Ethereum Islem Hacmi)
    3. sp500_close (S&P 500 Makro Endeksi)
    4. fear_greed_score (Korku ve Acgozluluk Skoru)
    5. btc_rsi (14 Saatlik RSI)
    6. btc_macd (MACD Gostergesi)

Egitim / Test Protokolu:
- Egitim Kumesi : Ilk 17,150 saat
- Test Kumesi   : Son 168 saat (ARIMA ile birebir ayni son 7 gunluk test haftasi)
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def run_arimax_experiment():
    print("=" * 75)
    print("CryptoAI - ARIMAX(1,1,1) Cok Degiskenli Model Egitimi Baslatildi")
    print("=" * 75)
    
    # 1. Veri Setini Oku
    csv_path = "crypto_macro_dataset.csv"
    if not os.path.exists(csv_path):
        csv_path = r"C:\Users\beyz\Desktop\bitirme projesi\crypto_macro_dataset.csv"
        
    print(f"[1/4] Veri seti yukleniyor: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    elif 'Datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['Datetime'])
    else:
        df['datetime'] = pd.date_range(start='2024-10-26', periods=len(df), freq='h')
        
    df.set_index('datetime', inplace=True)
    
    # Hedef ve Dis Degiskenler
    endog = df['btc_close'].values
    exog_cols = ['eth_close', 'sp500_close', 'fear_greed_score', 'btc_rsi', 'btc_macd']
    exog = df[exog_cols].values
    dates = df.index
    
    print(f"  -> Toplam Veri Sayisi : {len(endog):,} saat")
    print(f"  -> Kullanilan Dis Degiskenler (Exog): {', '.join(exog_cols)}")
    
    # 2. Egitim ve Test Kumesi Ayrilmasi (Son 168 saat = 7 Gun)
    test_size = 168
    train_endog = endog[:-test_size]
    test_endog = endog[-test_size:]
    
    train_exog = exog[:-test_size]
    test_exog = exog[-test_size:]
    test_dates = dates[-test_size:]
    
    print(f"  -> Egitim Kumesi      : {len(train_endog):,} saat")
    print(f"  -> Test Kumesi        : {len(test_endog):,} saat ({test_dates[0].strftime('%Y-%m-%d %H:%M')} - {test_dates[-1].strftime('%Y-%m-%d %H:%M')})")
    
    # Dis degiskenleri olcekle (Numerical stability icin StandardScaler)
    scaler = StandardScaler()
    train_exog_scaled = scaler.fit_transform(train_exog)
    test_exog_scaled = scaler.transform(test_exog)
    
    # 3. ARIMAX Modeli Egitimi ve 1-Adim Ileri Tahmin
    print("\n[2/4] ARIMAX(1, 1, 1) modeli egitiliyor ve 1 saatlik ileri tahminler uretiliyor...")
    start_time = time.time()
    
    # Once egitim verisiyle modeli fit et
    history_endog = list(train_endog)
    history_exog = list(train_exog_scaled)
    predictions = []
    
    # Ilk modeli egit
    model = ARIMA(endog=train_endog, exog=train_exog_scaled, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Test haftasi uzerinde rolling walk-forward tahmin
    for t in range(test_size):
        # 1 adim sonrasi icin tahmin et (o anki saatteki exog verisiyle)
        curr_exog = test_exog_scaled[t:t+1]
        yhat = model_fit.forecast(steps=1, exog=curr_exog)[0]
        predictions.append(yhat)
        
        # Gerceklesen degeri modele ekle
        model_fit = model_fit.append(endog=[test_endog[t]], exog=curr_exog, refit=False)
        
    duration = time.time() - start_time
    predictions = np.array(predictions)
    print(f"  -> Tahmin islemi {duration:.2f} saniyede tamamlandi.")
    
    # 4. Metriklerin Hesaplanmasi
    print("\n[3/4] ARIMAX basari metrikleri hesaplaniyor...")
    mae = mean_absolute_error(test_endog, predictions)
    rmse = np.sqrt(mean_squared_error(test_endog, predictions))
    mape = np.mean(np.abs((test_endog - predictions) / test_endog)) * 100
    r2 = r2_score(test_endog, predictions)
    
    # Yon Dogrulugu
    actual_direction = np.sign(np.diff(np.append([train_endog[-1]], test_endog)))
    pred_direction = np.sign(predictions - np.append([train_endog[-1]], test_endog[:-1]))
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    print("=" * 75)
    print("ARIMAX(1, 1, 1) SONUC RAPORU:")
    print("=" * 75)
    print(f"  * Ortalama Mutlak Hata (MAE)     : ${mae:,.2f}")
    print(f"  * Hata Kareler Koku (RMSE)       : ${rmse:,.2f}")
    print(f"  * Yuzdesel Hata Orani (MAPE)     : %{mape:.2f}")
    print(f"  * Belirleme Katsayisi (R2)       : {r2:.4f}")
    print(f"  * Yon Tahmin Dogrulugu (DA)      : %{direction_accuracy:.2f}")
    print("=" * 75)
    
    # 5. Karsilastirma Grafigi
    print("\n[4/4] Profesyonel ARIMAX grafigi ciziliyor...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Ust Panel
    ax1.plot(test_dates, test_endog, label='Gercek Bitcoin Fiyati ($)', color='#00B2FE', linewidth=2.0)
    ax1.plot(test_dates, predictions, label='ARIMAX(1,1,1) [BTC + Makro + Duygu]', color='#FF9500', linestyle='--', linewidth=1.8, alpha=0.95)
    ax1.set_title(f'Bitcoin Saatlik Fiyat Tahmini: Gercek vs ARIMAX(1,1,1)\n(MAE: ${mae:.1f} | MAPE: %{mape:.2f} | Yon Dogrulugu: %{direction_accuracy:.1f})', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel('Fiyat ($ USD)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Alt Panel
    residuals = test_endog - predictions
    ax2.plot(test_dates, residuals, color='#AF52DE', linewidth=1.2, label='ARIMAX Tahmin Hatasi ($)')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    ax2.set_ylabel('Hata ($)', fontsize=10)
    ax2.set_xlabel('Zaman (Saat)', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    out_graph_local = "arimax_tahmin_grafigi.png"
    out_graph_desktop = r"C:\Users\beyz\Desktop\bitirme projesi\arimax_tahmin_grafigi.png"
    plt.savefig(out_graph_local, dpi=300)
    try:
        plt.savefig(out_graph_desktop, dpi=300)
    except Exception:
        pass
    plt.close()
    
    print(f"  -> Grafik kaydedildi: {out_graph_local}")
    print("\nARIMAX Egitimi ve Degerlendirmesi Basariyla Tamamlandi!")

if __name__ == "__main__":
    run_arimax_experiment()
