"""
=============================================================================
CryptoAI - Cok Boyutlu LSTM Derin Ogrenme Model Egitimi ve Degerlendirmesi
=============================================================================
BM401 Bilgisayar Muhendisligi Proje Tasarimi
Prof. Dr. Abdullah Talha KABAKUS
Hazirlayanlar: Beyza Ummuhan Goryunar & Ahmet Faruk Durmus

Model Tanimi:
- Cok Boyutlu (Multivariate) LSTM (Long Short-Term Memory)
- Girdi Oznitelikleri (7 Adet):
    1. btc_close (Bitcoin Fiyati)
    2. btc_volume (Bitcoin Islem Hacmi)
    3. eth_close (Ethereum Fiyati)
    4. sp500_close (S&P 500 Makro Endeksi)
    5. fear_greed_score (Korku ve Acgozluluk Skoru)
    6. btc_rsi (14 Saatlik RSI)
    7. btc_macd (MACD Gostergesi)
- Geriye Bakis Penceresi (Lookback Window): 24 Saat (1 Gunluk gecmis sekans)
- Hedef: 1 Saat Sonraki Bitcoin Kapanis Fiyati

Egitim / Test Protokolu:
- Egitim Kumesi : Ilk 17,150 saat
- Test Kumesi   : Son 168 saat (ARIMA ve ARIMAX ile birebir ayni son 7 gunluk test haftasi)
=============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. LSTM Model Mimarisi
# ==========================================
class CryptoLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, dropout=0.2):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Son zaman adimindaki gizli durumu cikis katmanina ver
        out = self.fc(lstm_out[:, -1, :])
        return out


def create_sequences(features, targets, seq_length=24):
    """Zaman serisini kayan pencere (sliding window) formatina cevirir."""
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)


def run_lstm_experiment():
    print("=" * 75)
    print("CryptoAI - Cok Boyutlu LSTM Derin Ogrenme Egitimi Baslatildi")
    print("=" * 75)
    
    # 1. Veri Setini Oku
    csv_path = "crypto_macro_dataset.csv"
    if not os.path.exists(csv_path):
        csv_path = r"C:\Users\beyz\Desktop\bitirme projesi\crypto_macro_dataset.csv"
        
    print(f"[1/5] Veri seti yukleniyor: {csv_path}")
    df = pd.read_csv(csv_path)
    
    feature_cols = [
        'btc_close', 'btc_volume', 'eth_close', 'sp500_close', 
        'fear_greed_score', 'btc_rsi', 'btc_macd'
    ]
    
    dates = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else pd.date_range('2024-10-26', periods=len(df), freq='h')
    df['datetime'] = dates
    df.set_index('datetime', inplace=True)
    
    print(f"  -> Toplam Veri Sayisi : {len(df):,} saat")
    print(f"  -> Girdi Boyutlari    : {len(feature_cols)} Oznitelik ({', '.join(feature_cols)})")
    
    # 2. Egitim ve Test Ayrilmasi (Son 168 Saat Test Haftasi)
    test_size = 168
    seq_length = 24  # Son 24 saate bakarak sonraki saati tahmin et
    
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size - seq_length:]  # Test sekansi icin basta 24 saatlik tampon
    test_dates = df.index[-test_size:]
    
    print(f"  -> Egitim Kumesi      : {len(train_df):,} saat")
    print(f"  -> Test Kumesi        : {test_size:,} saat (Son 7 Gun)")
    print(f"  -> Sekans Penceresi   : {seq_length} saatlik geriye bakis (Lookback)")
    
    # 3. MinMax Olcekleme (0 ile 1 arasina normalize et)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    train_X_scaled = scaler_X.fit_transform(train_df[feature_cols].values)
    train_y_scaled = scaler_y.fit_transform(train_df[['btc_close']].values)
    
    test_X_scaled = scaler_X.transform(test_df[feature_cols].values)
    test_y_scaled = scaler_y.transform(test_df[['btc_close']].values)
    
    # Sekanslari olustur
    X_train, y_train = create_sequences(train_X_scaled, train_y_scaled, seq_length)
    X_test, y_test = create_sequences(test_X_scaled, test_y_scaled, seq_length)
    
    # PyTorch Tensor'larina cevir
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 4. Model Kurulumu ve Egitim Dongusu
    print("\n[2/5] LSTM Modeli PyTorch uzerinde 20 Epok boyunca egitiliyor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  -> Kullanilan Donanim : {device.type.upper()}")
    
    model = CryptoLSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    
    start_time = time.time()
    epochs = 20
    model.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  -> Epok [{epoch:02d}/{epochs:02d}] - Loss (MSE): {avg_loss:.6f}")
            
    train_duration = time.time() - start_time
    print(f"  -> Egitim {train_duration:.2f} saniyede tamamlandi.")
    
    # 5. Test Haftasi Uzerinde Tahmin Uretimi
    print("\n[3/5] Test haftasi (Son 168 saat) icin tahminler cikariliyor...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        test_preds_scaled = model(X_test_tensor).cpu().numpy()
        
    # Tahminleri ve gercek degerleri orijinal Dolar seviyesine donustur
    predictions = scaler_y.inverse_transform(test_preds_scaled).flatten()
    actuals = scaler_y.inverse_transform(y_test).flatten()
    
    # 6. Basari Metrikleri
    print("\n[4/5] LSTM basari metrikleri hesaplaniyor...")
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)
    
    # Yon Dogrulugu
    actual_direction = np.sign(np.diff(np.append([train_df['btc_close'].iloc[-1]], actuals)))
    pred_direction = np.sign(predictions - np.append([train_df['btc_close'].iloc[-1]], actuals[:-1]))
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    print("=" * 75)
    print("LSTM DERIN OGRENME SONUC RAPORU:")
    print("=" * 75)
    print(f"  * Ortalama Mutlak Hata (MAE)     : ${mae:,.2f}")
    print(f"  * Hata Kareler Koku (RMSE)       : ${rmse:,.2f}")
    print(f"  * Yuzdesel Hata Orani (MAPE)     : %{mape:.2f}")
    print(f"  * Belirleme Katsayisi (R2)       : {r2:.4f}")
    print(f"  * Yon Tahmin Dogrulugu (DA)      : %{direction_accuracy:.2f}")
    print("=" * 75)
    
    # 7. Grafikleri Olustur ve Kaydet
    print("\n[5/5] LSTM tahmin grafigi ciziliyor...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Ust Panel
    ax1.plot(test_dates, actuals, label='Gercek Bitcoin Fiyati ($)', color='#00B2FE', linewidth=2.0)
    ax1.plot(test_dates, predictions, label='LSTM Derin Ogrenme Tahmini ($)', color='#FF2D55', linestyle='--', linewidth=1.8, alpha=0.95)
    ax1.set_title(f'Bitcoin Saatlik Fiyat Tahmini: Gercek vs Cok Boyutlu LSTM\n(MAE: ${mae:.1f} | MAPE: %{mape:.2f} | Yon Dogrulugu: %{direction_accuracy:.1f})', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel('Fiyat ($ USD)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Alt Panel
    residuals = actuals - predictions
    ax2.plot(test_dates, residuals, color='#5856D6', linewidth=1.2, label='LSTM Tahmin Hatasi ($)')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    ax2.set_ylabel('Hata ($)', fontsize=10)
    ax2.set_xlabel('Zaman (Saat)', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    out_graph_local = "lstm_tahmin_grafigi.png"
    out_graph_desktop = r"C:\Users\beyz\Desktop\bitirme projesi\lstm_tahmin_grafigi.png"
    plt.savefig(out_graph_local, dpi=300)
    try:
        plt.savefig(out_graph_desktop, dpi=300)
        plt.savefig(r"C:\Users\beyz\.gemini\antigravity\brain\1eeae60b-d9e7-4fd8-b45e-a9d60ad1bd17\lstm_tahmin_grafigi.png", dpi=300)
    except Exception:
        pass
    plt.close()
    
    print(f"  -> Grafik kaydedildi: {out_graph_local}")
    print("\nLSTM Egitimi ve Degerlendirmesi Basariyla Tamamlandi!")
    
    # Tahminleri diske kaydet (Ortak 3'lu karsilastirma icin)
    np.save('lstm_preds.npy', predictions)
    np.save('actuals.npy', actuals)

if __name__ == "__main__":
    run_lstm_experiment()
