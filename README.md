# Intelligent 5G Handover Optimization using Reinforcement Learning

Bu proje, 5G ağlarında akıllı baz istasyonu geçişi (handover) optimizasyonu için Reinforcement Learning kullanır.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Modelleri Eğit
```bash
python train.py
```
Bu komut hem DQN hem de PPO ajanlarını eğitir ve `models/` klasörüne kaydeder.

### 2. İnteraktif Dashboard (Hocaya Göstermek İçin!)
```bash
streamlit run dashboard.py
```
Bu komut web tabanlı interaktif dashboard'u başlatır. Tarayıcıda açılır ve şunları gösterir:
- Gerçek zamanlı kullanıcı hareketleri ve baz istasyonları
- Canlı performans grafikleri
- Algoritma karşılaştırması (Baseline vs DQN vs PPO)

### 2. İnteraktif Web Dashboard (Hocaya Göstermek İçin!)
```bash
python web_app.py
```
Tarayıcıda `http://localhost:5000` adresini aç. Dashboard şunları gösterir:
- Gerçek zamanlı kullanıcı hareketleri ve baz istasyonları
- Canlı performans metrikleri
- Algoritma seçimi (Baseline/DQN/PPO)
- Karşılaştırma sonuçları

### 3. Komut Satırı Değerlendirme
```bash
python evaluate.py
```
Bu komut baseline, DQN ve PPO yöntemlerini karşılaştırır ve sonuçları görselleştirir.

## Proje Yapısı

- `environment.py`: 5G simülasyon ortamı (Gymnasium)
- `base_station.py`: Baz istasyonu sınıfı (SINR hesaplama, yük yönetimi)
- `users.py`: Kullanıcı tipleri (Yaya, Araç, Acil Durum Aracı)
- `train.py`: RL ajanlarını eğitme
- `evaluate.py`: Performans değerlendirme ve görselleştirme

## Özellikler

- 500x500m simülasyon alanı
- 3 baz istasyonu
- 3 kullanıcı tipi (farklı hızlarda)
- DQN ve PPO algoritmaları
- Ödül fonksiyonu: SINR, ping-pong cezası, enerji tüketimi, acil durum önceliği
- Performans metrikleri: Handover sayısı, ortalama SINR, enerji verimliliği

## Sonuçlar

Eğitim sonrası `comparison_results.png` dosyasında görselleştirilmiş karşılaştırma grafikleri oluşturulur.
