import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import os

# ============================================================
# ДАННЫЕ GWTC-1 и GWTC-2 (встроенные)
# ============================================================

# GWTC-1: 11 событий
gwtc1_masses = [
    (35.6, 3.5, 30.6, 3.0),   # GW150914
    (23.3, 6.2, 13.6, 3.9),   # GW151012
    (13.7, 4.5, 7.6, 2.5),    # GW151226
    (31.0, 4.6, 19.4, 3.4),   # GW170104
    (10.9, 2.6, 7.6, 1.6),    # GW170608
    (50.6, 8.9, 33.6, 6.4),   # GW170729
    (34.7, 4.8, 23.8, 3.5),   # GW170809
    (30.7, 3.8, 25.3, 3.1),   # GW170814
    (1.46, 0.1, 1.27, 0.08),  # GW170817
    (35.5, 4.6, 26.7, 3.6),   # GW170818
    (39.6, 5.8, 29.4, 4.6),   # GW170823
]

# GWTC-2: ключевые события
gwtc2_masses = [
    (24.5, 4.0, 13.2, 3.0),   # GW190408_181802
    (29.7, 3.0, 8.2, 1.0),    # GW190412
    (33.2, 5.0, 23.3, 4.0),   # GW190413_052954
    (38.7, 6.0, 28.5, 5.0),   # GW190413_134308
    (41.0, 7.0, 30.5, 6.0),   # GW190421_213856
    (37.3, 5.0, 28.1, 4.0),   # GW190424_180648
    (2.0, 0.3, 1.5, 0.2),     # GW190425
    (5.6, 1.0, 1.3, 0.2),     # GW190426_152155
    (36.9, 5.0, 27.1, 4.0),   # GW190503_185404
    (24.6, 4.0, 13.5, 3.0),   # GW190512_180714
    (34.0, 5.0, 24.5, 4.0),   # GW190513_205428
    (34.3, 5.0, 25.1, 4.0),   # GW190514_065416
    (37.1, 6.0, 27.3, 5.0),   # GW190517_055101
    (63.7, 10.0, 40.6, 7.0),  # GW190519_153544
    (85.0, 15.0, 66.0, 12.0), # GW190521
    (37.7, 6.0, 28.9, 5.0),   # GW190521_074359
    (32.8, 5.0, 23.9, 4.0),   # GW190527_092055
    (49.5, 8.0, 33.0, 6.0),   # GW190602_175927
    (37.5, 6.0, 27.6, 5.0),   # GW190620_030421
    (33.1, 5.0, 24.2, 4.0),   # GW190630_185205
    (43.8, 7.0, 31.0, 5.0),   # GW190701_203306
    (29.1, 5.0, 15.6, 3.0),   # GW190702_222702
    (41.3, 7.0, 29.8, 5.0),   # GW190706_222641
    (16.2, 3.0, 10.1, 2.0),   # GW190707_093326
    (28.5, 5.0, 15.8, 3.0),   # GW190708_232457
    (46.3, 8.0, 31.5, 6.0),   # GW190719_215514
    (8.6, 1.5, 5.3, 1.0),     # GW190720_000836
    (36.2, 6.0, 26.2, 5.0),   # GW190727_060333
    (19.8, 3.0, 12.1, 2.0),   # GW190728_064510
    (36.8, 6.0, 27.2, 5.0),   # GW190803_022618
    (34.2, 5.0, 24.5, 4.0),   # GW190805_211137
    (24.1, 2.5, 2.59, 0.08),  # GW190814 - КЛЮЧЕВОЕ
    (29.9, 5.0, 18.3, 3.0),   # GW190828_063405
    (33.3, 5.0, 23.8, 4.0),   # GW190828_065509
    (32.2, 5.0, 20.8, 4.0),   # GW190909_114149
    (42.8, 7.0, 30.3, 5.0),   # GW190910_112807
    (35.2, 5.0, 25.3, 4.0),   # GW190915_235702
    (8.9, 1.5, 5.0, 1.0),     # GW190924_021846
    (38.9, 6.0, 27.6, 5.0),   # GW190929_012149
    (37.2, 6.0, 26.7, 5.0),   # GW190930_133541
]

# Собираем все массы
masses = []
errors = []

for m1, e1, m2, e2 in gwtc1_masses:
    masses.extend([m1, m2])
    errors.extend([e1, e2])

for m1, e1, m2, e2 in gwtc2_masses:
    masses.extend([m1, m2])
    errors.extend([e1, e2])

masses = np.array(masses)
errors = np.array(errors)

print("=" * 60)
print("ПРОВЕРКА ТЕОРИИ ХРОНОСФЕРЫ")
print("=" * 60)
print(f"Всего масс: {len(masses)}")
print(f"Диапазон: {masses.min():.1f} - {masses.max():.1f} M_⊙")

# Фильтр: только интересующий диапазон 1-10 M_⊙
mask = (masses >= 1) & (masses <= 10)
m_light = masses[mask]
e_light = errors[mask]

print(f"\nМассы в диапазоне 1-10 M_⊙: {len(m_light)}")
print(f"События в массовой щели (2-3 M_⊙): {len(m_light[(m_light >= 2) & (m_light <= 3)])}")

# Монте-Карло распределение
n_samples = 1000
bins = np.linspace(1, 5, 30)
hist_sum = np.zeros(len(bins)-1)

for i in range(n_samples):
    sample = []
    for m, e in zip(m_light, e_light):
        if e > 0 and m > 0:
            log_m = np.log(m)
            log_e = np.log((m + e) / m)
            sample.append(np.exp(np.random.normal(log_m, log_e)))
        else:
            sample.append(m)
    h, _ = np.histogram(sample, bins=bins)
    hist_sum += h

hist_avg = hist_sum / n_samples
centers = (bins[:-1] + bins[1:]) / 2
hist_norm = hist_avg / hist_avg.sum() / np.diff(bins)

# Поиск пика в щели
gap_mask = (centers >= 2) & (centers <= 3)
x_gap = centers[gap_mask]
y_gap = hist_norm[gap_mask]
peak_mass = x_gap[np.argmax(y_gap)]
peak_val = y_gap[np.argmax(y_gap)]

# Фон
bg_mask = (centers >= 1) & (centers <= 5) & (~gap_mask)
bg_mean = hist_norm[bg_mask].mean()
bg_std = hist_norm[bg_mask].std()
sigma = (peak_val - bg_mean) / bg_std if bg_std > 0 else 0

print(f"\n[РЕЗУЛЬТАТ]")
print(f"Пик в массовой щели: {peak_mass:.2f} M_⊙")
print(f"Значимость пика: {sigma:.1f}σ")
print(f"Отклонение от 2.5 M_⊙: {abs(peak_mass - 2.5):.2f} M_⊙")

# Построение графика
plt.figure(figsize=(10, 6))
plt.bar(centers, hist_norm, width=np.diff(bins), alpha=0.7, 
        color='blue', label='Данные GWTC-1+2')
plt.axvline(x=2.5, color='red', linestyle='--', linewidth=2,
            label='Предсказание Хроносферы (2.5 M⊙)')
plt.axvline(x=peak_mass, color='green', linestyle=':',
            label=f'Пик в данных: {peak_mass:.2f} M⊙ ({sigma:.1f}σ)')
plt.axvspan(2, 3, alpha=0.1, color='gray', label='Массовая щель')
plt.plot(2.59, 0, 'r*', markersize=15, label='GW190814 (2.59 M⊙)')
plt.xlabel('Масса (M⊙)')
plt.ylabel('Плотность вероятности')
plt.title('Проверка теории Хроносферы: массовая щель')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(1, 5)
plt.tight_layout()

# Сохраняем график
filename = 'mass_gap_hronosphere.png'
plt.savefig(filename, dpi=150)
print(f"\n✅ График сохранён как {filename}")

# Показываем файлы
print("\nФайлы в директории:", os.listdir())

# Скачиваем на компьютер
from google.colab import files
files.download(filename)

# Показываем график
plt.show()
