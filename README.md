# Лабораторная работа 2 — JPEG-inspired image compressor

Полная реализация JPEG-подобного компрессора с нуля: RGB↔YCbCr,
DCT (наивная и матричная), квантование, зигзаг-обход, разностное
кодирование DC, RLE для AC, кодирование переменной длины и Хаффман
по стандарту ITU-T.81, собственный файловый формат.

## Установка

```bash
pip install numpy pillow matplotlib jupyter
```

## Структура

```
jpeg_lab/
├── src/                      — все модули
│   ├── raw_format.py         — задание 1.1 (собственный raw формат)
│   ├── color_space.py        — задание 1.2 (RGB <-> YCbCr)
│   ├── resampling.py         — задания 1.3 и 2.1 (down/upsample, интерполяция)
│   ├── dct.py                — задания 1.4 и 2.6 (DCT, квантование, quality)
│   ├── zigzag.py             — задание 2.2
│   ├── entropy.py            — задания 2.3 и 2.4 (DC diff, RLE, VLI)
│   ├── huffman.py            — задание 2.5 (таблицы ITU-T.81 + кодер/декодер)
│   ├── compressor.py         — задание 2.7 (полный цикл + файловый формат .myjpg)
│   └── test_data.py          — генерация 5 тестовых изображений
├── images/                   — исходные PNG
├── output/                   — сжатые .myjpg, декодированные PNG, графики
├── notebooks/
│   └── demo.ipynb            — интерактивная демонстрация (откройте в VS Code)
├── run_tests.py              — юнит-тесты всех модулей
└── run_experiments.py        — запускает все эксперименты + строит графики
```

## Запуск

```bash
# 1. сгенерировать тестовые изображения
python -m src.test_data

# 2. пройти все юнит-тесты
python run_tests.py

# 3. провести все эксперименты (нужно для отчёта)
python run_experiments.py
```

После этого:
- `output/*.myjpg` — сжатые файлы для каждой картинки × каждого quality
- `output/*_decoded.png` — декодированные обратно (для визуального сравнения)
- `output/size_vs_quality_*.png` — графики
- `output/size_report.csv` — сырые числа

## Открытие в VS Code

1. Открыть папку `jpeg_lab/` (`File → Open Folder`).
2. Выбрать интерпретатор Python с установленными numpy/pillow/matplotlib.
3. `notebooks/demo.ipynb` работает прямо в VS Code с расширением Jupyter.

## Файловый формат .myjpg

```
magic        : 4  bytes  = b'MJPG'
version      : 1  byte
color_mode   : 1  byte   (0 = grayscale, 1 = YCbCr color)
quality      : 1  byte   (1..99)
reserved     : 1  byte
width        : 4  bytes  (uint32 LE)
height       : 4  bytes  (uint32 LE)
payload_len  : 4  bytes  (uint32 LE)
payload      : Huffman bitstream
```

Таблицы Хаффмана и квантования не хранятся в файле — декодер использует
те же фиксированные стандартные таблицы ITU-T.81, масштабируя Q-таблицу
по сохранённому значению `quality`.

## Результаты (5 тестовых изображений, Q=10..90)

| image | raw size | q=10 | q=50 | q=90 |
|---|---|---|---|---|
| lena_synth (color) | 786 448 B | 9 126 B (87×) | 14 874 B (53×) | 32 866 B (24×) |
| color_pattern      | 786 448 B | 12 797 B (62×) | 25 186 B (31×) | 47 906 B (16×) |
| gray               | 262 160 B |  6 269 B (42×) | 10 072 B (26×) | 20 048 B (13×) |
| bw_round           | 262 160 B |  7 193 B (36×) | 12 758 B (21×) | 20 552 B (13×) |
| bw_dither          | 262 160 B | 58 611 B ( 4×) | 115 444 B (2×) | 206 760 B (1×) |

**Замечание по dither:** дизерированное изображение состоит из
высокочастотного шума — DCT его «размазывает» по всем коэффициентам,
квантование почти ничего не зануляет, поэтому коэффициент сжатия
очень низкий. Это известный факт: JPEG плохо работает с дизерингом.

## Сложность этапов (краткое резюме)

| этап | время | память |
|---|---|---|
| RGB↔YCbCr | O(HW) | O(HW) |
| split 8×8 blocks | O(HW) | O(HW) |
| DCT (naive per block) | O((NM)²) | O(NM) |
| DCT (matrix per block) | O(N³) | O(N²) |
| Квантование | O(NM) | O(NM) |
| Зигзаг | O(NM) | O(NM) |
| DC diff | O(B) (B = число блоков) | O(B) |
| AC RLE | O(B·NM) | O(B·NM) |
| Хаффман | O(общей длины потока) | O(размера таблицы) |

Общая асимптотика: **O(HW·log(...)) ≈ O(HW)** для фиксированного
размера блока.
