
# 🧱 ResNetTrunk

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![Stars](https://img.shields.io/github/stars/abobantai/ResNet-constructors?style=social)]()
[![Forks](https://img.shields.io/github/forks/abobantai/ResNet-constructors?style=social)]()

---

## 📚 Описание

Кастомный, настраиваемый конструктор **ResNet-подобных сетей** на PyTorch.

Поддерживает:
- произвольные блоки (`block`),
- любую архитектуру (число блоков, каналы, страйды),
- настраиваемые активации (передаваемые строками или слоями).

---

## 🛠 Установка

```bash
# Просто скопируйте файл в свой проект
```

---

## 🔥 Быстрый старт

```python
from model import create

# Простая модель
net = create(
    num_blocks=[2, 2, 2],
    num_classes=100,
    act1="relu",
    act2="silu"
)

print(net)
```

---

## ⚙️ Аргументы функции `create`

| Параметр      | Тип             | Описание |
|---------------|------------------|----------|
| `block`       | класс             | Блок сети (по умолчанию `ResBlockV1`) |
| `strides`     | список int        | Страйды этапов (по умолчанию `[1]*len(num_blocks)`) |
| `num_blocks`  | список int        | Количество блоков на каждом этапе |
| `channels`    | список int        | Количество каналов на каждом этапе |
| `koef`        | список int        | Множители ширины каналов (по умолчанию `[1]*len(num_blocks)`) |
| `num_classes` | int               | Количество выходных классов |
| `act1`        | строка/слой       | Первая активация в блоке |
| `act2`        | строка/слой       | Вторая активация в блоке |

---

## 🧩 Структура модулей

| Модуль                | Описание |
|------------------------|----------|
| `ResTruck`             | Архитектура сети: начальная свертка, блоки, глобальный pooling. |
| `Resnet_construct`     | Добавляет полносвязный слой для классификации. |
| `get_activation`       | Упрощает использование активаций. |
| `create`               | Удобная точка входа для сборки модели. |

---

## 🎨 Поддерживаемые активации

Можно передавать строку:

- `"relu"`
- `"silu"`
- `"gelu"`
- `"leaky_relu"`

Или сразу свой слой:

```python
from torch.nn import PReLU

model = create(
    num_blocks=[2, 2, 2],
    act1=PReLU(),
    act2="gelu"
)
```

---

## 🧠 Пример продвинутой модели

```python
# Глубокая сеть под ImageNet
net = create(
    num_blocks=[3, 4, 6, 3],
    strides=[1, 2, 2, 2],
    channels=[64, 128, 256, 512],
    koef=[1, 2, 4, 8],
    num_classes=1000,
    act1="leaky_relu",
    act2="silu"
)
```

---

## 📜 Лицензия

Этот проект лицензирован под [MIT License](LICENSE).
