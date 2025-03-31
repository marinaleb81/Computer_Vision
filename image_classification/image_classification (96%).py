import os
import math
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import random
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Конфигурация с вынесенными параметрами для простоты
CONFIG = {
    # Пути к данным локальные
    #"test_dir": r"e:\Python_projects\Machine_Learning\pythonProject\Компьютерное зрение\test",
    #"train_dir": r"e:\Python_projects\Machine_Learning\pythonProject\Компьютерное зрение\train",
    
    # Пути к данным при выгрузке
    "test_dir": os.path.join(script_dir, "test"),
    "train_dir": os.path.join(script_dir, "train"),

    # Основные параметры обучения
    "image_size": (512, 512),  # Размер изображений для модели, было 448 - увеличение хорошо сказалось "image_size": (512, 512)
    "batch_size": 1,  # Размер батча, было 2 - уменьшение хорошо сказалось
    "learning_rate": 5e-6,  # Скорость обучения 5e-6 (точность 92%), попробовать 3e-6 (точность 89%)
    "epochs": 100,  # Максимальное количество эпох, было 50 - больше лучше, но дольше

    # Параметры ранней остановки
    "early_stopping": {
        "monitor": "accuracy",  # Метрика для мониторинга
        "patience": 10,  # Количество эпох без улучшения
        "restore_best_weights": True  # Восстановление лучших весов
    },

    # Параметры модели
    "model": {
        "base_model": "vgg16",  # Название базовой модели, лучше предсказывает
        "unfreeze_layers": 1,  # Количество размороженных слоев с конца, датасет маленький - 1-2 оптимально
        "dense_units": 256,  # Количество нейронов в полносвязном слое, хуже с 512 (оставить 256)
        "dropout_rate": 0.5  # Коэффициент dropout. Контролирует степень регуляризации - оптимальный - не трогать!
    },

    # Параметры базовой аугментации ImageDataGenerator
    "augmentation": {
        "rotation_range": 15,  # Диапазон поворота
        "width_shift_range": 0.15,  # Сдвиг по ширине
        "height_shift_range": 0.15,  # Сдвиг по высоте
        "shear_range": 0.1,  # Сдвиг
        "zoom_range": 0.15,  # Масштабирование
        "horizontal_flip": True,  # Горизонтальный переворот
        "fill_mode": "nearest",  # Метод заполнения
        "brightness_range": [0.85, 1.15]  # Диапазон яркости
    },

    # Параметры аугментации
    "custom_augmentation": {
        "noise": {
            "apply_prob": 0.2,  # Вероятность применения шума
            "sigma": 12  # Интенсивность шума
        },
        "blur": {
            "apply_prob": 0.4,  # Вероятность размытия
            "kernel_size": (5, 5)  # Размер ядра размытия
        },
        "sharpen": {
            "apply_prob": 0.4,  # Вероятность повышения резкости
            "strength": 9  # Интенсивность повышения резкости
        },
        "contrast": {
            "apply_prob": 0.6,  # Вероятность изменения контраста
            "factor_range": (0.5, 1.7)  # Диапазон изменения контраста
        },
        "dirt_simulation": {
            "apply_prob": 0.4,  # Вероятность добавления загрязнений
            "dots_range": (3, 15),  # Диапазон количества точек
            "radius_range": (1, 3),  # Диапазон радиуса точек
            "color_range": (0, 150)  # Диапазон цвета (темноты) точек
        }
    },

    "random_seed": 42
}


def plot_training_history(history, timestamp):
    """Создание и сохранение графика обучения"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    filename = f'training_history_{timestamp}.png'
    plt.savefig(filename)
    plt.close()


def apply_noise(image, config):
    """Добавление шума"""
    row, col, ch = image.shape
    mean = 0
    sigma = config["custom_augmentation"]["noise"]["sigma"]
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_blur(image, config):
    """Размытие"""
    prob = config["custom_augmentation"]["blur"]["apply_prob"]
    kernel_size = config["custom_augmentation"]["blur"]["kernel_size"]
    if random.random() < prob:
        return cv2.GaussianBlur(image, kernel_size, 0)
    return image


def apply_sharpen(image, config):
    """Резкость"""
    prob = config["custom_augmentation"]["sharpen"]["apply_prob"]
    strength = config["custom_augmentation"]["sharpen"]["strength"]
    if random.random() < prob:
        kernel = np.array([[-1, -1, -1], [-1, strength, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    return image


def adjust_contrast(image, config):
    """Контрастность"""
    prob = config["custom_augmentation"]["contrast"]["apply_prob"]
    factor_range = config["custom_augmentation"]["contrast"]["factor_range"]
    if random.random() < prob:
        factor = random.uniform(factor_range[0], factor_range[1])
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    return image


def apply_dirt_simulation(image, config):
    """Имитация мелких загрязнений"""
    prob = config["custom_augmentation"]["dirt_simulation"]["apply_prob"]
    dots_range = config["custom_augmentation"]["dirt_simulation"]["dots_range"]
    radius_range = config["custom_augmentation"]["dirt_simulation"]["radius_range"]
    color_range = config["custom_augmentation"]["dirt_simulation"]["color_range"]

    if random.random() < prob:
        number_of_dots = random.randint(dots_range[0], dots_range[1])
        for _ in range(number_of_dots):
            x = random.randint(0, image.shape[1] - 1)
            y = random.randint(0, image.shape[0] - 1)
            radius = random.randint(radius_range[0], radius_range[1])
            color = random.randint(color_range[0], color_range[1])
            cv2.circle(image, (x, y), radius, (color, color, color), -1)
    return image


def create_custom_preprocess_function(config):
    def custom_preprocess_function(image):
        if not isinstance(image, np.ndarray):
            img = np.array(image)
        else:
            img = image.copy()

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Различные техники аугментации с вероятностью
        if random.random() < config["custom_augmentation"]["noise"]["apply_prob"]:
            img = apply_noise(img, config)

        img = apply_blur(img, config)
        img = apply_sharpen(img, config)
        img = adjust_contrast(img, config)
        img = apply_dirt_simulation(img, config)

        return img / 255.0

    return custom_preprocess_function


def setup_data_generators(config):
    custom_preprocess = create_custom_preprocess_function(config)

    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocess,
        rotation_range=config["augmentation"]["rotation_range"],
        width_shift_range=config["augmentation"]["width_shift_range"],
        height_shift_range=config["augmentation"]["height_shift_range"],
        shear_range=config["augmentation"]["shear_range"],
        zoom_range=config["augmentation"]["zoom_range"],
        horizontal_flip=config["augmentation"]["horizontal_flip"],
        fill_mode=config["augmentation"]["fill_mode"],
        brightness_range=config["augmentation"]["brightness_range"]
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        config["train_dir"],
        target_size=config["image_size"],
        batch_size=config["batch_size"],
        class_mode='binary',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        config['test_dir'],
        target_size=config["image_size"],
        batch_size=1,
        class_mode=None,
        shuffle=False
    )

    return train_generator, test_generator


def build_model(config):
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(*config["image_size"], 3))

    # Настройка замораживания слоев
    base_model.trainable = True
    unfreeze_layers = config["model"]["unfreeze_layers"]
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    # Структура верхних слоев
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(config["model"]["dense_units"], activation='relu'),
        layers.Dropout(config["model"]["dropout_rate"]),
        layers.Dense(1, activation='sigmoid')
    ])

    # Компиляция модели
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def visualize_augmentations(train_generator, num_samples=5):
    """Визуализация примеров аугментации для проверки"""
    plt.figure(figsize=(15, 10))

    for i in range(num_samples):
        img_batch, label_batch = next(train_generator)
        for j in range(min(4, len(img_batch))):
            plt.subplot(num_samples, 4, i * 4 + j + 1)
            img = img_batch[j]
            label = 'dirty' if label_batch[j] > 0.5 else 'cleaned'
            plt.imshow(img)
            plt.title(f"{label}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.close()
    print("Примеры аугментации сохранены в 'augmentation_examples.png'")


def create_submission(model, test_generator, config, timestamp):
    """Создание файла с предсказаниями"""
    steps = math.ceil(test_generator.samples / test_generator.batch_size)
    predictions = model.predict(test_generator, steps=steps, verbose=1)

    filenames = []
    for f in test_generator.filenames:
        f = f.replace('\\', '/')
        filename = f.split('/')[-1].replace('.jpg', '')  # только имя файла без пути и расширения
        filenames.append(filename)

    submission = pd.DataFrame({
        'id': filenames,
        'label': (predictions.flatten() > 0.5).astype(int)
    })

    submission['label'] = submission['label'].map({
        0: 'cleaned',
        1: 'dirty'
    })

    # Имя файла с временной меткой
    output_file = f'submission_enhanced_aug_{timestamp}.csv'
    submission.to_csv(output_file, index=False)

    # Сохраняем конфигурацию с тем же именем, для анализа
    config_file = f'config_enhanced_aug_{timestamp}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

    cleaned_count = (submission['label'] == 'cleaned').sum()
    dirty_count = (submission['label'] == 'dirty').sum()
    print(f"\nСтатистика предсказаний:")
    print(f"- Cleaned: {cleaned_count} ({cleaned_count / len(submission) * 100:.1f}%)")
    print(f"- Dirty: {dirty_count} ({dirty_count / len(submission) * 100:.1f}%)")
    print(f"Файл с предсказаниями сохранен как: {output_file}")
    print(f"Конфигурация сохранена как: {config_file}")

    return submission


def main():
    # Установка случайных сидов для воспроизводимости
    random_seed = CONFIG["random_seed"]
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)

    print("=== Классификация изображений Cleaned vs Dirty с фокусом на аугментацию ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    train_generator, test_generator = setup_data_generators(CONFIG)

    print("Генерация примеров аугментированных изображений...")
    visualize_augmentations(train_generator)

    print("\nСоздание и обучение модели...")
    model = build_model(CONFIG)
    early_stopping = EarlyStopping(
        monitor=CONFIG["early_stopping"]["monitor"],
        patience=CONFIG["early_stopping"]["patience"],
        restore_best_weights=CONFIG["early_stopping"]["restore_best_weights"],
        verbose=1
    )

    print("\nНачало обучения...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=CONFIG["epochs"],
        callbacks=[early_stopping],
        verbose=1
    )

    print("\nСоздание графика обучения...")
    plot_training_history(history, timestamp)

    print("\nСоздание файла с предсказаниями...")
    submission = create_submission(model, test_generator, CONFIG, timestamp)

    print("\nОбучение и создание предсказаний завершено успешно!")


if __name__ == "__main__":
    main()
