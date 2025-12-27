import os
import shutil
import uuid
import imghdr
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

RAW_DIR = "DATASET_AGRINET/DATASET_AGRINET/TRAINING/MAIZE"
CLEAN_DIR = "maize_deficiency_clean"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42
SUPPORTED = {"jpeg", "png", "bmp"}
SPLIT = [0.7, 0.15, 0.15]

ImageFile.LOAD_TRUNCATED_IMAGES = False

def clean_and_restructure(raw_dir, clean_dir):
    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir)
    os.makedirs(clean_dir, exist_ok=True)

    bad_files = []
    saved_count = 0

    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    print(f"Found {len(classes)} classes: {classes}")

    for deficiency_class in classes:
        class_path = os.path.join(raw_dir, deficiency_class)
        dest_class = os.path.join(clean_dir, deficiency_class)
        os.makedirs(dest_class, exist_ok=True)

        for root, _, files in os.walk(class_path):
            for fname in files:
                src = os.path.join(root, fname)
                try:
                    if os.path.getsize(src) == 0:
                        bad_files.append((src, "zero size"))
                        continue

                    kind = imghdr.what(src)
                    if kind is None or kind.lower() not in SUPPORTED:
                        bad_files.append((src, f"unsupported kind={kind}"))
                        continue

                    with Image.open(src) as im:
                        im.verify()
                    with Image.open(src) as im:
                        im = im.convert("RGB")
                        
                        dest_filename = f"{uuid.uuid4().hex}.jpg"
                        dest_path = os.path.join(dest_class, dest_filename)
                        im.save(dest_path, format="JPEG", quality=90)
                        saved_count += 1
                        
                        import random
                        from PIL import ImageEnhance, ImageOps
                        
                        augmentations = []
                        
                        aug1 = im.transpose(Image.FLIP_LEFT_RIGHT)
                        augmentations.append(aug1)
                        
                        aug2 = im.rotate(random.uniform(-20, 20), fillcolor=(0, 0, 0))
                        augmentations.append(aug2)
                        
                        enhancer = ImageEnhance.Brightness(im)
                        aug3 = enhancer.enhance(random.uniform(0.7, 1.3))
                        augmentations.append(aug3)
                        
                        enhancer = ImageEnhance.Contrast(im)
                        aug4 = enhancer.enhance(random.uniform(0.8, 1.2))
                        augmentations.append(aug4)

                        width, height = im.size
                        crop_size = int(min(width, height) * 0.8)
                        left = random.randint(0, width - crop_size)
                        top = random.randint(0, height - crop_size)
                        aug5 = im.crop((left, top, left + crop_size, top + crop_size))
                        aug5 = aug5.resize(im.size)
                        augmentations.append(aug5)
                        
                        for idx, aug_img in enumerate(augmentations):
                            aug_filename = f"{uuid.uuid4().hex}_aug{idx}.jpg"
                            aug_path = os.path.join(dest_class, aug_filename)
                            aug_img.save(aug_path, format="JPEG", quality=90)
                            saved_count += 1

                except Exception as e:
                    bad_files.append((src, str(e)))

    print(f"Cleaned and saved {saved_count} images to {clean_dir}")
    print(f"Skipped {len(bad_files)} bad files")

clean_and_restructure(RAW_DIR, CLEAN_DIR)

def balance_classes(clean_dir):
    class_dirs = [d for d in os.listdir(clean_dir) if os.path.isdir(os.path.join(clean_dir, d))]
    class_counts = {}
    
    for class_dir in class_dirs:
        class_path = os.path.join(clean_dir, class_dir)
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        class_counts[class_dir] = len(files)
        print(f"{class_dir}: {len(files)} images")
    
    min_count = min(class_counts.values())
    print(f"Minimum class size: {min_count}, balancing all classes")
    
    import random
    random.seed(SEED)
    
    for class_dir in class_dirs:
        class_path = os.path.join(clean_dir, class_dir)
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        if len(files) > min_count:
            files_to_remove = random.sample(files, len(files) - min_count)
            for file_to_remove in files_to_remove:
                os.remove(os.path.join(class_path, file_to_remove))

balance_classes(CLEAN_DIR)

def split_dataset(clean_dir, split_ratio):
    final_base = "maize_deficiency_final"
    if os.path.exists(final_base):
        shutil.rmtree(final_base)
    os.makedirs(final_base, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(final_base, split), exist_ok=True)

    classes = [d for d in os.listdir(clean_dir) if os.path.isdir(os.path.join(clean_dir, d))]
    for deficiency_class in classes:
        files = [os.path.join(clean_dir, deficiency_class, f) for f in os.listdir(os.path.join(clean_dir, deficiency_class))]
        train_files, testval = train_test_split(files, test_size=1 - split_ratio[0], random_state=SEED)
        val_files, test_files = train_test_split(testval, test_size=split_ratio[2] / sum(split_ratio[1:]), random_state=SEED)

        for split, subset in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            dest = os.path.join(final_base, split, deficiency_class)
            os.makedirs(dest, exist_ok=True)
            for f in subset:
                shutil.copy(f, dest)
        
        print(f"{deficiency_class}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    return final_base

FINAL_DIR = split_dataset(CLEAN_DIR, SPLIT)

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(FINAL_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(FINAL_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(FINAL_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SHAPE)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = tf.keras.applications.resnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)

if num_classes == 2:
    outputs = layers.Dense(1, activation="sigmoid")(x)
    loss = "binary_crossentropy"
else:
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    loss = "sparse_categorical_crossentropy"

model = models.Model(inputs, outputs)
model.compile(optimizer=optimizers.Adam(1e-4), loss=loss, metrics=["accuracy"])
model.summary()

checkpoint = ModelCheckpoint("maize_deficiency_best.h5", monitor="val_loss", save_best_only=True)
early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

EPOCHS = 30

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early, reduce_lr]
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss=loss,
    metrics=["accuracy"]
)

checkpoint_ft = ModelCheckpoint("maize_deficiency_best.h5", monitor="val_loss", save_best_only=True)
early_ft = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr_ft = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

EPOCHS_FT = 15

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    callbacks=[checkpoint_ft, early_ft, reduce_lr_ft]
)

for key in history_ft.history:
    history.history[key].extend(history_ft.history[key])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("maize_deficiency_training_curves.png", dpi=300, bbox_inches='tight')
plt.show()

model.save("maize_deficiency_model_final.keras")
print("model saved: ")
print("   - maize_deficiency_model_final.keras")
print("   - maize_deficiency_best.h5")
print("   - maize_deficiency_classes.txt")

with open("maize_deficiency_classes.txt", "w") as f:
    for cls in class_names:
        f.write(f"{cls}\n")

test_loss, test_acc = model.evaluate(test_ds)
print(f"Model test accuracy: {test_acc:.4f}")
print(f"Model test loss: {test_loss:.4f}")
