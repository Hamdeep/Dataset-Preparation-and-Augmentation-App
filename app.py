import os
import random
import zipfile
import shutil
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
import streamlit as st
import tempfile


def augment_image(image, augment_type):
    if augment_type == 'brightness':
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))
    elif augment_type == 'contrast':
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))
    elif augment_type == 'saturation':
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))
    elif augment_type == 'hue':
        image = image.convert('HSV')
        channels = list(image.split())
        channels[0] = channels[0].point(lambda p: (p + random.randint(-30, 30)) % 256)
        image = Image.merge('HSV', tuple(channels)).convert('RGB')
    elif augment_type == 'blur':
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))

    return image

def create_directories(base_dir):
    images_output_dir = os.path.join(base_dir, 'images')
    labels_output_dir = os.path.join(base_dir, 'labels')

    train_dir_images = os.path.join(images_output_dir, 'train')
    test_dir_images = os.path.join(images_output_dir, 'test')
    val_dir_images = os.path.join(images_output_dir, 'val')

    train_dir_labels = os.path.join(labels_output_dir, 'train')
    test_dir_labels = os.path.join(labels_output_dir, 'test')
    val_dir_labels = os.path.join(labels_output_dir, 'val')

    for directory in [train_dir_images, test_dir_images, val_dir_images, train_dir_labels, test_dir_labels, val_dir_labels]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    return images_output_dir, labels_output_dir

def copy_files(file_pairs, dest_dir_images, dest_dir_labels, images_root_dir, labels_root_dir):
    for image_file, label_file in file_pairs:
        shutil.copy(os.path.join(images_root_dir, image_file), os.path.join(dest_dir_images, image_file))
        shutil.copy(os.path.join(labels_root_dir, label_file), os.path.join(dest_dir_labels, label_file))

def augment_and_copy_files(images_root_dir, labels_root_dir, augmentations):
    all_image_files = os.listdir(images_root_dir)
    all_label_files = os.listdir(labels_root_dir)

    new_image_files = []
    new_label_files = []

    for image_filename in all_image_files:
        if image_filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_root_dir, image_filename)
            label_path = os.path.join(labels_root_dir, os.path.splitext(image_filename)[0] + '.txt')

            new_image_files.append(image_filename)
            new_label_files.append(os.path.splitext(image_filename)[0] + '.txt')

            if augmentations:
                # Open the image
                image = Image.open(image_path)

                for augment_type in augmentations:
                    if augmentations[augment_type]:
                        augmented_image = augment_image(image, augment_type)
                        
                        base_name, ext = os.path.splitext(image_filename)
                        random_number = random.randint(1000, 9999)
                        augmented_image_filename = f'{base_name}_{augment_type}_aug_{random_number}{ext}'
                        augmented_image_path = os.path.join(images_root_dir, augmented_image_filename)
                        augmented_label_filename = f'{base_name}_{augment_type}_aug_{random_number}.txt'
                        augmented_label_path = os.path.join(labels_root_dir, augmented_label_filename)

                        augmented_image.save(augmented_image_path)
                        new_image_files.append(augmented_image_filename)

                        if os.path.exists(label_path):
                            with open(label_path, 'r') as label_file:
                                label_content = label_file.read()
                            with open(augmented_label_path, 'w') as augmented_label_file:
                                augmented_label_file.write(label_content)
                            new_label_files.append(augmented_label_filename)

    return new_image_files, new_label_files

def process_zip_file(zip_file, train_ratio, test_ratio, val_ratio, augmentations):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)
        
        images_root_dir = os.path.join(tmpdirname, 'images')
        labels_root_dir = os.path.join(tmpdirname, 'labels')

        all_image_files, all_label_files = augment_and_copy_files(images_root_dir, labels_root_dir, augmentations)

        assert len(all_image_files) == len(all_label_files), "Mismatch between number of image and label files"

        test_val_ratio = test_ratio + val_ratio
        train_files, test_val_files = train_test_split(list(zip(all_image_files, all_label_files)), test_size=test_val_ratio / 100, random_state=42)
        test_files, val_files = train_test_split(test_val_files, test_size=val_ratio / test_val_ratio, random_state=42)

        output_dir = tempfile.mkdtemp()
        images_output_dir, labels_output_dir = create_directories(output_dir)

        copy_files(train_files, os.path.join(images_output_dir, 'train'), os.path.join(labels_output_dir, 'train'), images_root_dir, labels_root_dir)
        copy_files(test_files, os.path.join(images_output_dir, 'test'), os.path.join(labels_output_dir, 'test'), images_root_dir, labels_root_dir)
        copy_files(val_files, os.path.join(images_output_dir, 'val'), os.path.join(labels_output_dir, 'val'), images_root_dir, labels_root_dir)

        return output_dir

st.title("Dataset Preparation and Augmentation App")

# State management
if 'augment_applied' not in st.session_state:
    st.session_state.augment_applied = False

if 'dataset_processed' not in st.session_state:
    st.session_state.dataset_processed = False

# Initialize ratios
if 'train_ratio' not in st.session_state:
    st.session_state.train_ratio = 70

if 'test_ratio' not in st.session_state:
    st.session_state.test_ratio = 15

if 'val_ratio' not in st.session_state:
    st.session_state.val_ratio = 15

def update_ratios(source):
    if source == 'train' and st.session_state.train_ratio % 10 != 0:
        st.session_state.train_ratio = round(st.session_state.train_ratio / 10) * 10

    if source == 'train':
        remaining = 100 - st.session_state.train_ratio
        st.session_state.test_ratio = remaining // 2
        st.session_state.val_ratio = remaining // 2
    elif source == 'test':
        st.session_state.val_ratio = st.session_state.test_ratio
        st.session_state.train_ratio = 100 - 2 * st.session_state.test_ratio
    elif source == 'val':
        st.session_state.test_ratio = st.session_state.val_ratio
        st.session_state.train_ratio = 100 - 2 * st.session_state.val_ratio



uploaded_file = st.file_uploader("Upload a zip file", type="zip")

if uploaded_file is not None:
    augment = st.checkbox('Augment Images')
    
    if augment and not st.session_state.augment_applied:
        with st.form(key='augment_form'):
            st.markdown("**Select Augmentations**")
            augmentations = {
                'brightness': st.checkbox('Brightness'),
                'contrast': st.checkbox('Contrast'),
                'saturation': st.checkbox('Saturation'),
                'hue': st.checkbox('Hue'),
                'blur': st.checkbox('Blur')
            }
            apply_augment_button = st.form_submit_button(label='Apply Augmentation')

        if apply_augment_button:
            st.session_state.augment_applied = True
            st.session_state.augmentations = augmentations
            st.success("Augmentation applied successfully. Now, proceed to split the dataset.")

    if st.session_state.augment_applied or not augment:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input('Train (%)', min_value=0, max_value=100, value=st.session_state.train_ratio, step=10, key='train_ratio', on_change=update_ratios, args=('train',))
        with col2:
            st.number_input('Test (%)', min_value=0, max_value=100, value=st.session_state.test_ratio, key='test_ratio', on_change=update_ratios, args=('test',))
        with col3:
            st.number_input('Validation (%)', min_value=0, max_value=100, value=st.session_state.val_ratio, key='val_ratio', on_change=update_ratios, args=('val',))

        with st.form(key='split_form'):
            split_button = st.form_submit_button(label='Split Dataset')

        if split_button:
            if st.session_state.train_ratio + st.session_state.test_ratio + st.session_state.val_ratio != 100:
                st.error("The sum of train, test, and validation percentages must equal 100.")
            else:
                augmentations = st.session_state.augmentations if augment else {}
                output_dir = process_zip_file(uploaded_file, st.session_state.train_ratio, st.session_state.test_ratio, st.session_state.val_ratio, augmentations)
                st.session_state.dataset_processed = True
                st.session_state.output_dir = output_dir

if st.session_state.dataset_processed:
    try:
        output_zip_path = shutil.make_archive("prepared_dataset", 'zip', st.session_state.output_dir)

        with open(output_zip_path, 'rb') as f:
            st.download_button('Download prepared dataset', f, file_name='prepared_dataset.zip')

        shutil.rmtree(st.session_state.output_dir)

    except FileNotFoundError:
        print("There was an issue creating the zip file. Please try again.")
