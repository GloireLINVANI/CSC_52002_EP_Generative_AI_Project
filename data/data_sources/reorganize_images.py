import os
import shutil
import argparse

def reorganize_images(source_root, destination_root):
    # Ensure destination exists
    os.makedirs(destination_root, exist_ok=True)

    # Traverse the source directory
    for dataset in os.listdir(source_root):
        dataset_path = os.path.join(source_root, dataset)
        if not os.path.isdir(dataset_path):
            continue  # Skip non-directory files

        for mask in os.listdir(dataset_path):
            mask_path = os.path.join(dataset_path, mask)
            if not os.path.isdir(mask_path):
                continue

            for model in os.listdir(mask_path):
                model_path = os.path.join(mask_path, model)
                if not os.path.isdir(model_path):
                    continue

                for batch in os.listdir(model_path):
                    batch_path = os.path.join(model_path, batch)
                    if not os.path.isdir(batch_path):
                        continue

                    for image in os.listdir(batch_path):
                        image_path = os.path.join(batch_path, image)
                        if not os.path.isfile(image_path):
                            continue

                        # Construct the new filename and path
                        new_filename = f"{os.path.splitext(image)[0]}_{batch}.png"
                        new_dir = os.path.join(destination_root, mask, dataset, model)
                        os.makedirs(new_dir, exist_ok=True)
                        new_path = os.path.join(new_dir, new_filename)

                        # Move the file
                        shutil.move(image_path, new_path)
                        print(f"Moved: {image_path} -> {new_path}")

    print("Reorganization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize images into a new folder structure.")
    parser.add_argument("source_root", help="Path to the source dataset root")
    parser.add_argument("destination_root", help="Path to the destination root")

    args = parser.parse_args()
    reorganize_images(args.source_root, args.destination_root)