import os
import pandas as pd
import shutil

def upfall_dataset(base_dir, output_dir, label_mapping):
    os.makedirs(output_dir, exist_ok=True)

    for subject in range(1, 18):
        for camera in range(1, 3):
            for activity in range(1, 12):
                for trial in range(1, 4):
                    csv_path = os.path.join(
                        base_dir,
                        f'Subject{subject}',
                        f'Activity{activity}',
                        f'Camera{camera}',
                        f'Subject{subject}Activity{activity}Trial{trial}.csv'
                    )

                    if not os.path.exists(csv_path):
                        continue

                    try:
                        data = pd.read_csv(csv_path, skiprows=2)
                    except pd.errors.EmptyDataError:
                        print(f"Invalid CSV: {csv_path}")
                        continue

                    for _, row in data.iterrows():
                        label = row[-1]
                        if label not in label_mapping:
                            continue

                        timestamp = row[0]
                        label_name = label_mapping[label]
                        folder_name = f'Subject{subject}Activity{activity}Trial{trial}Camera{camera}'
                        label_folder = os.path.join(output_dir, f'Camera{camera}', f'{label_name}_{folder_name}')
                        os.makedirs(label_folder, exist_ok=True)

                        img_name = f'{timestamp.replace(":", "_")}.png'
                        src_path = os.path.join(
                            base_dir,
                            f'Subject{subject}',
                            f'Activity{activity}',
                            f'Camera{camera}',
                            folder_name,
                            img_name
                        )
                        dest_path = os.path.join(label_folder, img_name)

                        if os.path.exists(src_path):
                            shutil.copy(src_path, dest_path)
                            print(f'Copied: {src_path} to {dest_path}')

    print("Organization complete.")

def main():
    label_mapping = {
        1: 'Falling forward using hands',
        2: 'Falling forward using knees',
        3: 'Falling backward',
        4: 'Falling sideways',
        5: 'Falling sitting in empty chair',
        6: 'Walking',
        7: 'Standing',
        8: 'Sitting',
        9: 'Picking up an object',
        10: 'Jumping',
        11: 'Laying'
    }
    
    parser = argparse.ArgumentParser(description="Process pose data from UP-FALL dataset.")
    parser.add_argument('--base_folder', type=str, required=True, help="Path to the base folder.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder.")
    
    args = parser.parse_args()
    
    upfall_dataset(args.base_folder, args.output_folder, label_mapping)

if __name__ == "__main__":
    main()