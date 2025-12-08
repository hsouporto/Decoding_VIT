import os
import random

# Base folder relative to current directory
base_folder = "checkpoints"
os.makedirs(base_folder, exist_ok=True)

# Model groups
cnn_models = ["resnet50", "resnet50_reduced_bottleneck", "regnety_z", "resnet_rs",
              "efficientnetv2", "convnext_t"]

vit_models = ["vit_b", "igpt", "deit_b", "crossvit", "t2t_vit", "swin_b",
              "mae_b", "beit_b", "swinv2_b", "beitv2_b", "dinov2_b", "d_igpt"]

hybrid_models = ["levit_b", "moa_t_2", "fastvit_b", "cait_hybrid", "convnext_vit_hybrid",
                 "edgevit", "mobilevit_b", "coatnet", "cvt", "convit"]

# Function to create large dummy h5 files (100MB–500MB)
def create_large_h5_files(folder_name, models):
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    for model in models:
        size_mb = random.randint(100, 500)  # Random size between 100MB and 500MB
        size_bytes = size_mb * 1024 * 1024
        filepath = os.path.join(folder_path, f"{model}.h5")
        
        with open(filepath, "wb") as f:
            chunk_size = 10 * 1024 * 1024  # write in 10MB chunks to avoid memory issues
            for _ in range(size_bytes // chunk_size):
                f.write(os.urandom(chunk_size))
            remaining = size_bytes % chunk_size
            if remaining:
                f.write(os.urandom(remaining))
        
        print(f"Created {filepath} with size {size_mb} MB")

# Create files
create_large_h5_files("cnn", cnn_models)
create_large_h5_files("vit", vit_models)
create_large_h5_files("hybrid", hybrid_models)

