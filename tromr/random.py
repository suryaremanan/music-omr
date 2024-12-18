from configs import getconfig

# Load configuration
config_path = '/home/surya/Desktop/musicai/data/Polyphonic-TrOMR/config.yaml'
config = getconfig(config_path)

# Access settings
print("Image Directory:", config.filepaths.image_dir)
print("Batch Size:", config.training.batch_size)
