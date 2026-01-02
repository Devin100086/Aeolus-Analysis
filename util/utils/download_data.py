import kagglehub

# Download latest version
path = kagglehub.dataset_download("flnny123/mfddmulti-modal-flight-delay-dataset")

print("Path to dataset files:", path)