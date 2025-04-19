import kagglehub

# Download latest version
path = kagglehub.dataset_download("balraj98/cvcclinicdb")

print("Path to dataset files:", path)