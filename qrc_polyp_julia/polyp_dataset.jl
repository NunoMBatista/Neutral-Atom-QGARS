using Images, FileIO, ImageTransformations, Random

function create_polyp_dataset(polyp_dir, no_polyp_dir; split_ratio=0.8, target_size=(28, 28))
    # Load and process polyp images
    polyp_files = readdir(polyp_dir, join=true)
    no_polyp_files = readdir(no_polyp_dir, join=true)
    
    # Shuffle files
    Random.seed!(42) 
    shuffle!(polyp_files)
    shuffle!(no_polyp_files)
    
    # Define train/test split
    n_polyp_train = floor(Int, length(polyp_files) * split_ratio)
    n_no_polyp_train = floor(Int, length(no_polyp_files) * split_ratio)
    
    polyp_train = polyp_files[1:n_polyp_train]
    polyp_test = polyp_files[n_polyp_train+1:end]
    no_polyp_train = no_polyp_files[1:n_no_polyp_train]
    no_polyp_test = no_polyp_files[n_no_polyp_train+1:end]
    
    # Create train dataset
    train_files = vcat(polyp_train, no_polyp_train)
    train_targets = vcat(ones(Int, length(polyp_train)), zeros(Int, length(no_polyp_train)))
    
    # Create test dataset
    test_files = vcat(polyp_test, no_polyp_test)
    test_targets = vcat(ones(Int, length(polyp_test)), zeros(Int, length(no_polyp_test)))
    
    # Shuffle train and test data
    train_indices = shuffle(1:length(train_files))
    test_indices = shuffle(1:length(test_files))
    
    train_files = train_files[train_indices]
    train_targets = train_targets[train_indices]
    test_files = test_files[test_indices]
    test_targets = test_targets[test_indices]
    
    # Process images and create features arrays
    function process_images(files, target_size)
        n_samples = length(files)
        features = zeros(Float32, target_size[1], target_size[2], n_samples)
        
        for (i, file) in enumerate(files)
            img = load(file)
            # Convert to grayscale if needed
            if eltype(img) <: Colorant
                img = Gray.(img)
            end
            # Resize
            img_resized = imresize(img, target_size)
            # Normalize to [0,1] and convert to features array
            features[:, :, i] = Float32.(img_resized)
        end
        
        return features
    end
    
    # Process train and test images
    train_features = process_images(train_files, target_size)
    test_features = process_images(test_files, target_size)
    
    # Create metadata
    train_metadata = Dict{String, Any}(
        "n_samples" => length(train_files),
        "n_polyp" => length(polyp_train),
        "n_no_polyp" => length(no_polyp_train),
    )
    
    test_metadata = Dict{String, Any}(
        "n_samples" => length(test_files),
        "n_polyp" => length(polyp_test),
        "n_no_polyp" => length(no_polyp_test),
    )
    
    # Create dataset structs
    train_dataset = (
        metadata = train_metadata,
        split = :train,
        features = train_features,
        targets = train_targets
    )
    
    test_dataset = (
        metadata = test_metadata,
        split = :test,
        features = test_features,
        targets = test_targets
    )
    
    return train_dataset, test_dataset
end