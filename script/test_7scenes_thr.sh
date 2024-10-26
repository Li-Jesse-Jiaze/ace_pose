#!/bin/bash

# Define dataset names
datasets=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

# Define patch_threshold values
patch_thresholds=(0.0 0.1 0.2)

# External focal length
focal_length=525

# Iterate through each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # Iterate through each patch_threshold value
    for threshold in "${patch_thresholds[@]}"; do
        echo "  Using patch_threshold: $threshold"
        
        # Create a unique results folder
        results_folder="results_7Scenes/${dataset}_threshold_${threshold}"
        mkdir -p "$results_folder"
        
        # Define paths
        train_rgb_path="data/7scenes_${dataset}/train/rgb/*.png"
        train_pose_path="data/7scenes_${dataset}/train/poses/*.pose.txt"
        test_rgb_path="data/7scenes_${dataset}/test/rgb/*.png"
        test_pose_path="data/7scenes_${dataset}/test/poses/*.txt"
        
        # Run train_ace.py
        echo "    Training ACE network..."
        python train.py "$train_rgb_path" "$results_folder/ace_network.pt" \
            --pose_files "$train_pose_path" \
            --pose_refinement none \
            --use_external_focal_length $focal_length \
            --use_aug False \
            --patch_threshold $threshold
        
        # Check if train_ace.py was successful
        if [ $? -ne 0 ]; then
            echo "    train_ace.py failed. Skipping this combination."
            continue
        fi
        
        # Run register_mapping.py
        echo "    Registering mapping..."
        python reloc.py "$test_rgb_path" "$results_folder/ace_network.pt" \
            --use_external_focal_length $focal_length \
            --session query
        
        # Check if register_mapping.py was successful
        if [ $? -ne 0 ]; then
            echo "    register_mapping.py failed. Skipping this combination."
            continue
        fi
        
        # Run eval_poses.py and save output
        echo "    Evaluating poses..."
        eval_output="${results_folder}/eval_poses.txt"
        python eval_poses.py "$results_folder/poses_query.txt" "$test_pose_path" 2>&1 | tee "$eval_output"
        
        # Check if eval_poses.py was successful
        if [ $? -ne 0 ]; then
            echo "    eval_poses.py failed. Check $eval_output for details."
        else
            echo "    Results saved to $results_folder, evaluation output saved to $eval_output"
        fi
    done
done

echo "Batch testing completed."
