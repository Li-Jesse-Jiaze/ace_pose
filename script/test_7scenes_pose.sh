#!/bin/bash

# Define dataset names
datasets=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

# Define pose_refinement methods
pose_refinements=("none" "adamw" "lm")

# Fixed patch_threshold value
patch_threshold=0.0

# External focal length
focal_length=525

# Iterate through each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # Iterate through each pose_refinement method
    for refinement in "${pose_refinements[@]}"; do
        echo "  Using pose_refinement method: $refinement"
        
        # Determine additional parameters and result folder suffix based on refinement method
        additional_params=""
        if [ "$refinement" == "lm" ]; then
            additional_params="--pose_refinement_f 100"
        fi
        
        # Create a unique results folder
        results_folder="results_Cambridge/${dataset}_refinement_${refinement}"
        mkdir -p "$results_folder"
        
        # Define paths
        train_rgb_path="data/7scenes_${dataset}/train/rgb/*.png"
        train_pose_path="data/7scenes_${dataset}/train/poses/*.pose_noisy.txt"
        test_rgb_path="data/7scenes_${dataset}/test/rgb/*.png"
        test_pose_path="data/7scenes_${dataset}/test/poses/*.txt"
        
        # Run train_ace.py
        echo "    Training ACE network..."
        python train.py "$train_rgb_path" "$results_folder/ace_network.pt" \
            --pose_files "$train_pose_path" \
            --pose_refinement "$refinement" \
            --use_external_focal_length $focal_length \
            --use_aug False \
            --patch_threshold $patch_threshold \
            $additional_params
        
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
            echo "    reloc.py failed. Skipping this combination."
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
