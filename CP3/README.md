# 1.041-1.200-CP3

1.041-1.200-CP3 for 2024S

## Installation

Find instruction from the CL3 3 handout.

## Usage

### 1. Training

For training, run the following command:

```
python training_main.py
```
By default, it is equal to running
```
python training_main.py --flow 750 --lane 4.0 --length 750 --speed 13.89 --left 0.25 --reward waittime
```
You can check the meanings of each parameter from `training_main.py`.

### 2. Transfer

For individual transfer, run the following command:

- For transfer along different inflows (from 750 veh/hr to 800 veh/hr):

```
python transfer_main.py --flow 800 --model_num 1 --source_path_name "intersection_flow750_lane4.0_length750_speed13.89_left0.25/" --num_episodes 50
```

- For transfer along different speed limits (from 13.89 m/s to 10.0 m/s):

```
python transfer_main.py --speed 10.0 --model_num 1 --source_path_name "intersection_flow1000_lane4.0_length750_speed13.89_left0.25/" --num_episodes 50
```

## Trained models and Results

You can find the trained models from this Google Drive link: [Link](https://drive.google.com/file/d/1pe2IYM2drxKG8OxWJrfL8XG71Zs7-WjS/view?usp=sharing)

## Design your own source tasks for zero-shot transfer learning

We encourage your creative selection of source training tasks to achieve higher transfer performance evaluated in all tasks. To do this, put your chosen source tasks into an array, such as `source_tasks_own = [10, 15, 5, 3, 8, 12, 18, 0, 19, 7]`, and proceed to test your choices using the `analysis/transfer_learning.ipynb` notebook. Please attach your result comparing performance of your model with baselines. Feel free to use the Temporal Transfer Learning strategies discussed in the course to guide your analysis.

## Model-based Transfer Learning

- You can find the ground truth transferability matrices from `./data/intersection_{variable}_tarnsfer_result.csv` and their heatmaps.
- In `./analysis/`, you can find the code to generate the heatmaps and the code for Model-based Transfer Learning.
