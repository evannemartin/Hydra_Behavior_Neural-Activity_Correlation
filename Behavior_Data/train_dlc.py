import tensorflow as tf
import deeplabcut

# CODE FOR DEBUGGING TENSORFLOW
#print(tf.config.list_physical_devices('GPU'))
#tf.debugging.set_log_device_placement(True)
# Create some tensors
#a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#c = tf.matmul(a, b)
#print(c)
#tf.debugging.set_log_device_placement(False)
#input()

config_path = "path_to_config/config.yaml"

# Train
deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')
deeplabcut.train_network(config_path, saveiters=10000, maxiters=500000, allow_growth=True) #the maxiters recommended for Res-Nets


# Test
deeplabcut.evaluate_network(config_path, plotting=True)

deeplabcut.analyze_videos(config_path,['path_to_video_unlabeled1','path_to_video_unlabeled2','...'],auto_track=True, save_as_csv=True)
deeplabcut.create_labeled_video(config_path,['path_to_video_unlabeled1','path_to_videounlabeled2','...'], save_frames = True)
