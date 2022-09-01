import deeplabcut

project_name = "hydra_vion"
experimenter = "Evanne_Martin" #better to not put space
config_path = deeplabcut.create_new_project(project_name, experimenter, ['Full path of video 1', 'Full path of video2','...'], multianimal=False, copy_videos=True)
# config_path should always be absolute
# you can now see the folder created


#to add videos later in the project without labelling again everything :
#deeplabcut.add_new_videos(config_path, ['Full path of video 1 to add', 'Full path of video 2 to add'], copy_videos=True)

deeplabcut.extract_frames(config_path,mode="automatic",algo="kmeans",userfeedback=False)
deeplabcut.label_frames(config_path)
