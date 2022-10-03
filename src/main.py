import os
from dotenv import load_dotenv
import cv2
import supervisely as sly

# for convenient debug, has no effect in production
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

team_id = int(os.environ["CONTEXT_TEAMID"])
workspace_id = int(os.environ["CONTEXT_WORKSPACEID"])

# create empty project with one dataset on server
project = api.project.create(workspace_id, name="tytorial-bindings", change_name_if_conflict=True)
dataset = api.dataset.create(project.id, name="dataset-01")

# define annotation classes and upload them on server
class_car = sly.ObjClass(name="car", geometry_type=sly.Bitmap, color=[255, 0, 255])
meta = sly.ProjectMeta(obj_classes=[class_car])
api.project.update_meta(project.id, meta)

# upload demo image
image_path = "images/image.jpg"
image_name = sly.fs.get_file_name_with_ext(image_path)
image_info = api.image.upload_path(dataset.id, image_name, image_path)
print(f"Image has been successfully uploaded: id={image_info.id}")
# will be needed later for creating annotation
height, width = cv2.imread(image_path).shape[0:2] 

# create Supervisely annotation from masks images
labels_masks = []
for mask_path in [
    "images/car_masks/car_01.jpg",
    "images/car_masks/car_02.jpg",
    "images/car_masks/car_03.jpg",
]:
    # read only first channel of the black-and-white image
    image_black_and_white = cv2.imread(mask_path)[:, :, 0]
    
    # supports masks with values (0, 1) or (0, 255) or (False, True)
    mask = sly.Bitmap(image_black_and_white)
    label = sly.Label(geometry=mask, obj_class=class_car)
    labels_masks.append(label)

# create object bindings


ann = sly.Annotation(img_size=[height, width], labels=labels_masks)

# upload annotation without bindings


project_id = 13409
image_id = 3314153
image_info = api.image.get_info_by_id(image_id)

project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

# get bindings from existing annotation
ann_json = api.annotation.download_json(image_id)
ann = sly.Annotation.from_json(ann_json, project_meta)

# access to binnding keys
for label in ann.labels:
    print(label.binding_key)

# get all binding groups
binds = ann.get_bindings()
for binding_key, labels in binds.items():
    if binding_key is not None:
        print(f"Group [{binding_key}] has {len(labels)} labels")
    else:
        # Binding key None defines all labels that do not belong to any binding group
        print(f"{len(labels)} labels do not have any labels")
        
# discard binding for some labels. For example for labels of class car
for label in ann.labels:
    if label.obj_class.name == "car":
        label.binding_key = None

# discard binding for all labels in annotation
ann.discard_bindings()

# update annotation on server

