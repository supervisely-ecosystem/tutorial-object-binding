import os
from typing import List
from dotenv import load_dotenv
import cv2
import uuid
import supervisely as sly

# for convenient debug, has no effect in production
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

################################################################
# Part 1: Create labels with binding and upload them to server #
################################################################


api = sly.Api()

team_id = int(os.environ["CONTEXT_TEAMID"])
workspace_id = int(os.environ["CONTEXT_WORKSPACEID"])

# create empty project with one dataset on server
project = api.project.create(
    workspace_id, name="tytorial-bindings", change_name_if_conflict=True
)
dataset = api.dataset.create(project.id, name="dataset-01")
print(f"Open project in browser: {project.url}")

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
labels_masks: List[sly.Label] = []
for mask_path in [
    "images/car_masks/car_01.png",
    "images/car_masks/car_02.png",
    "images/car_masks/car_03.png",
]:
    # read only first channel of the black-and-white image
    image_black_and_white = cv2.imread(mask_path)[:, :, 0]

    # supports masks with values (0, 1) or (0, 255) or (False, True)
    mask = sly.Bitmap(image_black_and_white)
    label = sly.Label(geometry=mask, obj_class=class_car)
    labels_masks.append(label)

ann = sly.Annotation(img_size=[height, width], labels=labels_masks)

# we know that all three masks are parts of a single object
# let's bind them together:
key = uuid.uuid4().hex  # key can be any unique string
for label in ann.labels:
    label.binding_key = key

# upload annotation to server
api.annotation.upload_ann(image_info.id, ann)

# comment to test part 2
# exit(0)

################################################################
# Part 2: Work with existing binding
################################################################

# get project meta from server
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project.id))

# get annotation from server
ann_json = api.annotation.download_json(image_info.id)
ann = sly.Annotation.from_json(ann_json, meta)

# access to binnding keys
print("Labels bindings:")
for label in ann.labels:
    print(label.binding_key)

# get all binding groups
binds = ann.get_bindings()
for i, (binding_key, labels) in enumerate(binds.items()):
    if binding_key is not None:
        print(f"Group # {i} [key={binding_key}] has {len(labels)} labels")
    else:
        # Binding key None defines all labels that do not belong to any binding group
        print(f"{len(labels)} labels do not have any labels")

# discard binding for some labels. For example for all labels of class car
for label in ann.labels:
    if label.obj_class.name == "car":
        label.binding_key = None

# or you can discard binding for all labels in annotation
ann.discard_bindings()

# upload updated annotation to server
api.annotation.upload_ann(image_info.id, ann)
