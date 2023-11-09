import supervisely as sly
import os
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, file_exists
import shutil

from tqdm import tqdm

def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:

    images_path = os.path.join("stamver","scans","scans")
    masks_path = os.path.join("stamver","ground-truth-maps","ground-truth-maps")
    tags_path = os.path.join("stamver","info","info")
    ds_name = "ds"
    masks_ext = "-gt.png"
    anns_ext = ".txt"
    batch_size = 30


    def create_ann(image_path):
        labels = []
        tags = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        image_name = get_file_name(image_path)

        ann_path = os.path.join(tags_path, image_name + anns_ext)
        if file_exists(ann_path):
            with open(ann_path) as f:
                content = f.read().split("\n")[1]
                ann_data = content.split("\t")
                if ann_data[0] == "1":
                    signature_value = "not_present"
                else:
                    signature_value = "present"
                signature = sly.Tag(signature_meta, value=signature_value)
                tags.append(signature)

                if ann_data[1] == "0":
                    textoverlap_value = "false"
                else:
                    textoverlap_value = "true"
                textoverlap = sly.Tag(textoverlap_meta, value=textoverlap_value)
                tags.append(textoverlap)

                numstamps_value = int(ann_data[2])
                numstamps = sly.Tag(numstamps_meta, value=numstamps_value)
                tags.append(numstamps)

            mask_path = os.path.join(masks_path, image_name + masks_ext)
            ann_np = sly.imaging.image.read(mask_path)[:, :, 0]
            mask = ann_np == 0
            ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
            if ret < 5:
                for i in range(1, ret):
                    obj_mask = curr_mask == i
                    curr_bitmap = sly.Bitmap(obj_mask)
                    if curr_bitmap.area > 10:
                        curr_label = sly.Label(curr_bitmap, obj_class)
                        labels.append(curr_label)
            else:
                curr_bitmap = sly.Bitmap(mask)
                curr_label = sly.Label(curr_bitmap, obj_class)
                labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)


    obj_class = sly.ObjClass("stamp", sly.Bitmap)
    signature_meta = sly.TagMeta(
        "signature", sly.TagValueType.ONEOF_STRING, possible_values=["present", "not_present"]
    )
    textoverlap_meta = sly.TagMeta(
        "overlap_with_printed_text", sly.TagValueType.ONEOF_STRING, possible_values=["true", "false"]
    )
    numstamps_meta = sly.TagMeta("number_of_stamps", sly.TagValueType.ANY_NUMBER)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[obj_class],
        tag_metas=[signature_meta, textoverlap_meta, numstamps_meta],
    )
    api.project.update_meta(project.id, meta.to_json())

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    images_names = os.listdir(images_path)

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

    for img_names_batch in sly.batched(images_names, batch_size=batch_size):
        images_pathes_batch = [os.path.join(images_path, image_path) for image_path in img_names_batch]

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
        api.annotation.upload_anns(img_ids, anns_batch)

        progress.iters_done_report(len(img_names_batch))

    return project
