import cv2
import glob
import numpy as np
import os
import pydicom

from pydicom.pixel_data_handlers.util import apply_voi_lut


def window_ct(x: np.ndarray, WL: int, WW: int) -> np.ndarray[np.uint8]:
    # applying windowing to CT
    lower, upper = WL - WW // 2, WL + WW // 2
    x = np.clip(x, lower, upper)
    x = (x - lower) / (upper - lower)
    return (x * 255.0).astype("uint8")


def convert_to_8bit(
    array: np.ndarray, windows: tuple[int, int] | list[tuple[int, int]]
) -> np.ndarray[np.uint8]:
    if not isinstance(windows, list):
        windows = [windows]

    array_list = []
    for win in windows:
        assert isinstance(win, tuple)
        WL, WW = win
        array_list.append(window_ct(array.copy(), WL, WW))

    array = np.stack(array_list, axis=-1)
    if array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)

    array = array.astype("uint8")
    return array


def convert_to_16bit(array: np.ndarray) -> np.ndarray[np.uint16]:
    # array is expected to be in Hounsfield units
    # i.e., RescaleSlope and RescaleIntercept already applied
    # traditional 12-bit range was -1024 to 3071
    array = np.clip(array, -1024, 3071)
    array += 1024
    # now values range from 0 to 4095 (12-bit)
    array = array.astype("uint16")  # return as 16-bit, 12-bit is not an option
    return array


def is_valid_dicom(
    ds: pydicom.FileDataset,
    required_attributes: list[str] = [
        "pixel_array",
        "RescaleSlope",
        "RescaleIntercept",
        "ImagePositionPatient",
        "ImageOrientationPatient",
    ],
) -> bool:
    attributes_present = [hasattr(ds, attr) for attr in required_attributes]
    return all(attributes_present)


def center_crop_or_pad_borders(
    image: np.ndarray, size: tuple[int, int], pad_val: int = 0
) -> np.ndarray:
    height, width = image.shape[:2]
    new_height, new_width = size
    if new_height < height:
        # crop top and bottom
        crop_top = (height - new_height) // 2
        crop_bottom = height - new_height - crop_top
        image = image[crop_top:-crop_bottom]
    elif new_height > height:
        # pad top and bottom
        pad_top = (new_height - height) // 2
        pad_bottom = new_height - height - pad_top
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (0, 0)),
            mode="constant",
            constant_values=pad_val,
        )

    if new_width < width:
        # crop left and right
        crop_left = (width - new_width) // 2
        crop_right = width - new_width - crop_left
        image = image[:, crop_left:-crop_right]
    elif new_width > width:
        # pad left and right
        pad_left = (new_width - width) // 2
        pad_right = new_width - width - pad_left
        image = np.pad(
            image,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_val,
        )

    return image


def most_common_element(lst: list):
    return max(set(lst), key=lst.count)


def determine_dicom_orientation(ds: pydicom.FileDataset) -> int:
    iop = ds.ImageOrientationPatient

    # Calculate the direction cosine for the normal vector of the plane
    normal_vector = np.cross(iop[:3], iop[3:])

    # Determine the plane based on the largest component of the normal vector
    abs_normal = np.abs(normal_vector)
    if abs_normal[0] > abs_normal[1] and abs_normal[0] > abs_normal[2]:
        return 0  # sagittal
    elif abs_normal[1] > abs_normal[0] and abs_normal[1] > abs_normal[2]:
        return 1  # coronal
    else:
        return 2  # axial


def load_image_from_dicom(path: str, to_8bit: bool = True) -> np.ndarray:
    dicom = pydicom.dcmread(path)
    arr = apply_voi_lut(dicom.pixel_array, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        # invert image if needed
        arr = arr.max() - arr

    if to_8bit:
        arr = arr - arr.min()
        arr = arr / arr.max()
        arr = (arr * 255).astype("uint8")

    return arr


def load_ct_image_from_dicom(
    path: str,
    dtype: str | None = None,
    windows: tuple[int, int] | list[tuple[int, int]] | None = None,
) -> np.ndarray:
    if dtype is not None:
        assert isinstance(dtype, str)
        assert dtype in ["uint8", "uint16"]
        if dtype == "uint8":
            assert windows is not None
    dicom = pydicom.dcmread(path)
    array = dicom.pixel_array.astype("float32")
    m, b = float(dicom.RescaleSlope), float(dicom.RescaleIntercept)
    array = array * m + b
    if dtype is None:
        # return raw Hounsfield units
        return array
    elif dtype == "uint16":
        array = convert_to_16bit(array)
        return array
    elif dtype == "uint8":
        array = convert_to_8bit(array, windows)
    return array


def load_stack_from_dicom_folder(
    path: str,
    dtype: str | None = None,
    windows: tuple[int, int] | list[tuple[int, int]] | None = None,
    dicom_extension: str = ".dcm",
    sort_by_instance_number: bool = False,  # default sort by ImagePositionPatient
    exclude_invalid_dicoms: bool = False,  # will likely throw error if invalid dicom present
    fix_unequal_shapes: str = "crop_pad",
) -> dict:
    if dtype is not None:
        assert isinstance(dtype, str)
        assert dtype in ["uint8", "uint16"]
        if dtype == "uint8":
            assert windows is not None

    dicom_files = glob.glob(os.path.join(path, f"*{dicom_extension}"))
    if len(dicom_files) == 0:
        raise Exception(
            f"No DICOM files found in `{path}` using `dicom_extension={dicom_extension}`"
        )

    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    required_attributes = [
        "pixel_array",
        "RescaleSlope",
        "RescaleIntercept",
        "ImagePositionPatient",
        "ImageOrientationPatient",
    ]
    if sort_by_instance_number:
        required_attributes.append("InstanceNumber")
    if exclude_invalid_dicoms:
        is_valid = [
            is_valid_dicom(d, required_attributes=required_attributes) for d in dicoms
        ]
        dicoms = [d for d, i in zip(dicoms, is_valid) if i]
        dicom_files = [f for f, i in zip(dicom_files, is_valid) if i]

    slices = [dcm.pixel_array.astype("float32") for dcm in dicoms]
    shapes = np.stack([s.shape for s in slices], axis=0)
    if not np.all(shapes == shapes[0]):
        unique_shapes, counts = np.unique(shapes, axis=0, return_counts=True)
        standard_shape = tuple(unique_shapes[np.argmax(counts)])
        print(
            f"warning: different array shapes present, using {fix_unequal_shapes} -> {standard_shape}"
        )
        if fix_unequal_shapes == "crop_pad":
            min_val = min([s.min() for s in slices])
            slices = [
                center_crop_or_pad_borders(s, standard_shape, pad_val=min_val)
                if s.shape != standard_shape
                else s
                for s in slices
            ]
        elif fix_unequal_shapes == "resize":
            slices = [
                cv2.resize(
                    s, (standard_shape[1], standard_shape[0])
                )  # cv2.resize expects (width, height)
                if s.shape != standard_shape
                else s
                for s in slices
            ]
    slices = np.stack(slices, axis=0)

    if sort_by_instance_number:
        positions = [float(d.InstanceNumber) for d in dicoms]
    else:
        # sort using ImagePositionPatient
        # orientation is index to use for sorting
        orientation = [determine_dicom_orientation(dcm) for dcm in dicoms]
        # use most common
        orientation = most_common_element(orientation)
        positions = [float(d.ImagePositionPatient[orientation]) for d in dicoms]
    indices = np.argsort(positions)
    slices = slices[indices]
    sorted_dicom_files = [dicom_files[idx] for idx in indices]

    # rescale into Hounsfield units
    m, b = (
        [float(d.RescaleSlope) for d in dicoms],
        [float(d.RescaleIntercept) for d in dicoms],
    )
    m, b = most_common_element(m), most_common_element(b)
    slices = slices * m + b

    output = {}

    if dtype is None:
        # return raw Hounsfield units
        output["image"] = slices
    elif dtype == "uint16":
        output["image"] = convert_to_16bit(slices)
        return slices
    elif dtype == "uint8":
        output["image"] = convert_to_8bit(slices, windows)

    output["dicom_files"] = sorted_dicom_files

    return slices
