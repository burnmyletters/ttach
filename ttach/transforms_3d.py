from functools import partial
from typing import Optional, List, Union, Tuple
from . import functional_3d as F
from .base import DualTransform, ImageOnlyTransform


class HorizontalFlip(DualTransform):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = F.hflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = F.hflip(mask)
        return mask

    def apply_deaug_label(self, label, apply=False, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, apply=False, **kwargs):
        return keypoints


class VerticalFlip(DualTransform):
    """Flip images vertically (up->down)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = F.vflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = F.vflip(mask)
        return mask

    def apply_deaug_label(self, label, apply=False, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, apply=False, **kwargs):
        return keypoints


class ChannelFlip(DualTransform):
    """Flip images channel-wise (up->down)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = F.cflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = F.cflip(mask)
        return mask

    def apply_deaug_label(self, label, apply=False, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, apply=False, **kwargs):
        return keypoints


class Rotate90(DualTransform):
    """Rotate images 0/90/180/270 degrees

    Args:
        angles (list): angles to rotate images
    """

    identity_param = 0

    def __init__(self, angles: List[int]):
        if self.identity_param not in angles:
            angles = [self.identity_param] + list(angles)

        super().__init__("angle", angles)

    def apply_aug_image(self, image, angle=0, **kwargs):
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        return F.rot90(image, k)

    def apply_deaug_mask(self, mask, angle=0, **kwargs):
        return self.apply_aug_image(mask, -angle)

    def apply_deaug_label(self, label, angle=0, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, angle=0, **kwargs):
        return keypoints


class Rotate90ch(DualTransform):
    """Rotate images 0/90/180/270 degrees

    Args:
        angles (list): angles to rotate images
    """

    identity_param = 0

    def __init__(self, angles: List[int]):
        if self.identity_param not in angles:
            angles = [self.identity_param] + list(angles)

        super().__init__("angle", angles)

    def apply_aug_image(self, image, angle=0, **kwargs):
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        return F.rot90ch(image, k)

    def apply_deaug_mask(self, mask, angle=0, **kwargs):
        return self.apply_aug_image(mask, -angle)

    def apply_deaug_label(self, label, angle=0, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, angle=0, **kwargs):
        return keypoints


class Rotate90hv(DualTransform):
    """Rotate images 0/90/180/270 degrees

    Args:
        angles (list): angles to rotate images
    """

    identity_param = 0

    def __init__(self, angles: List[int]):
        if self.identity_param not in angles:
            angles = [self.identity_param] + list(angles)

        super().__init__("angle", angles)

    def apply_aug_image(self, image, angle=0, **kwargs):
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        return F.rot90hv(image, k)

    def apply_deaug_mask(self, mask, angle=0, **kwargs):
        return self.apply_aug_image(mask, -angle)

    def apply_deaug_label(self, label, angle=0, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, angle=0, **kwargs):
        return keypoints


class Rotate90cv(DualTransform):
    """Rotate images 0/90/180/270 degrees

    Args:
        angles (list): angles to rotate images
    """

    identity_param = 0

    def __init__(self, angles: List[int]):
        if self.identity_param not in angles:
            angles = [self.identity_param] + list(angles)

        super().__init__("angle", angles)

    def apply_aug_image(self, image, angle=0, **kwargs):
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        return F.rot90cv(image, k)

    def apply_deaug_mask(self, mask, angle=0, **kwargs):
        return self.apply_aug_image(mask, -angle)

    def apply_deaug_label(self, label, angle=0, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, angle=0, **kwargs):
        return keypoints
