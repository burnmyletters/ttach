from .base import Compose
from . import transforms_3d as tta3d
from . import transforms as tta


def flip_transform3d():
    return Compose([tta3d.HorizontalFlip(), tta3d.VerticalFlip(), tta3d.ChannelFlip()])


def hflip_transform3d():
    return Compose([tta3d.HorizontalFlip()])


def vflip_transform3d():
    return Compose([tta3d.VerticalFlip()])


def cflip_transform3d():
    return Compose([tta3d.ChannelFlip()])


def d4_ransform3d():
    return Compose(
        [
            tta3d.HorizontalFlip(),
            tta3d.VerticalFlip(),
            tta3d.ChannelFlip(),
            tta3d.Rotate90ch(angles=[0, 90, 180, 270]),
            tta3d.Rotate90cv(angles=[0, 90, 180, 270]),
            tta3d.Rotate90hv(angles=[0, 90, 180, 270]),
        ]
    )


def flip_transform():
    return Compose([tta.HorizontalFlip(), tta.VerticalFlip()])


def hflip_transform():
    return Compose([tta.HorizontalFlip()])


def vflip_transform():
    return Compose([tta.VerticalFlip()])


def d4_transform():
    return Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )


def multiscale_transform(scales, interpolation="nearest"):
    return Compose([tta.Scale(scales, interpolation=interpolation)])


def five_crop_transform(crop_height, crop_width):
    return Compose([tta.FiveCrops(crop_height, crop_width)])


def ten_crop_transform(crop_height, crop_width):
    return Compose([tta.HorizontalFlip(), tta.FiveCrops(crop_height, crop_width)])
