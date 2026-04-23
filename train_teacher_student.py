from unet.student_unet_model import StudentUNet

student_model = StudentUNet(
    n_channels=1,
    n_classes=1,
    base_c=16,
    bilinear=True,
    return_features=True
)