import arvet.util.unreal_transform as uetf


def get_MH_01_easy():
    return [
        ('AIUE_V01_002', uetf.create_serialized((2240, 415, 75), (0, 0, 160))),
        ('AIUE_V01_004', uetf.create_serialized((485, -45, 145), (0, 0, 145))),
        ('AIUE_V01_005', uetf.create_serialized((-925, -1135, 120), (0, 0, 0))),
    ]


def get_MH_02_easy():
    return [
        ('AIUE_V01_002', uetf.create_serialized((2240, 415, 75), (0, 0, 156))),
        ('AIUE_V01_004', uetf.create_serialized((225, 0, 145), (0, 0, 140))),
        ('AIUE_V01_005', uetf.create_serialized((770, 180, 60), (0, 0, 180))),
    ]


def get_MH_03_medium():
    return [
        ('AIUE_V01_005', uetf.create_serialized((-490, -450, 275), (0, 0, -90))),
    ]


def get_MH_04_difficult():
    return []


def get_MH_05_difficult():
    return [
        ('AIUE_V01_005', uetf.create_serialized((210, -770, -60), (0, 0, 140))),
    ]


def get_V1_01_easy():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-300, 400, 170), (0, 0, -95))),
        ('AIUE_V01_001', uetf.create_serialized((-260, -490, 170), (0, 0, 0))),
        ('AIUE_V01_002', uetf.create_serialized((-765, 95, 280), (0, 0, 135))),
        ('AIUE_V01_002', uetf.create_serialized((-40, -175, 135), (0, 0, -75))),
        ('AIUE_V01_003', uetf.create_serialized((-115, 520, 135), (0, 0, 155))),
        ('AIUE_V01_004', uetf.create_serialized((385, 0, 425), (0, 0, -105))),
        ('AIUE_V01_005', uetf.create_serialized((-550, -85, 125), (0, 0, 135))),
    ]


def get_V1_02_medium():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-90, 495, 115), (0, 0, 90))),
        ('AIUE_V01_001', uetf.create_serialized((-310, -460, 115), (0, 0, 10))),
        ('AIUE_V01_002', uetf.create_serialized((207.5, 310, 95), (0, 0, -150))),
        ('AIUE_V01_002', uetf.create_serialized((-7.5, -190, 105), (0, 0, -75))),
        ('AIUE_V01_003', uetf.create_serialized((-115, 520, 135), (0, 0, 155))),
        ('AIUE_V01_004', uetf.create_serialized((405, 10, 105), (0, 0, -90))),
        ('AIUE_V01_005', uetf.create_serialized((-560, -1305, 480), (0, 0, -175))),
    ]


def get_V1_03_difficult():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-520, -120, 115), (0, 0, 180))),
        ('AIUE_V01_002', uetf.create_serialized((-522.5, 60, 260), (0, 0, -135))),
        ('AIUE_V01_003', uetf.create_serialized((-235, 255, 135), (0, 0, 155))),
        ('AIUE_V01_004', uetf.create_serialized((-230, -135, 435), (0, 0, 0))),
        ('AIUE_V01_005', uetf.create_serialized((-70, -495, 120), (0, 0, 70))),
    ]


def get_V2_01_easy():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-185, 400, 160), (0, 0, 0))),
        ('AIUE_V01_002', uetf.create_serialized((367.5, -95, 120), (0, 0, -105))),
        ('AIUE_V01_003', uetf.create_serialized((-345, 560, 135), (0, 0, -70))),
        ('AIUE_V01_004', uetf.create_serialized((-350, -35, 205), (0, 0, 115))),
        ('AIUE_V01_005', uetf.create_serialized((-490, -450, 100), (0, 0, -90))),
    ]


def get_V2_02_medium():
    return [
        ('AIUE_V01_002', uetf.create_serialized((-355, -10, 205), (0, 0, -170))),
        ('AIUE_V01_003', uetf.create_serialized((-345, 560, 170), (0, 0, -90))),
        ('AIUE_V01_004', uetf.create_serialized((405, 10, 105), (0, 0, -90))),
        ('AIUE_V01_005', uetf.create_serialized((-540, -290, 195), (0, 0, 150))),
    ]


def get_V2_03_difficult():
    return [
        ('AIUE_V01_002', uetf.create_serialized((-355, -10, 205), (0, 0, -170))),
        ('AIUE_V01_003', uetf.create_serialized((-345, 560, 160), (0, 0, -110))),
        ('AIUE_V01_004', uetf.create_serialized((-695, -30, 430), (0, 0, -5))),
        ('AIUE_V01_005', uetf.create_serialized((-540, -290, 195), (0, 0, 150))),
    ]
