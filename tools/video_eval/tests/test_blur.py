import json

from PIL import Image, ImageChops, ImageDraw

from video_eval.blur import PersonDetection, blur_image, build_sheets
from video_eval.common import read_jsonl, write_jsonl


class OnePerson:
    def detect(self, _image):
        return [
            PersonDetection(
                bbox=(20, 10, 100, 115),
                nose=(60, 35, 0.9),
                left_shoulder=(42, 55, 0.9),
                right_shoulder=(78, 55, 0.9),
            )
        ]


class InitialFaceOnly:
    def __init__(self):
        self.calls = 0

    def detect(self, _image):
        self.calls += 1
        return [(45, 20, 75, 50)] if self.calls == 1 else []


def test_synthetic_face_is_obscured_and_verified():
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 20, 75, 50), fill="peachpuff", outline="black", width=3)
    draw.ellipse((52, 29, 55, 32), fill="black")
    draw.ellipse((65, 29, 68, 32), fill="black")

    blurred, mode, upload_ok = blur_image(image, people=OnePerson(), faces=InitialFaceOnly())

    assert mode == "head"
    assert upload_ok is True
    assert ImageChops.difference(image, blurred).getbbox() is not None


def test_low_visibility_pose_blurs_whole_person_box():
    class LowVisibility:
        def detect(self, _image):
            return [PersonDetection(bbox=(10, 10, 90, 90))]

    class NoFaces:
        def detect(self, _image):
            return []

    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 90, 90), fill="black")
    ImageDraw.Draw(image).ellipse((35, 25, 65, 55), fill="white")
    blurred, mode, upload_ok = blur_image(image, people=LowVisibility(), faces=NoFaces())
    assert mode == "person"
    assert upload_ok is True
    assert ImageChops.difference(image, blurred).getbbox() is not None


def test_missing_pose_and_local_label_is_not_uploadable():
    class NoPeople:
        def detect(self, _image):
            return []

    class NoFaces:
        def detect(self, _image):
            return []

    _, mode, upload_ok = blur_image(
        Image.new("RGB", (100, 100), "white"), people=NoPeople(), faces=NoFaces()
    )
    assert mode == "unknown"
    assert upload_ok is False


def test_upload_false_frame_never_appears_in_sheet(tmp_path):
    root = tmp_path / "private"
    clip = root / "clips" / "test"
    blurred = clip / "blurred"
    blurred.mkdir(parents=True)
    for index, color in enumerate(("red", "blue")):
        Image.new("RGB", (160, 90), color).save(blurred / f"f_{index:06d}.jpg")
    write_jsonl(
        clip / "blur.jsonl",
        [
            {
                "blur_mode": "head",
                "blurred_path": "clips/test/blurred/f_000000.jpg",
                "frame_index": 0,
                "t_s": 0.0,
                "upload_ok": True,
            },
            {
                "blur_mode": "head",
                "blurred_path": "clips/test/blurred/f_000001.jpg",
                "frame_index": 1,
                "t_s": 0.5,
                "upload_ok": False,
            },
        ],
    )

    result = build_sheets("test", root=root)

    assert result["eligible_frames"] == 1
    manifest = read_jsonl(clip / "sheets.jsonl")
    assert [frame["frame_index"] for sheet in manifest for frame in sheet["frames"]] == [0]
    assert (clip / "sheets" / "s_0001.jpg").exists()
    assert not (clip / "sheets.reviewed.json").exists()
    reviewed = build_sheets("test", root=root, confirm_reviewed=True)
    assert reviewed["human_reviewed"] is True
    assert json.loads((clip / "sheets.reviewed.json").read_text())["reviewed"] is True
