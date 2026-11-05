MP3D_ID_TO_NAME = [
    "chair",
    "table | dining table | coffee table | desk",  # "table",
    "picture | framed photograph",  # "picture",
    "cabinet",
    "pillow",  # "pillow",
    "couch",  # "sofa",
    "bed",
    "nightstand",  # "chest of drawers",
    "potted plant",  # "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv",  # "tv monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]

HM3D_ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]

# Names used in Habitat task files differ from the COCO/VLM detector labels.
OBJECTNAV_CATEGORY_ALIASES = {
    "table": "table | dining table | coffee table | desk",
    "picture": "picture | framed photograph",
    "sofa": "couch",
    "chest_of_drawers": "nightstand",
    "chest of drawers": "nightstand",
    "plant": "potted plant",
    "tv_monitor": "tv",
    "tv monitor": "tv",
    "gym_equipment": "gym equipment",
}


def detector_category(category: str) -> str:
    return OBJECTNAV_CATEGORY_ALIASES.get(category, category)
