import json
from typing import List

from models.stain import Stain


def get_stored_stains(path: str):
    with open(path) as data_file:
        json_data = json.load(data_file)
        return [Stain.from_json(stain_data) for stain_data in json_data]


def store_stains(stain_list: List[Stain], path: str):
    with open(path, 'w') as data_file:
        json.dump([stain.to_json() for stain in stain_list], data_file)
