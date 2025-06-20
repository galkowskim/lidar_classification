from pathlib import Path

DATA_FOLDER = Path(__file__).parent.parent.parent / "data"

LABEL_MAPPING = {
    "Sand beach fronting upland (> 1 Km long)": 0,
    "Harbor area": 1,
    "Artificial beach": 2,
    "Beach, unspecified": 3,
    "Muddy coastline, including tidal flat, salt marsh": 4,
    "Vegetated (?green?) beach": 5,
    "Artificial shoreline (walk, dike, quay) without beach": 6,
    "Beach that is part of extensive non-cohesive sedimentary systems (barrier, spit, tombolo)": 7,
}

FOLDER_TO_LABEL = {
    "sandbeachfrontingupland1kmlong": "Sand beach fronting upland (> 1 Km long)",
    "harborarea": "Harbor area",
    "artificialbeach": "Artificial beach",
    "beachunspecified": "Beach, unspecified",
    "muddycoastlineincludingtidalflatsaltmarsh": "Muddy coastline, including tidal flat, salt marsh",
    "vegetatedgreenbeach": "Vegetated (?green?) beach",
    "artificialshorelinewalkdikequaywithoutbeach": "Artificial shoreline (walk, dike, quay) without beach",
    "beachthatispartofextensivenoncohesivesedimentarysystemsbarrierspittombolo": "Beach that is part of extensive non-cohesive sedimentary systems (barrier, spit, tombolo)",
}
