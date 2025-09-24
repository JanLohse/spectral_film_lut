FILE_FORMATS = {
    # --- Apple ProRes ---
    "ProRes Proxy": {
        "extension": ".mov",
        "kwargs": {"codec": "prores_ks", "macro_block_size": 1, "output_params": ["-profile:v", "proxy"]},
    },
    "ProRes LT": {
        "extension": ".mov",
        "kwargs": {"codec": "prores_ks", "macro_block_size": 1, "output_params": ["-profile:v", "lt"]},
    },
    "ProRes Standard": {
        "extension": ".mov",
        "kwargs": {"codec": "prores_ks", "macro_block_size": 1, "output_params": ["-profile:v", "standard"]},
    },
    "ProRes HQ": {
        "extension": ".mov",
        "kwargs": {"codec": "prores_ks", "macro_block_size": 1, "output_params": ["-profile:v", "hq"]},
    },
    "ProRes 4444": {
        "extension": ".mov",
        "kwargs": {
            "codec": "prores_ks",
            "macro_block_size": 1,
            "output_params": ["-profile:v", "4444"]
        },
    },

    # --- H.264 / H.265 ---
    "H.264": {
        "extension": ".mp4",
        "kwargs": {
            "codec": "libx264",
            "macro_block_size": 1,
            "output_params": ["-crf", "18", "-preset", "slow"],
        },
    },
    "H.265": {
        "extension": ".mp4",
        "kwargs": {
            "codec": "libx265",
            "macro_block_size": 1,
            "output_params": ["-crf", "18", "-preset", "slow"],
        },
    },

    # --- Image sequences ---
    "PNG Sequence": {
        "extension": ".png",
        "kwargs": {"format": "png"},
    },
    "TIFF Sequence": {
        "extension": ".tif",
        "kwargs": {"format": "tiff"},
    },

    # --- Lossless archival ---
    "FFV1": {
        "extension": ".mkv",
        "kwargs": {"codec": "ffv1", "macro_block_size": 1},
    },
    "Raw AVI": {
        "extension": ".avi",
        "kwargs": {"codec": "rawvideo", "macro_block_size": 1},
    },
}
