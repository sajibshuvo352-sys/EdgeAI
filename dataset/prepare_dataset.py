"""
prepare_dataset.py
------------------
Downloads and prepares a standardized 100-image ImageNet validation subset with
exact ground-truth class labels (0-999) and stable image identifiers.
Generates dataset/labels.csv manifest and saves images to dataset/images/.
Also serves as calibration data for INT8 quantization.
"""

import os
import sys
import ssl
import csv
import json
import urllib.request
import logging

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MANIFEST_PATH = os.path.join(DATASET_DIR, "labels.csv")

# 100 diverse, representative ImageNet sample files covering a broad taxonomic spectrum
SAMPLE_FILENAMES = [
    "n01440764_tench.JPEG",
    "n01443537_goldfish.JPEG",
    "n01484850_great_white_shark.JPEG",
    "n01491361_tiger_shark.JPEG",
    "n01494475_hammerhead.JPEG",
    "n01514668_cock.JPEG",
    "n01514859_hen.JPEG",
    "n01518878_ostrich.JPEG",
    "n01531178_goldfinch.JPEG",
    "n01558993_robin.JPEG",
    "n01614925_bald_eagle.JPEG",
    "n01665541_leatherback_turtle.JPEG",
    "n01667114_mud_turtle.JPEG",
    "n01675722_banded_gecko.JPEG",
    "n01682714_American_alligator.JPEG",
    "n01728572_chameleon.JPEG",
    "n01768244_trilobite.JPEG",
    "n01770081_harvestman.JPEG",
    "n01770393_scorpion.JPEG",
    "n01773797_black_widow.JPEG",
    "n01774384_tarantula.JPEG",
    "n01774750_wolf_spider.JPEG",
    "n01806143_peacock.JPEG",
    "n01819313_macaw.JPEG",
    "n01820546_lorikeet.JPEG",
    "n01877812_wallaby.JPEG",
    "n01882714_koala.JPEG",
    "n01910747_jellyfish.JPEG",
    "n01917283_sea_anemone.JPEG",
    "n01924916_flatworm.JPEG",
    "n01930112_nematode.JPEG",
    "n01944390_snail.JPEG",
    "n01945685_slug.JPEG",
    "n01950731_sea_slug.JPEG",
    "n01978455_rock_crab.JPEG",
    "n01984695_lobster.JPEG",
    "n02085620_Chihuahua.JPEG",
    "n02085782_Japanese_spaniel.JPEG",
    "n02085936_Maltese_dog.JPEG",
    "n02086079_Pekinese.JPEG",
    "n02086240_Shih-Tzu.JPEG",
    "n02086646_Blenheim_spaniel.JPEG",
    "n02086910_papillon.JPEG",
    "n02087046_toy_terrier.JPEG",
    "n02087394_Rhodesian_ridgeback.JPEG",
    "n02088364_beagle.JPEG",
    "n02091831_Saluki.JPEG",
    "n02096051_Airedale.JPEG",
    "n02099601_golden_retriever.JPEG",
    "n02099712_Labrador_retriever.JPEG",
    "n02106665_Rottweiler.JPEG",
    "n02110185_Siberian_husky.JPEG",
    "n02119789_kit_fox.JPEG",
    "n02123045_tabby.JPEG",
    "n02123159_tiger_cat.JPEG",
    "n02123394_Persian_cat.JPEG",
    "n02124075_Egyptian_cat.JPEG",
    "n02129165_lion.JPEG",
    "n02129604_tiger.JPEG",
    "n02130308_cheetah.JPEG",
    "n02132136_brown_bear.JPEG",
    "n02133544_American_black_bear.JPEG",
    "n02134084_ice_bear.JPEG",
    "n02281406_American_lobster.JPEG",
    "n02391049_zebra.JPEG",
    "n02410509_bison.JPEG",
    "n02481823_chimpanzee.JPEG",
    "n02504458_African_elephant.JPEG",
    "n02509815_lesser_panda.JPEG",
    "n02510455_giant_panda.JPEG",
    "n02607072_anode.JPEG",
    "n02747177_ashcan.JPEG",
    "n02782093_balloon.JPEG",
    "n02791187_barbell.JPEG",
    "n02804414_bassinet.JPEG",
    "n02814533_beach_wagon.JPEG",
    "n02814833_beacon.JPEG",
    "n02835271_bicycle-built-for-two.JPEG",
    "n02840245_binder.JPEG",
    "n02841315_binoculars.JPEG",
    "n02892201_brass_plaque.JPEG",
    "n02910353_broom.JPEG",
    "n02917067_bulletproof_vest.JPEG",
    "n02942699_camera.JPEG",
    "n02950826_cannon.JPEG",
    "n02951585_canoe.JPEG",
    "n02974003_car_wheel.JPEG",
    "n03028079_cliff_dwelling.JPEG",
    "n03045698_cloak.JPEG",
    "n03100240_convertible.JPEG",
    "n03109150_corkscrew.JPEG",
    "n03126707_crane.JPEG",
    "n03134739_croquet_ball.JPEG",
    "n03146219_cuirass.JPEG",
    "n03160309_dam.JPEG",
    "n03179701_desk.JPEG",
    "n03196217_digital_clock.JPEG",
    "n03201201_dishwasher.JPEG",
    "n03223299_doormat.JPEG",
    "n03272010_electric_fan.JPEG"
]

RAW_BASE_URL = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
CLASS_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

def fetch_class_index():
    ctx = ssl._create_unverified_context()
    logging.info(f"Downloading ImageNet class index from {CLASS_INDEX_URL}...")
    req = urllib.request.Request(CLASS_INDEX_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as resp:
        class_index = json.loads(resp.read().decode("utf-8"))
    
    # Map synset -> (class_idx, class_name)
    synset_to_info = {}
    for idx_str, (synset, class_name) in class_index.items():
        synset_to_info[synset] = (int(idx_str), class_name)
    return synset_to_info

def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    ctx = ssl._create_unverified_context()
    
    synset_to_info = fetch_class_index()
    logging.info(f"Loaded {len(synset_to_info)} ImageNet class mappings.")
    
    manifest_rows = []
    
    logging.info(f"Downloading {len(SAMPLE_FILENAMES)} validation images...")
    for i, fname in enumerate(SAMPLE_FILENAMES):
        synset = fname.split("_")[0]
        if synset not in synset_to_info:
            logging.warning(f"Synset {synset} not found in class index. Skipping.")
            continue
        
        class_id, class_name = synset_to_info[synset]
        img_url = RAW_BASE_URL + fname
        dest_path = os.path.join(IMAGES_DIR, fname)
        rel_path = os.path.join("images", fname).replace("\\", "/")
        
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            try:
                req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, context=ctx, timeout=10) as resp, open(dest_path, "wb") as f:
                    f.write(resp.read())
            except Exception as e:
                logging.error(f"Failed to download {img_url}: {e}")
                continue
        
        manifest_rows.append({
            "image_id": f"img_{i+1:03d}",
            "image_path": rel_path,
            "ground_truth_label": class_id,
            "class_name": class_name,
            "synset_id": synset
        })
        if (i + 1) % 20 == 0:
            logging.info(f"Downloaded {i+1}/{len(SAMPLE_FILENAMES)} images.")
            
    # Write manifest CSV
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_id", "image_path", "ground_truth_label", "class_name", "synset_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    logging.info(f"Dataset preparation complete. Saved {len(manifest_rows)} records to {MANIFEST_PATH}.")

if __name__ == "__main__":
    main()
