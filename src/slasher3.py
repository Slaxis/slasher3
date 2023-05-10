import os
from slasher3.log import logger
from slasher3.config import local_model_is_trained, slasher_config
from slasher3.xray import XRayDatasets
from slasher3.solvers import Slasher
from slasher3.nets.slashernet import SlasherNet
from slasher3.device import get_device
from slasher3.dorothy import Dorothy

if __name__ == "__main__":
    try:
        database = XRayDatasets(from_local_files=True,
                                local_folders=slasher_config.image_dir,
                                splitter=slasher_config.splitter,
                                training_set=slasher_config.training_set,
                                width=slasher_config.output_width)
        if not local_model_is_trained: # TRAIN AND SAVE MODEL B4 USING
            logger.info(__file__, f"Training started for model {slasher_config.config_name}")

            if slasher_config.local_or_cloud == "local":
                slasher = Slasher(slasher_config.config_name,
                                  slasher_config.override_device,
                                  slasher_config.random_seed,
                                  database,
                                  slasher_config.max_images,
                                  slasher_config.batch_size,
                                  slasher_config.export_path,
                                  slasher_config.epochs,
                                  slasher_config.training_lr,
                                  slasher_config.patience,
                                  slasher_config.output_width)
                knife, knife_stats = slasher.solve()
        else:
            if slasher_config.use_dorothy: # DOWNLOAD DOROTHY
                dorothy = Dorothy()
                dorothy.dataset('imageamento_anonimizado_valid', download=True, download_dir = slasher_config.image_dir.input)
            knife = SlasherNet(slasher_config.config_name,
                                slasher_config.export_path,
                                get_device(slasher_config.override_device),
                                output_width=slasher_config.output_width,
                                net_save=slasher_config.net_save)

        # CUT STUFF
        logger.info(__file__, f"Starting cutting procedure...")
        to_mask = database.sample(kind="unmasked", sample_type="input")
        masks = knife.predict(to_mask) # MASKS DATAFRAME
        export_path = os.path.join(slasher_config.image_dir.output, f"{slasher_config.config_name}.json")
        masks.to_json(export_path)

    except Exception as e:
        logger.error(__file__, e)