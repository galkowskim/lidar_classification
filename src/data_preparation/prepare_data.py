import click
from src.data_preparation.constants import DATA_FOLDER
import geopandas as gpd

from src.data_preparation.data_preprocessing_utils import (
    check_las_files,
    process_las_fragments,
    generate_labeled_tiles,
    save_labeled_tile_images,
    plot_lidar_by_coasttype,
    plot_lidar_tiles,
    save_aggregated_las_fragments_and_update_readme,
)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_plot_numbers(ctx: object, param: object, value: list[str]) -> list[str]:
    if not all(isinstance(x, str) for x in value):
        raise click.BadParameter("All plot numbers must be strings.")
    return value


@click.command()
@click.option(
    "--plot_numbers",
    required=True,
    type=str,
    help="List of plot numbers to process.",
    callback=validate_plot_numbers,
)
@click.option(
    "--shapefile_path",
    required=True,
    help="""Path to the shapefile from emodnet. Please give path from 
    `emodnet` folder (e.g. 'hel_gdansk_data/coastal_type_20210501_0_80k.shp'
    highest resolution shapefile).""",
)
@click.option(
    "--folder_name",
    required=True,
    help="Folder name which is helping identify the processed data. (e.g. zatoka_gdanska)",
)
@click.option(
    "--save_plots",
    is_flag=True,
    help="Save plots of the data.",
)
def main(
    plot_numbers: str,
    shapefile_path: str,
    folder_name: str,
    save_plots: bool,
) -> None:
    """
    Main function to prepare data for training.

    Parameters:
    - plot_numbers (list of str): List of plot numbers to process.
    """
    plot_numbers = plot_numbers.split(",")

    # Checking if data from Mobafire contains all the plots taken from the geoportal
    logger.info("Checking if all .las files are present.")
    check_las_files(plot_numbers)

    shapefile_path = DATA_FOLDER / "raw/shapefile/emodnet" / shapefile_path
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile {shapefile_path} does not exist.")

    logger.info("Loading shapefile.")
    gdf = gpd.read_file(shapefile_path)

    logger.info("Filtering shapefile by plot numbers.")
    df = process_las_fragments(plot_numbers, gdf)

    logger.info("Generating labeled tiles. This may take a while, depending on the size of the data.")
    labeled_tiles = generate_labeled_tiles(df, gdf)

    logger.info("Saving labeled tile images into the folder: data/torch/%s", folder_name)
    save_labeled_tile_images(labeled_tiles, df, folder_name)

    if save_plots:
        logger.info("Saving plots of the data.")
        plot_lidar_by_coasttype(df, folder_name=folder_name)
        plot_lidar_tiles(labeled_tiles, df, gdf, folder_name=folder_name)

    logger.info("Saving aggregated .las fragments and updating README.")
    save_aggregated_las_fragments_and_update_readme(df, folder_name, plot_numbers, shapefile_path)


if __name__ == "__main__":
    main()
