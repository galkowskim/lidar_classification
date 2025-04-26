from pyproj import Transformer

import os
from collections import defaultdict

import re
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from tqdm import tqdm
import laspy
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from src.data_preparation.constants import DATA_FOLDER
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_las_fragments(plot_numbers: list[str], gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Processes a list of .las file fragments, transforms their coordinates,
    and associates each point with the nearest coastal type.

    Parameters:
    - fragments (list of str): List of .las file prefixes (without extension).
    - data_path (Path): Path object pointing to the directory containing .las files.
    - gdf (GeoDataFrame): GeoDataFrame containing 'coasttype' and 'geometry' columns.
    - transform_to_lonlat (function): Function to transform coordinates to longitude and latitude.

    Returns:
    - final_df (DataFrame): DataFrame with columns ['x', 'y', 'z', 'coasttype'].
    """
    final_df = pd.DataFrame(columns=["x", "y", "z", "coasttype"])

    for prefix in tqdm(plot_numbers, desc="Processing .las fragments"):
        las_file_path = DATA_FOLDER / "las" / f"{prefix}.las"
        if not las_file_path.exists():
            print(f"File {las_file_path} does not exist. Skipping.")
            continue

        las = laspy.read(las_file_path)
        dataset = np.vstack([las.x, las.y, las.z]).T

        transformed_dataset = transform_to_lonlat(dataset)
        lidar_df = pd.DataFrame(transformed_dataset, columns=["x", "y", "z"])

        lidar_df["geometry"] = [Point(xy) for xy in zip(lidar_df["x"], lidar_df["y"])]
        lidar_gdf = gpd.GeoDataFrame(lidar_df, geometry="geometry", crs=gdf.crs)

        if lidar_gdf.crs != gdf.crs:
            lidar_gdf = lidar_gdf.to_crs(gdf.crs)

        # Perform spatial join to find the nearest coastal type
        merged_gdf = gpd.sjoin_nearest(
            lidar_gdf,
            gdf[["coasttype", "geometry"]],
            how="left",
            distance_col="dist_to_line",
        )

        final_df = pd.concat([final_df, merged_gdf[["x", "y", "z", "coasttype"]]], ignore_index=True)

    return final_df


def check_las_files(plot_numbers: list[str]) -> dict:
    """
    Checks for the presence of .las files corresponding to each fragment in the specified directory.

    Parameters:
    - fragments (list): A list of fragment prefixes to check.
    - las_dir (str): Path to the directory containing .las files.

    Returns:
    - dict: A dictionary with counts of 'in' (found) and 'not_in' (not found) fragments.
    """
    logger.info("Checking if all .las files are present.")
    logger.info("Plot numbers: %s", plot_numbers)

    in_no = defaultdict(int)

    for prefix in plot_numbers:
        if any(file.startswith(prefix) and file.endswith(".las") for file in os.listdir(DATA_FOLDER / "las")):
            in_no["in"] += 1
        else:
            in_no["not_in"] += 1

    assert len(plot_numbers) == in_no["in"], "Not all fragments are in the data"
    return in_no


def transform_to_lonlat(dataset: np.array) -> np.array:
    """
    Convert coordinates from .las file to longitude and latitude.

    Args:
        dataset (np.array): Array of coordinates.

    Returns:
        np.array: Array of longitude and latitude coordinates.
    """

    source = "EPSG:2180"  # PL-1992 is the projection used in the dataset
    dest = "EPSG:4326"  # WGS84, the standard for GPS coordinates used all over the world

    transformer = Transformer.from_crs(source, dest, always_xy=True)

    x = dataset[:, 0]
    y = dataset[:, 1]
    z = dataset[:, 2]

    lon, lat = transformer.transform(x, y)

    transformed_coordinates = np.vstack((lon, lat, z)).T
    return transformed_coordinates


def sanitize_coastal_type(coast_type: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", coast_type)


def pointcloud_to_image(df: pd.DataFrame, bins: int = 128) -> np.ndarray:
    """Convert LiDAR point cloud to 2D grayscale image (elevation map)."""
    x = df["x"].values
    y = df["y"].values
    z = df["z"].values

    img, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=z)
    count, _, _ = np.histogram2d(x, y, bins=bins)
    img = img / (count + 1e-6)  # avoid division by zero

    img = np.nan_to_num(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)  # normalize [0, 1]
    img = (img * 255).astype(np.uint8)  # to uint8

    return img


def generate_tile_image(
    df: pd.DataFrame, x_range: tuple[float], y_range: tuple[float], bins: int = 128
) -> np.ndarray | None:
    """Select points within tile and generate image."""
    x_mask = (df["x"] >= x_range[0]) & (df["x"] <= x_range[1])
    y_mask = (df["y"] >= y_range[0]) & (df["y"] <= y_range[1])
    patch = df[x_mask & y_mask]
    if patch.empty:
        return None
    return pointcloud_to_image(patch, bins=bins)


def generate_labeled_tiles(
    final_df: pd.DataFrame, gdf: gpd.GeoDataFrame, tile_size: float = 0.001, crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Generates a grid of tiles over the extent of final_df and assigns coastal types
    based on intersection with geometries in gdf.

    Parameters:
    - final_df (DataFrame): DataFrame containing 'x' and 'y' columns representing coordinates.
    - gdf (GeoDataFrame): GeoDataFrame with 'geometry' and 'coasttype' columns.
    - tile_size (float): Size of each tile in degrees. Default is 0.001 (~100m at Polish latitude).
    - crs (str): Coordinate Reference System for the tiles. Default is "EPSG:4326".

    Returns:
    - labeled_tiles (GeoDataFrame): GeoDataFrame of tiles with assigned coastal types.
    """
    x_min, x_max = final_df["x"].min(), final_df["x"].max()
    y_min, y_max = final_df["y"].min(), final_df["y"].max()

    x_coords = np.arange(x_min, x_max, tile_size)
    y_coords = np.arange(y_min, y_max, tile_size)

    tile_geoms = []
    for x0 in x_coords:
        for y0 in y_coords:
            x1, y1 = x0 + tile_size, y0 + tile_size
            tile_geom = box(x0, y0, x1, y1)
            tile_geoms.append(tile_geom)

    tile_gdf = gpd.GeoDataFrame(geometry=tile_geoms, crs=crs)

    joined = gpd.sjoin(tile_gdf, gdf[["geometry", "coasttype"]], how="left", predicate="intersects")

    labeled_tiles = joined.dropna(subset=["coasttype"])

    return labeled_tiles


def save_labeled_tile_images(labeled_tiles: gpd.GeoDataFrame, df: pd.DataFrame, folder_name: str) -> None:
    """
    Generates and saves grayscale images for each labeled tile.

    Parameters:
    - labeled_tiles (GeoDataFrame): GeoDataFrame containing tile geometries and associated 'coasttype'.
    - df (DataFrame): DataFrame containing point cloud data with 'x', 'y', 'z' columns.
    - output_root (str): Root directory where images will be saved.
    - generate_tile_image_func (function): Function to generate image from point cloud data.
    - sanitize_func (function): Function to sanitize the 'coasttype' string for directory naming.
    """

    os.makedirs(DATA_FOLDER / "torch" / folder_name, exist_ok=True)

    for idx, row in labeled_tiles.iterrows():
        x_min, y_min, x_max, y_max = row.geometry.bounds
        x_range = (x_min, x_max)
        y_range = (y_min, y_max)
        coasttype = row["coasttype"].replace(" ", "_").lower()
        coasttype_clean = sanitize_coastal_type(coasttype)

        class_dir = os.path.join(DATA_FOLDER / "torch" / folder_name, coasttype_clean)
        os.makedirs(class_dir, exist_ok=True)

        img = generate_tile_image(df, x_range, y_range)
        if img is not None:
            save_path = os.path.join(class_dir, f"{idx}.png")
            plt.imsave(save_path, img, cmap="gray")


def plot_lidar_by_coasttype(
    df: pd.DataFrame,
    sample_frac: float = 0.0005,
    folder_name: str = None,
    figsize: tuple[int] = (12, 8),
    dpi: int = 300,
):
    """
    Plots LiDAR points colored by coastal type.

    Parameters:
    - df (DataFrame): DataFrame containing 'x', 'y', and 'coasttype' columns.
    - sample_frac (float): Fraction of points to sample for plotting to reduce overplotting. Default is 0.0005.
    - save_path (str, optional): File path to save the plot. If None, the plot is displayed.
    - figsize (tuple): Size of the figure in inches. Default is (12, 8).
    - dpi (int): Resolution of the saved figure in dots per inch. Default is 300.
    """
    plt.figure(figsize=figsize)
    for group in df["coasttype"].dropna().unique():
        data = df.loc[df["coasttype"] == group].sample(frac=sample_frac)
        plt.scatter(data["x"], data["y"], s=1, label=group, alpha=0.6)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("LiDAR Points Colored by Coastal Type")
    plt.legend(markerscale=5, fontsize="small")
    plt.grid(True)
    plt.tight_layout()

    if folder_name:
        save_path = DATA_FOLDER / "plots" / folder_name
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path / "lidar_by_coast_type.png", dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_lidar_tiles(
    labeled_tiles: gpd.GeoDataFrame,
    df: pd.DataFrame,
    gdf: gpd.geodataframe,
    folder_name: str = None,
    figsize: tuple[int] = (16, 12),
    tile_alpha: float = 0.8,
    boundary_alpha: float = 0.7,
    dpi: int = 300,
):
    """
    Plots LiDAR raster tiles georeferenced on a coordinate system.

    Parameters:
    - labeled_tiles (GeoDataFrame): Tiles with associated coastal types.
    - df (DataFrame): LiDAR points dataframe with x, y, z, and coasttype.
    - gdf (GeoDataFrame): Original coastlines GeoDataFrame for overlay.
    - generate_tile_image_func (function): Function to generate an image from a tile.
    - save_path (str, optional): Path to save the figure. If None, the plot is displayed.
    - figsize (tuple): Size of the figure.
    - tile_alpha (float): Transparency of tile images.
    - boundary_alpha (float): Transparency of coastline boundaries.
    - dpi (int): Resolution of the saved figure.
    """
    tiles_to_plot = []

    for idx, row in tqdm(labeled_tiles.iterrows(), total=len(labeled_tiles), desc="Generating tiles"):
        x0, y0, x1, y1 = row.geometry.bounds
        img = generate_tile_image(df, (x0, x1), (y0, y1))
        if img is not None:
            tiles_to_plot.append((img, (x0, x1, y0, y1)))

    # Compute global bounds
    all_x0 = [extent[0] for _, extent in tiles_to_plot]
    all_x1 = [extent[1] for _, extent in tiles_to_plot]
    all_y0 = [extent[2] for _, extent in tiles_to_plot]
    all_y1 = [extent[3] for _, extent in tiles_to_plot]

    min_x, max_x = min(all_x0), max(all_x1)
    min_y, max_y = min(all_y0), max(all_y1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    for img, (x0, x1, y0, y1) in tqdm(tiles_to_plot, desc="Plotting tiles"):
        ax.imshow(img, extent=(x0, x1, y0, y1), origin="lower", cmap="gray", alpha=tile_alpha)

    gdf.plot(ax=ax, edgecolor="red", linewidth=1, facecolor="none", alpha=boundary_alpha)

    ax.set_xlabel("Longitude (x)")
    ax.set_ylabel("Latitude (y)")
    ax.set_title("LiDAR Raster Tiles Georeferenced on Coordinate System")
    ax.grid(True)
    plt.tight_layout()

    if folder_name:
        save_folder = DATA_FOLDER / "plots" / folder_name
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(save_folder / "lidar_tile_map.png", dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_aggregated_las_fragments_and_update_readme(
    df: pd.DataFrame, folder_name: str, plot_numbers: list[str], shapefile_path: str
) -> None:
    """
    Save the aggregated .las fragments and update the README file.

    Parameters:
    - df (DataFrame): DataFrame containing 'x', 'y', 'z', and 'coasttype' columns.
    - folder_name (str): Folder name for saving the processed data.
    """
    output_dir = DATA_FOLDER / "processed" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{folder_name}.csv"
    df.to_csv(csv_path, index=False)

    readme_path = DATA_FOLDER / "processed" / "README.md"
    if not readme_path.exists():
        readme_path.touch()

    with readme_path.open("a") as f:
        f.write(f"{folder_name}: {plot_numbers}, {shapefile_path}\n")
