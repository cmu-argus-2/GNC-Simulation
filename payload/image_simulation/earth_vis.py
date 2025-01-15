import rasterio
import os
import numpy as np
import matplotlib.pyplot as plt

# Define MGRS latitude bands and UTM exceptions
latitude_bands = [
    {"name": "C", "min_lat": -80, "max_lat": -72},
    {"name": "D", "min_lat": -72, "max_lat": -64},
    {"name": "E", "min_lat": -64, "max_lat": -56},
    {"name": "F", "min_lat": -56, "max_lat": -48},
    {"name": "G", "min_lat": -48, "max_lat": -40},
    {"name": "H", "min_lat": -40, "max_lat": -32},
    {"name": "J", "min_lat": -32, "max_lat": -24},
    {"name": "K", "min_lat": -24, "max_lat": -16},
    {"name": "L", "min_lat": -16, "max_lat": -8},
    {"name": "M", "min_lat": -8, "max_lat": 0},
    {"name": "N", "min_lat": 0, "max_lat": 8},
    {"name": "P", "min_lat": 8, "max_lat": 16},
    {"name": "Q", "min_lat": 16, "max_lat": 24},
    {"name": "R", "min_lat": 24, "max_lat": 32},
    {"name": "S", "min_lat": 32, "max_lat": 40},
    {"name": "T", "min_lat": 40, "max_lat": 48},
    {"name": "U", "min_lat": 48, "max_lat": 56},
    {"name": "V", "min_lat": 56, "max_lat": 64},
    {"name": "W", "min_lat": 64, "max_lat": 72},
    {"name": "X", "min_lat": 72, "max_lat": 84},  # X spans 12° latitude
]

utm_exceptions = [
    {"zone": 32, "min_lon": 3, "max_lon": 12, "bands": ["V"]},  # Norway
    {"zone": 31, "min_lon": 0, "max_lon": 9, "bands": ["X"]},  # Svalbard
    {"zone": 33, "min_lon": 9, "max_lon": 21, "bands": ["X"]},  # Svalbard
    {"zone": 35, "min_lon": 21, "max_lon": 33, "bands": ["X"]},  # Svalbard
    {"zone": 37, "min_lon": 33, "max_lon": 42, "bands": ["X"]},  # Svalbard
]


class EarthImageSimulator:
    def __init__(self, geotiff_folder=None, resolution=None, hfov=None):
        """
        Initialize the Earth image simulator.

        Parameters:
            geotiff_folder (str): Path to the folder containing GeoTIFF files.
            resolution (tuple): Camera resolution (width, height).
            hfov (float): Horizontal field of view in degrees.
        """
        if geotiff_folder is None:
            geotiff_folder = os.path.abspath(os.path.join(__file__, "../../../data/landsat/region_mosaics"))
        if resolution is None:
            resolution = np.array([4608, 2592])  # width, height
        if hfov is None:
            hfov = 66.1
        self.cache = GeoTIFFCache(geotiff_folder)
        self.resolution = resolution
        self.camera = Camera(self.resolution, hfov)

    def simulate_image(self, position, orientation):
        """
        Simulate an Earth image given the satellite position and orientation.

        Parameters:
            position (np.ndarray): Satellite position in ECEF coordinates (3,).
            orientation (np.ndarray): Satellite orientation as a 3x3 rotation matrix.

        Returns:
            np.ndarray: Simulated RGB image.
        """
        # Generate ray directions in ECEF frame
        ray_directions_ecef = self.camera.rays_in_ecef(orientation)

        # Intersect rays with the Earth
        intersection_points = intersect_ellipsoid(ray_directions_ecef, position)

        # Convert intersection points to lat/lon
        lat_lon = convert_to_lat_lon(intersection_points)

        # Flatten latitude/longitude grid
        lat_lon_flat = lat_lon.reshape(-1, 2)
        latitudes = lat_lon_flat[:, 0]
        longitudes = lat_lon_flat[:, 1]

        # Calculate present MGRS regions
        mgrs_regions = calculate_mgrs_zones(latitudes, longitudes)
        present_regions = np.unique([region for region in mgrs_regions if region is not None])

        # Initialize full image with zeros
        width, height = self.resolution
        pixel_colors_full = np.zeros((height, width, 3), dtype=np.uint8)

        # Load and assign data for each region
        for region in present_regions:
            data, trans = self.cache.load_geotiff_data(region)
            if data is None:
                continue

            # Mask for the current region
            region_mask = (mgrs_regions == region).reshape(height, width)

            # Skip if no pixels belong to this region
            if not np.any(region_mask):
                continue

            # Query pixel colors for the region
            pixel_colors_region = query_pixel_colors(
                latitudes[region_mask.flatten()],
                longitudes[region_mask.flatten()],
                data,
                trans
            )

            # Assign pixel values to the full image
            pixel_colors_full[region_mask] = pixel_colors_region

        return pixel_colors_full

    def display_image(self, image):
        """
        Display the simulated image.

        Parameters:
            image (np.ndarray): Simulated RGB image.
        """
        plt.imshow(image)
        plt.axis('off')
        plt.show()


class GeoTIFFCache:
    def __init__(self, geotiff_folder):
        self.geotiff_folder = geotiff_folder
        self.cache = {}

        for region in ["10S", "10T", "11R", "12R", "16T", "17R", "17T", "18S",
                       "32S", "32T", "33S", "33T", "52S", "53S", "54S", "54T"]:
            region_folder = os.path.join(self.geotiff_folder, region, "sentinel_gsd_185")
            if not os.path.exists(region_folder):
                print(f"WARNING: Region folder '{region_folder}' not found.")
                break
        else:
            print("All region folders found!")

    def load_geotiff_data(self, region):
        if region in self.cache:
            return self.cache[region]

        region_folder = os.path.join(self.geotiff_folder, region, "sentinel_gsd_185")
        if not os.path.exists(region_folder):
            self.cache[region] = (None, None)
            return self.cache[region]
        region_files = os.listdir(region_folder)
        if not region_files:
            self.cache[region] = (None, None)
            return self.cache[region]

        selected_file = np.random.choice(region_files)
        file_path = os.path.join(region_folder, selected_file)
        with rasterio.open(file_path) as src:
            data = src.read()
            data = np.moveaxis(data, 0, -1)
            trans = src.transform
        self.cache[region] = (data, trans)
        return self.cache[region]

    def clear_cache(self):
        self.cache = {}


class Camera:
    def __init__(self, resolution, fov):
        """
        Initialize the camera parameters.

        Parameters:
            resolution (tuple): Resolution of the camera (width, height).
            fov (float): Field of view in degrees (assumes square FOV).
        """
        self.resolution = resolution
        self.fov = np.radians(fov)  # Convert FOV to radians

    def ray_directions(self):
        """
        Generate ray directions for the camera.

        Returns:
            np.ndarray: Array of ray directions (HxWx3) in the camera frame.
        """
        width, height = self.resolution
        half_width = np.tan(self.fov / 2)
        half_height = half_width * (height / width)

        x = np.linspace(-half_width, half_width, width)
        y = np.linspace(-half_height, half_height, height)
        xx, yy = np.meshgrid(x, y)
        zz = np.ones_like(xx)  # Assume unit depth

        # Stack and normalize ray directions
        ray_directions = np.stack([xx, yy, zz], axis=-1)
        ray_directions /= np.linalg.norm(ray_directions, axis=-1, keepdims=True)
        return ray_directions

    def rays_in_ecef(self, orientation):
        """
        Transform ray directions from the camera frame to the ECEF frame.

        Parameters:
            orientation (np.ndarray): 3x3 rotation matrix for orientation.

        Returns:
            np.ndarray: Array of ray directions (HxWx3) in the ECEF frame.
        """
        return self.ray_directions() @ orientation.T

    def pixel_to_bearing_unit_vector(self, pixel_coords):
        """
        Converts pixel coordinates to bearing unit vectors in the camera frame.

        Parameters:
            pixel_coords (np.ndarray): An array of shape (N, 2) with pixel coordinates.

        Returns:
            np.ndarray: An array of shape (N, 3) with bearing unit vectors in the camera frame.
        """
        width, height = self.resolution

        half_width = np.tan(self.fov / 2)
        half_height = half_width * (height / width)

        u = pixel_coords[:, 0]  # Pixel x-coordinates
        v = pixel_coords[:, 1]  # Pixel y-coordinates

        # Normalize pixel coordinates to range [-half_width, half_width] and [half_height, -half_height]
        # Assuming pixel (0,0) is at the top-left corner
        x = -half_width + (2 * half_width) * (u / (width - 1))
        y = half_height - (2 * half_height) * (v / (height - 1))  # Invert y-axis for image coordinates
        z = np.ones_like(x)

        # Stack and normalize direction vectors
        bearing_unit_vectors = np.stack([x, y, z], axis=-1)
        bearing_unit_vectors /= np.linalg.norm(bearing_unit_vectors, axis=1, keepdims=True)
        return bearing_unit_vectors


def get_nadir_rotation(satellite_position):
    # pointing nadir in world coordinates
    x, y, z = satellite_position
    zc = dir_vector = -np.array([x, y, z]) / np.linalg.norm([x, y, z])
    axis_of_rotation_z = np.cross(np.array([0, 0, 1]), dir_vector)
    rc = axis_of_rotation_z = axis_of_rotation_z / np.linalg.norm(axis_of_rotation_z)
    xc = -rc

    yc = south_vector = np.cross(rc, zc)
    R = np.stack([xc, yc, zc], axis=-1)
    return R


def intersect_ellipsoid(ray_directions, satellite_position, a=6378137.0, b=6356752.314245):
    """
    Vectorized computation of ray intersections with the WGS84 ellipsoid.

    Parameters:
        ray_directions (np.ndarray): Array of ray directions (Nx3).
        satellite_position (np.ndarray): Satellite position in ECEF (3,).
        a (float): Semi-major axis of the WGS84 ellipsoid (meters).
        b (float): Semi-minor axis of the WGS84 ellipsoid (meters).

    Returns:
        np.ndarray: Intersection points (Nx3), or NaN for rays that miss.
    """
    H, W, _ = ray_directions.shape
    ray_directions_flat = ray_directions.reshape(-1, 3)

    A = ray_directions_flat[:, 0] ** 2 / a ** 2 + ray_directions_flat[:, 1] ** 2 / a ** 2 + ray_directions_flat[:,
                                                                                            2] ** 2 / b ** 2
    B = 2 * (satellite_position[0] * ray_directions_flat[:, 0] / a ** 2 +
             satellite_position[1] * ray_directions_flat[:, 1] / a ** 2 +
             satellite_position[2] * ray_directions_flat[:, 2] / b ** 2)
    C = (satellite_position[0] ** 2 / a ** 2 +
         satellite_position[1] ** 2 / a ** 2 +
         satellite_position[2] ** 2 / b ** 2 - 1)
    discriminant = B ** 2 - 4 * A * C

    # Initialize intersection points as NaN
    intersection_points_flat = np.full_like(ray_directions_flat, np.nan)

    valid_mask = discriminant >= 0
    if np.any(valid_mask):
        # Compute roots of the quadratic equation
        sqrt_discriminant = np.sqrt(discriminant[valid_mask])
        t1 = (-B[valid_mask] - sqrt_discriminant) / (2 * A[valid_mask])
        t2 = (-B[valid_mask] + sqrt_discriminant) / (2 * A[valid_mask])

        # Choose the smallest positive t
        t = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
        t = np.where(t > 0, t, np.nan)  # Filter out negative t values

        # Calculate intersection points
        valid_ray_directions = ray_directions_flat[valid_mask]
        intersection_points_flat[valid_mask] = t[:, None] * valid_ray_directions + satellite_position
    # Reshape intersection points back to original ray grid shape
    intersection_points = intersection_points_flat.reshape(H, W, 3)
    return intersection_points


def convert_to_lat_lon(intersection_points, a=6378137.0, b=6356752.314245):
    """
    Convert intersection points (ECEF) to latitude and longitude.

    Parameters:
        intersection_points (np.ndarray): Array of intersection points (HxWx3) in ECEF coordinates.

    Returns:
        np.ndarray: Array of latitude and longitude (HxWx2), or NaN for invalid points.
    """

    H, W, _ = intersection_points.shape
    intersection_points_flat = intersection_points.reshape(-1, 3)

    valid_mask = ~np.isnan(intersection_points_flat).any(axis=1)

    lat_lon_flat = np.full((H * W, 2), np.nan)

    valid_points = intersection_points_flat[valid_mask]

    x, y, z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

    # Longitude calculation (same for geodetic and geocentric)
    lon = np.degrees(np.arctan2(y, x))

    # Geodetic latitude calculation (iterative approach)
    e2 = (a ** 2 - b ** 2) / a ** 2  # First eccentricity squared
    ep2 = (a ** 2 - b ** 2) / b ** 2  # Second eccentricity squared
    p = np.sqrt(x ** 2 + y ** 2)

    # Initial approximation of latitude
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3)

    # Convert to degrees
    lat = np.degrees(lat)

    # Store results in flat array
    lat_lon_flat[valid_mask, 0] = lat
    lat_lon_flat[valid_mask, 1] = lon

    return lat_lon_flat.reshape(H, W, 2)


def lat_lon_to_ecef(lat_lon, a=6378137.0, b=6356752.314245):
    """
    Convert latitude and longitude to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    Parameters:
        lat_lon (np.ndarray): Array of latitude and longitude (HxWx2).

    Returns:
        np.ndarray: Array of ECEF coordinates (HxWx3).
    """
    H, W, _ = lat_lon.shape
    lat_lon_flat = lat_lon.reshape(-1, 2)

    lat = lat_lon_flat[:, 0]
    lon = lat_lon_flat[:, 1]

    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # First eccentricity squared
    e2 = (a ** 2 - b ** 2) / a ** 2

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    # Assume height h = 0 (on the ellipsoid)
    x = N * np.cos(lat_rad) * np.cos(lon_rad)
    y = N * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2)) * np.sin(lat_rad)

    ecef_flat = np.column_stack((x, y, z))
    ecef = ecef_flat.reshape(H, W, 3)

    return ecef


def test_geodetic_conversion():
    # convert_to_ecef was ChatGPT generated, it also produced this test

    # Generate a grid of latitude and longitude values
    latitudes = np.linspace(-90, 90, num=10)
    longitudes = np.linspace(-180, 180, num=10)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    H, W = lat_grid.shape
    lat_lon = np.stack((lat_grid, lon_grid), axis=2)  # Shape (H, W, 2)

    # Convert lat/lon to ECEF using the inverse function
    ecef_points = lat_lon_to_ecef(lat_lon)

    # Convert ECEF back to lat/lon using the original function
    lat_lon_reconstructed = convert_to_lat_lon(ecef_points)

    # Compute differences
    lat_diff = lat_lon[:, :, 0] - lat_lon_reconstructed[:, :, 0]
    lon_diff = lat_lon[:, :, 1] - lat_lon_reconstructed[:, :, 1]

    # Print maximum differences
    print("Maximum latitude difference (degrees):", np.max(np.abs(lat_diff)))
    print("Maximum longitude difference (degrees):", np.max(np.abs(lon_diff)))


def calculate_mgrs_zones(latitudes, longitudes):
    """
    Vectorized computation of MGRS regions for given latitude and longitude arrays.

    Parameters:
        latitudes (np.ndarray): 1D or 2D array of latitudes in degrees.
        longitudes (np.ndarray): 1D or 2D array of longitudes in degrees.

    Returns:
        np.ndarray: Array of MGRS region identifiers (same shape as input).
    """
    # Create lookup tables for vectorized latitude band calculation
    latitude_band_names = np.array([band["name"] for band in latitude_bands])
    latitude_band_edges = np.array([[band["min_lat"], band["max_lat"]] for band in latitude_bands])

    # Flatten lat/lon for processing
    lat_flat = latitudes.ravel()
    lon_flat = longitudes.ravel()

    # Determine latitude bands
    lat_bands = np.full(lat_flat.shape, None, dtype=object)
    for i, (min_lat, max_lat) in enumerate(latitude_band_edges):
        mask = (lat_flat >= min_lat) & (lat_flat < max_lat)
        lat_bands[mask] = latitude_band_names[i]

    # Determine UTM zones (default calculation)
    utm_zones = ((lon_flat + 180) // 6 + 1).astype(int)

    # Apply UTM exceptions
    for exception in utm_exceptions:
        mask = (
                (lon_flat >= exception["min_lon"]) &
                (lon_flat < exception["max_lon"]) &
                np.isin(lat_bands, exception["bands"])
        )
        utm_zones[mask] = exception["zone"]

    # Combine UTM zones and latitude bands
    mgrs_regions = np.array([f"{zone}{band}" if band is not None else None
                             for zone, band in zip(utm_zones, lat_bands)])

    # Reshape to match input lat/lon shape
    return mgrs_regions.reshape(latitudes.shape)


def query_pixel_colors(latitudes, longitudes, image_data, trans):
    latitudes_flat = latitudes.flatten()
    longitudes_flat = longitudes.flatten()

    inverse_transform = ~trans

    cols, rows = inverse_transform * (longitudes_flat, latitudes_flat)

    # Round and convert to integers
    cols = np.floor(cols).astype(int)
    rows = np.floor(rows).astype(int)

    # Get image dimensions
    height, width, _ = image_data.shape

    # Create a mask for valid indices
    valid_mask = (
            (rows >= 0) & (rows < height) &
            (cols >= 0) & (cols < width)
    )

    # Prepare an array for the pixel values
    num_pixels = latitudes_flat.size
    num_bands = image_data.shape[-1]
    pixel_values = np.zeros((num_pixels, num_bands), dtype=image_data.dtype)

    # Only retrieve pixel values for valid indices
    if np.any(valid_mask):
        pixel_values[valid_mask] = image_data[rows[valid_mask], cols[valid_mask], :]

    # Handle invalid indices (e.g., set to NaN)
    # pixel_values[~valid_mask] = np.nan  # Uncomment if you prefer NaN for invalid pixels

    # Reshape the output to match the input shape (H x W x bands)
    output_shape = latitudes.shape + (num_bands,)
    pixel_values = pixel_values.reshape(output_shape)

    return pixel_values


def sweep_lat_lon_test():
    simulator = EarthImageSimulator()

    latitudes = np.linspace(-90, 90, 90)
    longitudes = np.linspace(-180, 180, 90)

    lat_lon = np.stack(np.meshgrid(latitudes, longitudes), axis=-1)
    ecef_positions = lat_lon_to_ecef(lat_lon).reshape(-1, 3)

    # scale to 600km altitude
    R_earth = 6371.0088e3
    ecef_positions *= (R_earth + 600e3) / R_earth

    empty_indices = []
    for i, ecef_position in enumerate(ecef_positions):
        orientation = get_nadir_rotation(ecef_position)
        simulated_image = simulator.simulate_image(ecef_position, orientation)

        print(f"{i}/{lat_lon.shape[0] * lat_lon.shape[1]}")
        if np.all(simulated_image == 0):
            empty_indices.append(i)
        else:
            print(f"Nonempty image at index {i}, lat/lon: {lat_lon[0, i, :]}")

    print(f"{len(empty_indices)}/{lat_lon.shape[1]} images are empty")
    print(f"Empty images at indices: {empty_indices}")

    with open("empty_images.txt", "w") as f:
        f.write(str(empty_indices))


def main():
    simulator = EarthImageSimulator()

    satellite_position = np.array([983017.6742974258, -6109867.766065873, 3098940.646932125])
    orientation = get_nadir_rotation(satellite_position)

    simulated_image = simulator.simulate_image(satellite_position, orientation)
    print(np.all(simulated_image == 0))


if __name__ == "__main__":
    test_geodetic_conversion()
    sweep_lat_lon_test()
    main()
