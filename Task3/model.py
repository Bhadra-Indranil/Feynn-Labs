import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ================================================================
# 1. Function to Read and Extract Data from a Satellite Image
# ================================================================
def read_geospatial_image(file_path):
    """
    Opens a satellite raster image and extracts its data and metadata.
    :param file_path: Path to the .tif geospatial image file.
    :return: Image data array and metadata profile.
    """
    with rasterio.open(file_path) as dataset:
        image_data = dataset.read()  # Extract pixel values
        image_metadata = dataset.profile  # Get metadata (projection, resolution, etc.)
    return image_data, image_metadata

# ================================================================
# 2. Function to Compute NDVI (Normalized Difference Vegetation Index)
# ================================================================
def calculate_ndvi(nir_band, red_band):
    """
    Computes the NDVI value from NIR and Red bands.
    NDVI = (NIR - Red) / (NIR + Red)
    :param nir_band: Near-infrared band (usually band 4 or 5)
    :param red_band: Red spectrum band (usually band 3 or 4)
    :return: NDVI array with values ranging from -1 to 1
    """
    # Adding a tiny value to denominator to avoid division by zero
    ndvi_result = (nir_band - red_band) / (nir_band + red_band + 1e-10)
    return ndvi_result

# ================================================================
# 3. Function to Prepare Data for Model Training
# ================================================================
def structure_training_dataset(image_data, ndvi_layer, ground_truth_labels):
    """
    Structures the feature dataset for ML training.
    :param image_data: Satellite image array with multiple bands
    :param ndvi_layer: Computed NDVI layer from the image
    :param ground_truth_labels: Labels (classification categories)
    :return: Feature matrix (X) and corresponding labels (y)
    """
    features_matrix = np.column_stack([image_data[3].ravel(), image_data[4].ravel(), ndvi_layer.ravel()])
    labels_vector = ground_truth_labels.ravel()  # Flatten label array
    return features_matrix, labels_vector

# ================================================================
# 4. Load and Display the Geospatial Image
# ================================================================
image_file_path = "path_to_satellite_image.tif"  # Modify with actual file path
image_data, image_info = read_geospatial_image(image_file_path)

# Display the first band of the loaded satellite image
show(image_data[0], title="Loaded Satellite Image")

# ================================================================
# 5. Compute and Visualize NDVI
# ================================================================
red_band_data = image_data[3, :, :]  # Extract Red Band
nir_band_data = image_data[4, :, :]  # Extract Near-Infrared (NIR) Band

ndvi_index = calculate_ndvi(nir_band_data, red_band_data)

# Display the NDVI map
plt.figure(figsize=(6, 6))
plt.imshow(ndvi_index, cmap='RdYlGn')
plt.colorbar()
plt.title("NDVI (Vegetation Index) Map")
plt.show()

# ================================================================
# 6. Load Ground Truth Labels
# ================================================================
label_file_path = "path_to_ground_truth_labels.tif"  # Modify with actual file path
with rasterio.open(label_file_path) as label_dataset:
    ground_truth = label_dataset.read(1)  # Read single-channel classification labels

# ================================================================
# 7. Prepare Data for Model Training
# ================================================================
X_features, y_labels = structure_training_dataset(image_data, ndvi_index, ground_truth)

# Split dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.3, random_state=42)

# ================================================================
# 8. Train the Machine Learning Model (Random Forest)
# ================================================================
# Initialize the Random Forest Classifier with 100 trees
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)  # Train the model

# ================================================================
# 9. Model Evaluation
# ================================================================
y_test_predictions = rf_classifier.predict(X_test)

# Compute Accuracy Score
model_accuracy = accuracy_score(y_test, y_test_predictions)
print(f"🔹 Model Accuracy: {model_accuracy * 100:.2f}%")

# Compute and Display Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_predictions)
print("📌 Confusion Matrix:")
print(conf_matrix)

# ================================================================
# 10. Generate Full Classification Map
# ================================================================
predicted_map = rf_classifier.predict(X_features).reshape(red_band_data.shape)

# Display the classified map
plt.figure(figsize=(6, 6))
plt.imshow(predicted_map, cmap="jet")
plt.colorbar()
plt.title("Predicted Land Cover Classification")
plt.show()
