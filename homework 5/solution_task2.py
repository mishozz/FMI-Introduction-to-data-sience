import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

MAJOR_TECTONIC_PLATES = [
    "Pacific", "North America", "South America", "Africa",
    "Eurasia", "India", "Australia", "Antarctica"
]

def init_data(): 
    volcanoes = pd.read_csv("volcano_data2.csv")
    tectonic_plates = gpd.read_file("PB2002_plates.json")
    
    return volcanoes, tectonic_plates

def assign_to_plate(volcanoes, plates):
    plate_names = []
    for volcano in volcanoes.geometry:
        found_plate = "Unknown"
        for _, plate in plates.iterrows():
            if plate['geometry'].contains(volcano):
                found_plate = plate['PlateName']
                break
        plate_names.append(found_plate)

    return plate_names


def plot_volcanoes_onto_plates(volcanoes_df, tectonic_plates, geometry):
    volcanoes_gdf = gpd.GeoDataFrame(volcanoes_df, geometry=geometry)
    volcanoes_gdf['Tectonic_Plate'] = assign_to_plate(volcanoes_gdf, tectonic_plates)
    
    _, ax = plt.subplots(figsize=(15, 8))

    tectonic_plates.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=1, alpha=0.5)
    unique_plates = volcanoes_gdf['Tectonic_Plate'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_plates))

    for i, plate in enumerate(unique_plates):
        plate_volcanoes = volcanoes_gdf[volcanoes_gdf['Tectonic_Plate'] == plate]
        plate_volcanoes.plot(
            ax=ax,
            color=colors(i),
            markersize=20,
            label=plate
        )
         
    plt.title("Volcanoes Clustered on Major Tectonic Plates")
    plt.legend(title="Tectonic Plates", loc='upper left', bbox_to_anchor=(1, 1))
    plt.axis('equal')  # Equal scaling for longitude and latitude
    plt.tight_layout()
    plt.show()
    print_volcanoes_per_plate(volcanoes_gdf)

def print_volcanoes_per_plate(volcanoes_gdf):
    plate_volcano_count = {}
    for _, volcano in volcanoes_gdf.iterrows():
        plate = volcano['Tectonic_Plate']
        if plate in plate_volcano_count:
            plate_volcano_count[plate] += 1
        else:
            plate_volcano_count[plate] = 1
    
    print( 'Total volcanoes per plater: ' + str(plate_volcano_count))

if __name__ == "__main__":
    volcanoes_df, non_filtered_tectonic_plates = init_data()
    tectonic_plates_major = non_filtered_tectonic_plates[non_filtered_tectonic_plates['PlateName'].isin(MAJOR_TECTONIC_PLATES)]
    
    geometry = [Point(xy) for xy in zip(volcanoes_df['Longitude'], volcanoes_df['Latitude'])]
    
    plot_volcanoes_onto_plates(volcanoes_df, tectonic_plates_major, geometry)
    plot_volcanoes_onto_plates(volcanoes_df, non_filtered_tectonic_plates, geometry)
