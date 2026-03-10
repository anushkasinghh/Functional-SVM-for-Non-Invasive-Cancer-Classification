import pandas as pd
from pathlib import Path

# path = "../ALLDataGross/healthyCohort"

def read_data(path):
    folder = Path(path)
    print(f"Folder exists: {folder.exists()}")
    print(f"Folder contents: {list(folder.glob('*'))}")
    print(f"DPT files found: {list(folder.glob('*.dpt'))}")

    dataframes = []

    for file in sorted(folder.glob("*.dpt")):
        # print(file)
        try:
            # whitespace flexible (tabs or spaces)
            df = pd.read_csv(file, sep=r"\s+", engine="python")
            if df.shape[1] != 2:
                df = pd.read_csv(file, sep=r",", engine="python")
            dataframes.append((file.stem, df))
            print(f"✅ Loaded {file.name} with {df.shape[1]} columns")
            df.columns = ["Wavenumber", "Intensity"]
        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")
    return dataframes


def create_combined_dataset(path, normVP, infoP):
    """
    Create a combined DataFrame with all spectra from all categories
    
    Parameters:
    path = {"allkg_path": path to allkgdata folder
    "blind_path": path to blinddata folder  
    "healthy_path": path to healthydata folder}

    normVP = [normVP_allkg, normVP_blind, normVP_path]
    
    infoP = [infoP_allkg, infoP_path, infoP_healthy]
    """
    
    # Load data from all categories
    allkg_data = read_data(path[0])
    blind_data = read_data(path[1])
    healthy_data = read_data(path[2])
    
    combined_records = []
    
    # Process allkgdata with metadata
    for i, (filename, df) in enumerate(allkg_data):
        combined_records.append({
            'patient_id': f"allkg_{i+1}",
            'original_filename': filename,
            'category': 'allkgdata',
            'normVP': normVP[0][i] if i < len(normVP[0]) else None,
            'infoP': infoP[0][i] if i < len(infoP[0]) else None,
            'wavenumber': df['Wavenumber'].values,
            'intensity': df['Intensity'].values,
            'spectrum_length': len(df)
        })
    
    # Process blinddata with metadata
    for i, (filename, df) in enumerate(blind_data):
        combined_records.append({
            'patient_id': f"blind_{i+1}",
            'original_filename': filename,
            'category': 'blinddata',
            'normVP': normVP[1][i] if i < len(normVP[1]) else None,
            'infoP': infoP[1][i] if i < len(infoP[1]) else None,
            'wavenumber': df['Wavenumber'].values,
            'intensity': df['Intensity'].values,
            'spectrum_length': len(df)
        })
    
    # Process healthydata with metadata
    for i, (filename, df) in enumerate(healthy_data):
        combined_records.append({
            'patient_id': f"healthy_{i+1}",
            'original_filename': filename,
            'category': 'healthydata',
            'normVP': normVP[2][i] if i < len(normVP[2]) else None,
            'infoP': infoP[2][i] if i < len(infoP[2]) else None,
            'wavenumber': df['Wavenumber'].values,
            'intensity': df['Intensity'].values,
            'spectrum_length': len(df)
        })
    
    combined_df = pd.DataFrame(combined_records)
    return combined_df

