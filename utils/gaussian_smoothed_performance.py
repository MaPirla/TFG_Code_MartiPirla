from scipy.ndimage import gaussian_filter1d

def calculate_euclidean_difference(vector_1, vector_2):
    """
    Returns the euclidean distance between the two datasets
    """
    difference = 0
    for idx in range(len(vector_1)):
        difference += (vector_2[idx]-vector_1[idx])**2
    return difference**(0.5)

def make_performance_data_agent(df):
    data = df[(~df["performance"].isna()) & (df["nte"]-1 == df["horizon"])]["performance"]
    
    return gaussian_filter1d(data, sigma=5)

def make_performance_data(df, subject):
    data = df[(df["subject"] == subject) & (df["nte"]-1 == df["horizon"])]["performance"]
    data.loc[data.isna()] = 0
    return gaussian_filter1d(data, sigma=5)