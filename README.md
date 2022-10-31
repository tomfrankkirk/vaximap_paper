# Vaximap paper

## Dataset 

The dataset contains all patients that were mapped up until Oct 2022 (430k). Judging from early experience, there may be duplicate entries in the dataset as users made multiple similar requests to familiarise themselves with the service (eg, with different cluster sizes). These have not been identified or removed. 

The file has been created using Python and Pandas.to_hdf(), and it is recommended to read the file also using Pandas. 

Each row in the pandas data frame represents the response to a single user request with the following columns: 

- `n_patients`: number of individual patients in request
- `n_clusters`: number of clusters the patients were sorted into 
- `latlong`: an array of size (N x 2), latitude and longitude coordinates for the individual patients. NB if multiple patients share the same postcode, they will have the same latlong coordinate. These coordiantes have been **shifted so they centre on the origin (0,0)** (for data protection reasons); this means that the absolute coordinates are meaningless, but the relative differences are correct. See below. 
- `clusters`: a list of lists. Each list contains index numbers (zero-indexed), sorted into the optimal order, into the rows of the latlong array. This is the optimal set of clusters that were calculated for the patients. 
- `region`: country code 
- `postcodes`: the latter rows have this data. This is a dict containing a histogram of patients in postal districts (eg, OX1 4AU -> OX1, SE15 5AW -> SE15). These areas are large enough that the data is not identifying. 
- `created`: date of request 
- `mode`: transport mode (driving or walking, there may be some cycling as well)

If you use this data, please cite the accompanying publication as well (DOI TBC). 