# Vaximap paper

The paper will serve 3 purposes: 
- explain how and why vaximap was created, and how it could be used in the future 
- provide analysis of the vaccine rollout from the dataset 
- serve as the official citation for public release of the dataset itself (hopefully this boosts the number of citations)

## Paper outline 

I'll fill this in after contacting some journals. The emphasis will be on the clinical utility of the service and any insights from the dataset, we won't go into lots of detail about how the system works. 


## Conda environment 

To clone the development environment used for the main vaximap site, run the following command. This will create an environment called 'vaximap'. 
```
conda env create -f environment.yml
```
## Dataset 

The dataset contains all patients that were mapped up until Nov 2021 (327k). Judging from early experience, there may be duplicate entries in the dataset as users made multiple similar requests to familiarise themselves with the service (eg, with different cluster sizes). These have not been identified or removed. 

The demo notebook shows how to access the dataset (the requirements file should cover the libraries required). Each row in the pandas data frame represents the response to a single user request. The responses contain the following columns:

- `n_patients`: number of individual patients in request
- `n_clusters`: number of clusters the patients were sorted into 
- `latlong`: an array of size (N x 2), latitude and longitude coordinates for the individual patients. NB if multiple patients share the same postcode, they will have the same latlong coordinate. These coordiantes have been **shifted so they centre on the origin (0,0)** (for data protection reasons); this means that the absolute coordinates are meaningless, but the relative differences are correct. See below. 
- `clusters`: a list of lists. Each list contains index numbers (zero-indexed), sorted into the optimal order, into the rows of the latlong array. This is the optimal set of clusters that were calculated for the patients. 
- `region`: country code 
- `postcodes`: only around 10% of the latter rows have this data. This is a dict containing the number of patients per top-level postcode in the request (eg, OX1 4AU -> OX1, SE15 5AW -> SE15). These areas are large enough that the data is not identifying. 
- `created`: date of request 
- `mode`: transport mode (driving or walking, there may be some cycling as well)

### Calculating distances between patients 

Calculating distances between two points expressed as latitude and longitude requires correction for curvilinear coordinates (ie, near the poles, lines of longitude are closer together). See for example: https://www.nhc.noaa.gov/gccalc.shtml

Because the latlong data does not contain absolute coordinates, only relative differences, calculating true distances is impossible. Instead, thus far I've approximated this by assuming the midpoint of all UK patients corresponds to the mid-coordinate of the UK ~(53, -1.2). By adding this back into the latlong coordinates, effectively you shift the set of patient coordinates into the middle of the UK and then true distances can be calculated. In practice I hope this won't make a big difference (there are barely any patients at the top of Scotland, for example). 

## Impact analysis 

The word document explains the methodology behind the preliminary impat analysis I prepared back in Feb. We used feedback from users to help estimate time savings, and then NHS wage rates to convert time savings into cost savings. 