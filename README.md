# Labor-market-trends

## Overview
A comprehensive pipeline that collects labor market-related data from labor market platforms (work24, saramin, wanted) and leverages embedding techniques to analyze trends.
## 구조
```
Labor-market-trends/
├── data/                   # Data collection and preprocessing
├── datasets/               # Data storage
├── model/                  # Embedding and analysis
├── output/                 # Saving analysis results
└──  main.ipynb             # main notebook file
```
## Detailed pipeline
### 1. Data Collection Steps (`data/` folder)
#### 1.1 Data Collection Module
- Function: Data collection script for each platform
- Storage location: Collected data is saved in CSV format in `datasets/` folder
- Main configuration**:
  - Platform-specific collection script
  - Platform-specific API KEY issuance required

#### 1.3 Data Preprocessing Module
- Input: RAW data from the `datasets/` folder.
- **Processing**: Data cleaning, formatting, missing values handling, etc.
- Output: Save preprocessed data in `datasets/` folder

 **Maintenance requirements:
- Currently operating on previously collected RAW data.
- Requires scheduling and automation to be implemented for continuous data collection

### 2. Save data (`datasets/` folder)

#### 폴더 구조
```
datasets/            
├── work24.csv                    # raw data
├── saramin.csv                   # raw data
├── wanted.csv                    # raw data         
├── work24_processed.csv          # Preprocessed data
├── saramin_processed.csv         # Preprocessed data
└── wanted_processed.csv          # Preprocessed data
```
### 3. Modeling and Analysis (`model/` folder)

#### 3.1 Create an embedding
- Purpose: Convert job posting text data to vector space
- Output: Vector representation of each text -> save as .pkl file

#### 3.2 Embedding Map (Embedding Map)
- Purpose: Implement embedding topographic map of job postings in domestic job market
- Description: KDE visualization after UMAP dimension reduction
- **Use case**: Visualization of data patterns and relationships

#### 3.3 Clustering
- Purpose: Grouping data with similar characteristics
- **Result**: Identify groups by labor market trends

### 4. Executable file (`main.ipynb`)

### 5. Save results (`output/` folder)
#### Main output.
- Embedded topographic map: interactive HTML format
- Cluster analysis results: organized by group characteristics


📩 Should you have any questions, please contact us at the following email address: **kde9867@gmail.com !**
