
# load original, unmodified dataset
orig_df <- read.csv("Premier_League.csv")

# set seed for reproducibility
set.seed(69)

# function calls to run for initial setup
clean_df <- clean_data(orig_df)
partition_data(clean_df)

# Regression Tasks - function calls to run
correlation_check(clean_df)

# clean the dataset of any unnecessary columns and fix data types
clean_data <- function(orig_df) {

  clean_df <- subset(orig_df, select= -c(home_blocked, away_blocked, home_pass, away_pass,
                                         home_off, away_off, home_offside, away_offside,
                                         home_tackles, away_tackles, home_duels, away_duels,
                                         home_saves, away_saves, home_fouls, away_fouls, links))

  # remove rows with Nan values (na.omit was not recognizing null values in dataset so solved this way)
  rowsToRemove <- c()
  # loop through rows
  for (i in 1:nrow(clean_df)) {
    # loop through columns
    for (j in 1:ncol(clean_df)) {
      # Check if value is Nan and if so, save row for later deletion
      if (is.na(clean_df[i, j]) || clean_df[i, j] == "Nan") {
        rowsToRemove <- c(rowsToRemove, i)
        break  # Nan was found so don't check row any further
      }
    }
  }
  # Remove rows with Nan values
  clean_df <- clean_df[-rowsToRemove, ]

  numericColumnNames <- c("attendance", "Goals.Home", "home_possessions", "home_shots")
  # ensure all numeric columns are of numeric type
  for (column in numericColumnNames) {
    clean_df[[column]] <- as.numeric(clean_df[[column]])
  }
  #clean_df$attendance <- as.numeric(clean_df$attendance)
  #clean_df$Goals.Home <- as.numeric(clean_df$Goals.Home)
  #clean_df$home_possessions <- as.numeric(clean_df$home_possessions)

  # "Home.Team" Change team names to numeric type so can use for correlation matrix

  return (clean_df)
}

# partition the dataset into 3 sets: training partition and 2 holdout partitions (validation and testing)
partition_data <- function(clean_df) {

  # get total number of data/rows
  totalRows <- nrow(clean_df)

  # calculate size of each of the 3 sets
  trainingSize <- floor(0.7 * totalRows)
  validationSize <- floor(0.15 * totalRows)
  testingSize <- totalRows - trainingSize - validationSize

  # shuffle indices of size(number of total rows) in dataset to randomize because organized by date
  randomizeData <- sample(totalRows)

  # set these randomized indices to one of the 3 partitioned set
  trainingIndices <- randomizeData[1:trainingSize]
  validationIndices <- randomizeData[(trainingSize + 1):(trainingSize + validationSize)]
  testingIndices <- randomizeData[(trainingSize + validationSize + 1):totalRows]

  # extract data at the assigned, randomized indices for the partitioned sets from the cleaned dataset
  trainingData <- clean_df[trainingIndices, ]
  validationData <- clean_df[validationIndices, ]
  testingData <- clean_df[testingIndices, ]

  # Write the partitioned sets to CSV files for further use
  write.csv(trainingData, "training_data.csv", row.names = FALSE)
  write.csv(validationData, "validation_data.csv", row.names = FALSE)
  write.csv(testingData, "testing_data.csv", row.names = FALSE)
}

correlation_check <- function(clean_df) {

  correlation_matrix <- cor(clean_df[, c("attendance", "Goals.Home", "home_possessions", "home_shots")])

  print(correlation_matrix)
}
