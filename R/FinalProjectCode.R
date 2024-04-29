# load original, unmodified dataset
orig_df <- read.csv("Premier_League.csv")

# set seed for reproducibility
set.seed(123)

# function calls to run
clean_df <- clean_data(orig_df)
partition_data(clean_df)

# clean the dataset of any unnecessary columns and missing data
clean_data <- function(orig_df) {
  clean_df <- subset(orig_df, select= -c(home_blocked, away_blocked, home_pass, away_pass,
                                         home_off, away_off, home_offside, away_offside,
                                         home_tackles, away_tackles, home_duels, away_duels,
                                         home_saves, away_saves, home_fouls, away_fouls, links))
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
