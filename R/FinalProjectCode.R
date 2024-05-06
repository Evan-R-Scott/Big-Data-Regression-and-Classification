library(caret)
library(dplyr)
library(glmnet)

# load original, unmodified dataset
orig_df <- read.csv("Premier_League.csv")

# set seed for reproducibility
set.seed(69)

#--------------------------------------------------
# FUNCTION CALLS TO RUN FOR PROJECT AFTER ALL FUNCTIONS HAVE BEEN LOADED IN ENVIRONMENT

# function calls to run for initial setup
clean_df <- clean_data()
partition_data()

training_data <- read.csv("training_data.csv")
testing_data <- read.csv("testing_data.csv")
validation_data <- read.csv("validation_data.csv")

# Regression Tasks - function calls to run
correlation_check()
linear_model()
bivariate_model()
regularize_model()
#--------------------------------------------------

# clean the dataset of any unnecessary columns and fix data types
clean_data <- function() {

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
      # check if value is Nan and if so, save row for later deletion
      if (is.na(clean_df[i, j]) || clean_df[i, j] == "Nan") {
        rowsToRemove <- c(rowsToRemove, i)
        break  # Nan was found so don't check row any further
      }
    }
  }
  # remove rows with Nan values
  clean_df <- clean_df[-rowsToRemove, ]

  # convert commas to periods to convert attendance from string to numeric data type
  clean_df$attendance <- gsub(",", ".", clean_df$attendance)

  # vector containing all columns to be converted to numeric
  numericColumnNames <- c("attendance", "Goals.Home","Away.Goals", "home_possessions",
                          "away_possessions", "home_shots", "away_shots", "home_on", "away_on",
                          "home_chances", "away_chances", "home_corners", "away_corners",
                          "home_yellow", "away_yellow", "home_red", "away_red")

  # ensure all numeric columns are of numeric type
  for (column in numericColumnNames) {
    clean_df[[column]] <- as.numeric(clean_df[[column]])
  }

  # converts home and away team names to numeric types based on table standings for correlation test
  standings <- data.frame(
    Team = c("Manchester City", "Arsenal", "Manchester United", "Newcastle United", "Liverpool",
             "Brighton and Hove Albion", "Aston Villa", "Tottenham Hotspur", "Brentford", "Fulham", "Crystal Palace",
             "Chelsea", "Wolverhampton Wanderers", "West Ham United", "Bournemouth", "Nottingham Forest", "Everton",
             "Leicester City", "Leeds United", "Southampton"),
    Position = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
  )
  team_position_map <- setNames(standings$Position, standings$Team)

  clean_df$numHome.Team <- ifelse(clean_df$Home.Team %in% names(team_position_map), team_position_map[clean_df$Home.Team], NA)
  clean_df$numAway.Team <- ifelse(clean_df$Away.Team %in% names(team_position_map), team_position_map[clean_df$Away.Team], NA)

  return (clean_df)
}

# partition the dataset into 3 sets: training partition and 2 holdout partitions (validation and testing)
partition_data <- function() {

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

# correlation matrix to verify at least one moderate to strong relationship
correlation_check <- function() {

  correlation_matrix <- cor(clean_df[, c("attendance", "numAway.Team", "numHome.Team", "Goals.Home", "home_possessions", "home_shots")])

  print(correlation_matrix)
}

# helper function for calculating error metrics (MAE, RMSE, R^2)
calculate_error_metrics <- function(model){

  # Predict on the training data
  train_predictions <- predict(model, newdata = training_data)

  # Calculate in-sample error metrics
  train_mae <- mean(abs(train_predictions - training_data$attendance))
  train_rmse <- sqrt(mean((train_predictions - training_data$attendance)^2))
  train_r_squared <- cor(train_predictions, training_data$attendance)^2

  # Predict on the validation data
  validation_predictions <- predict(model, newdata = validation_data)

  # Calculate out-of-sample error metrics
  validation_mae <- mean(abs(validation_predictions - validation_data$attendance))
  validation_rmse <- sqrt(mean((validation_predictions - validation_data$attendance)^2))
  validation_r_squared <- cor(validation_predictions, validation_data$attendance)^2

  # display metric results
  cat("In-sample MAE:", train_mae, "\n")
  cat("In-sample RMSE:", train_rmse, "\n")
  cat("In-sample R-squared:", train_r_squared, "\n")
  cat("\n")
  cat("Out-of-sample MAE:", validation_mae, "\n")
  cat("Out-of-sample RMSE:", validation_rmse, "\n")
  cat("Out-of-sample R-squared:", validation_r_squared, "\n")
}

# simple linear model with no transformations
linear_model <- function(){

  simpleLM <- lm(attendance ~ numHome.Team, data = training_data)
  print(summary(simpleLM))

  print("Linear Model Performance - Error Metrics:")
  calculate_error_metrics(simpleLM)
}

# bivariate model with polynomial/quadratic transformation
bivariate_model <- function() {
  bivarModel <- lm(attendance ~ numHome.Team + I(numHome.Team^2), data = training_data)
  print(summary(bivarModel))

  print("Bivariate Model Performance - Error Metrics:")
  calculate_error_metrics(bivarModel)
}

regularize_model <- function() {

  # create quadratic features for required data
  training_data$numHomeTeamSquared <- training_data$numHome.Team^2
  validation_data$NumHomeTeamSquared <- validation_data$numHome.Team^2
}
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
