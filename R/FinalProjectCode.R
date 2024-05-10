# setup
library(caret)
library(lubridate)
library(dplyr)
library(glmnet)
library(mgcv)
library(ggplot2)

# load original, unmodified dataset
orig_df <- read.csv("Premier_League.csv")

# set seed for reproducibility
set.seed(69)

#--------------------------------------------------
# FUNCTION CALLS TO RUN FOR PROJECT AFTER ALL FUNCTIONS HAVE BEEN LOADED IN ENVIRONMENT

# function calls to run for initial setup for both regression and classification tasks
clean_df <- clean_data()
partition_data()

# load partitioned data files for further use
training_data <- read.csv("training_data.csv")
testing_data <- read.csv("testing_data.csv")
validation_data <- read.csv("validation_data.csv")

# Regression Tasks - function calls to run
# unlist from list to vector
correlation_check()
linearResult <- linear_model(); linearModel <- linearResult[1]; linearTrainingPred <- unlist(linearResult[2]); linearValidationPred <- unlist(linearResult[3]); linearTrainingRMSE <- unlist(linearResult[4]); linearValidationRMSE <- unlist(linearResult[5])
bivariateResult <- bivariate_model(); bivariateModel<- bivariateResult[1]; bivariateTrainingPred <- unlist(bivariateResult[2]); bivariateValidationPred <- unlist(bivariateResult[3]); bivariateTrainingRMSE <- unlist(bivariateResult[4]); bivariateValidationRMSE <- unlist(bivariateResult[5])
ridgeResult <- regularizeBV_model(); ridgeModel <- ridgeResult[1]; ridgeTrainingPred <- unlist(ridgeResult[2]); ridgeValidationPred <- unlist(ridgeResult[3]); ridgeTrainingRMSE <- unlist(ridgeResult[4]); ridgeValidationRMSE <- unlist(ridgeResult[5])
splineResult <- spline_model(); splineModel <- splineResult[1]; splineTrainingPred <- unlist(splineResult[2]); splineValidationPred <- unlist(splineResult[3]); splineTrainingRMSE <- unlist(splineResult[4]); splineValidationRMSE <- unlist(splineResult[5])
bivariate_plot(linearTrainingPred, linearValidationPred, bivariateTrainingPred,
               bivariateValidationPred, ridgeTrainingPred, ridgeValidationPred,
               splineTrainingPred, splineValidationPred, training_data, validation_data)
best_model <- performance_table(linearTrainingRMSE, linearValidationRMSE,
                  bivariateTrainingRMSE, bivariateValidationRMSE,
                  ridgeTrainingRMSE,  ridgeValidationRMSE,
                  splineTrainingRMSE, splineValidationRMSE,
                  linearModel, bivariateModel, ridgeModel, splineModel)
final_model_uncontaminated(best_model)
linear_multivariate_model()
regularizeMV_model()
nonlinearMVModel()


#--------------------------------------------------

# clean the dataset of any unnecessary columns and fix data types
clean_data <- function() {

  clean_df <- subset(orig_df, select= -c(stadium, home_blocked, away_blocked, home_pass, away_pass,
                                         home_off, away_off, home_offside, away_offside,
                                         home_tackles, away_tackles, home_duels, away_duels,
                                         home_saves, away_saves, home_fouls, away_fouls, home_yellow,
                                         away_yellow, home_red, away_red, links))

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
  clean_df$attendance <- gsub(",", "", clean_df$attendance)

  # vector containing all columns to be converted to numeric
  numericColumnNames <- c("attendance", "Goals.Home","Away.Goals", "home_possessions",
                          "away_possessions", "home_shots", "away_shots", "home_on", "away_on",
                          "home_chances", "away_chances", "home_corners", "away_corners")

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
  # map team to the position they placed in 22/23 Premier League season
  team_position_map <- setNames(standings$Position, standings$Team)

  # add columns associating numbers with teams to dataframe
  clean_df$numHome.Team <- ifelse(clean_df$Home.Team %in% names(team_position_map), team_position_map[clean_df$Home.Team], NA)
  clean_df$numAway.Team <- ifelse(clean_df$Away.Team %in% names(team_position_map), team_position_map[clean_df$Away.Team], NA)

  # add columns for date(1 = Monday and 7 = Sunday) and time (Time from Midnight)
  clean_df$date <- dmy(clean_df$date)
  clean_df$DayOfWeek <- wday(clean_df$date, label = FALSE, week_start = 1)
  clean_df$TimeNumeric <- hour(hm(clean_df$clock)) * 60 + minute(hm(clean_df$clock))

  clean_df <- subset(clean_df, select= -c(date, clock))
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

#--------------- Regression Tasks Below ---------------

# correlation matrix to verify at least one moderate to strong relationship
correlation_check <- function() {

  correlation_matrix <- cor(clean_df[, c("attendance", "numAway.Team", "numHome.Team", "Goals.Home", "home_possessions", "home_shots", "DayOfWeek", "TimeNumeric", "home_chances", "away_chances")])

  print(correlation_matrix)
}

# helper function for calculating error metrics (MAE, RMSE, R^2)
calculate_error_metrics <- function(model){

  # Predict on the training data
  trainingPrediction <- predict(model, newdata = training_data)

  # Calculate in-sample error metrics
  trainMAE <- mean(abs(trainingPrediction - training_data$attendance))
  trainRMSE <- sqrt(mean((trainingPrediction - training_data$attendance)^2))
  traingRSQUARED <- cor(trainingPrediction, training_data$attendance)^2

  # Predict on the validation data
  validationPrediction <- predict(model, newdata = validation_data)

  # Calculate out-of-sample error metrics
  validationMAE <- mean(abs(validationPrediction - validation_data$attendance))
  validationRMSE <- sqrt(mean((validationPrediction - validation_data$attendance)^2))
  validationRSQUARED <- cor(validationPrediction, validation_data$attendance)^2

  # display metric results
  cat("In-sample MAE:", trainMAE, "\n")
  cat("In-sample RMSE:", trainRMSE, "\n")
  cat("In-sample R-squared:", traingRSQUARED, "\n")
  cat("\n")
  cat("Out-of-sample MAE:", validationMAE, "\n")
  cat("Out-of-sample RMSE:", validationRMSE, "\n")
  cat("Out-of-sample R-squared:", validationRSQUARED, "\n")

  return(list(trainingPrediction, validationPrediction, trainRMSE, validationRMSE))
}

# simple linear model with no transformations
linear_model <- function(){

  simpleLM <- lm(attendance ~ numHome.Team, data = training_data)
  print(summary(simpleLM))

  print("Linear Model Performance - Error Metrics:")
  result <- calculate_error_metrics(simpleLM)

  return(list(simpleLM, result[[1]], result[[2]], result[[3]], result[[4]]))
}

# bivariate model with polynomial/quadratic transformation
bivariate_model <- function() {
  bivarModel <- lm(attendance ~ numHome.Team + I(numHome.Team^2), data = training_data)
  print(summary(bivarModel))

  print("Bivariate Model Performance - Error Metrics:")
  result <- calculate_error_metrics(bivarModel)

  return(list(bivarModel, result[[1]], result[[2]], result[[3]], result[[4]]))
}

regularizeBV_model <- function() {

  # creates quadratic features for required data
  training_data$numHomeTeamSquared <- training_data$numHome.Team^2
  validation_data$numHomeTeamSquared <- validation_data$numHome.Team^2
  write.csv(training_data, "training_data.csv", row.names = FALSE)
  write.csv(validation_data, "validation_data.csv", row.names = FALSE)
  training_data <- read.csv("training_data.csv")
  validation_data <- read.csv("validation_data.csv")

  xTrain <- as.matrix(training_data[, c("numHome.Team", "numHomeTeamSquared")])
  yTrain <- training_data$attendance
  xValidation <- as.matrix(validation_data[, c("numHome.Team", "numHomeTeamSquared")])
  yValidation <- validation_data$attendance

  ridgeCV <- cv.glmnet(xTrain, yTrain, alpha = 0)

  optimalLambda <- ridgeCV$lambda.min

  ridgeModel <- glmnet(xTrain, yTrain, alpha = 0, lambda = optimalLambda)

  ridgeTrainingPredictions <- as.vector(predict(ridgeModel, newx = xTrain))
  ridgeValidationPredictions <- as.vector(predict(ridgeModel, newx = xValidation))

  ridgeTrainingMAE <- mean(abs(ridgeTrainingPredictions - yTrain))
  ridgeTrainingRMSE <- sqrt(mean((ridgeTrainingPredictions - yTrain)^2))
  ridgeTrainingRSQUARED <- cor(ridgeTrainingPredictions, yTrain)^2

  ridgeValidationMAE <- mean(abs(ridgeValidationPredictions - yValidation))
  ridgeValidationRMSE <- sqrt(mean((ridgeValidationPredictions - yValidation)^2))
  ridgeValidationRSQUARED <- cor(ridgeValidationPredictions, yValidation)^2

  cat("Regularize Model Performance - Error Metrics:\n")
  cat("Optimal Lambda:", optimalLambda, "\n")
  cat("In-sample MAE:", ridgeTrainingMAE, "\n")
  cat("In-sample RMSE:", ridgeTrainingRMSE, "\n")
  cat("In-sample R-squared:", ridgeTrainingRSQUARED, "\n")
  cat("\n")
  cat("Out-of-sample MAE:", ridgeValidationMAE, "\n")
  cat("Out-of-sample RMSE:", ridgeValidationRMSE, "\n")
  cat("Out-of-sample R-squared:", ridgeValidationRSQUARED, "\n")

  return(list(ridgeModel, ridgeTrainingPredictions, ridgeValidationPredictions, ridgeTrainingRMSE, ridgeValidationRMSE))
}

spline_model <- function() {
  splineModel <- gam(attendance ~ s(numHome.Team), family = gaussian(), data = training_data)
  print(summary(splineModel))

  cat("\n")
  print("Spline Model Performance - Error Metrics:")
  result <- calculate_error_metrics(splineModel)

  return(list(splineModel, result[[1]], result[[2]], result[[3]], result[[4]]))
}

bivariate_plot <- function(linearTrainingPred, linearValidationPred,
                           bivariateTrainingPred,bivariateValidationPred,
                           ridgeTrainingPred, ridgeValidationPred,
                           splineTrainingPred, splineValidationPred,
                           training_actual, validation_actual) {

  df1 <- data.frame(x = training_data$numHome.Team, y = training_data$attendance)
  plot(df1$y~df1$x)
  df6 <- data.frame(x = validation_data$numHome.Team, y = validation_data$attendance)
  plot(df6$y~df6$x)

  df2 <- data.frame(x = df1$x, y = linearTrainingPred)
  df3 <- data.frame(x = df1$x, y = bivariateTrainingPred)
  df4 <- data.frame(x = df1$x, y = ridgeTrainingPred)
  df5 <- data.frame(x = df1$x, y = splineTrainingPred)
  df7 <- data.frame(x = df6$x, y = linearValidationPred)
  df8 <- data.frame(x = df6$x, y = bivariateValidationPred)
  df9 <- data.frame(x = df6$x, y = ridgeValidationPred)
  df10 <- data.frame(x = df6$x, y = splineValidationPred)

  ggplot(df2, aes(x, y)) +
    geom_line(aes(color = "Linear"), linewidth = 0.8, alpha = 0.9) +
    geom_line(data = df3, aes(x, y, color = "Bivariate"), linewidth = 0.8, alpha = 0.9, linetype = "dashed") +
    geom_line(data = df4, aes(x, y, color = "Ridge"), linewidth = 0.8, alpha = 0.9, linetype = "dashed") +
    geom_line(data = df5, aes(x, y, color = "Spline"), linewidth = 0.8, alpha = 0.9, linetype = "dashed") +
    geom_line(data = df7, aes(x, y, color = "Linear (Validation)"), linewidth = 0.8, alpha = 0.9, linetype = "dashed") +
    geom_line(data = df8, aes(x, y, color = "Bivariate (Validation)"), linewidth = 0.8, alpha = 0.9, linetype = "dashed") +
    geom_line(data = df9, aes(x, y, color = "Ridge (Validation)"), linewidth = 0.8, alpha = 0.9, linetype = "dashed") +
    geom_line(data = df10, aes(x, y, color = "Spline (Validation)"), linewidth = 0.8, alpha = 0.9, linetype = "dashed") +
    geom_point(data = df1, aes(x, y), color = "darkgray") +
    geom_point(data = df6, aes(x, y), color = "black") +
    labs(x = "Team Number", y = "Attendance", title = "Models Predictions vs. Actual [Training & Validation]") +
    scale_color_manual(name = "Models",
                       values = c("Linear" = "skyblue", "Bivariate" = "skyblue4", "Ridge" = "royalblue",
                                  "Spline" = "slateblue", "Linear (Validation)" = "orange",
                                  "Bivariate (Validation)" = "red", "Ridge (Validation)" = "red4",
                                  "Spline (Validation)" = "sienna2"))
}

performance_table <- function(linearTrainingRMSE, linearValidationRMSE,
                              bivariateTrainingRMSE, bivariateValidationRMSE,
                              ridgeTrainingRMSE,  ridgeValidationRMSE,
                              splineTrainingRMSE, splineValidationRMSE,
                              linearModel, bivariateModel, ridgeModel, splineModel) {
  models <- c("Linear", "Bivariate", "Ridge", "Spline")
  inSampleRMSE <- c(linearTrainingRMSE, bivariateTrainingRMSE, ridgeTrainingRMSE, splineTrainingRMSE)
  outSampleRMSE <- c(linearValidationRMSE, bivariateValidationRMSE, ridgeValidationRMSE, splineValidationRMSE)

  performanceTable <- data.frame(Model = models, InSampleRMSE = inSampleRMSE, OutSampleRMSE = outSampleRMSE)
  performanceTable

  lowestRMSE_index <- which.min(outSampleRMSE)
  bestModel <- models[lowestRMSE_index]

  return(bestModel)
}

final_model_uncontaminated <- function(final_model){
  if (final_model == "Linear") {
    model <- lm(attendance ~ numHome.Team, data = training_data)
  } else if (final_model == "Bivariate") {
    model <- lm(attendance ~ numHome.Team + I(numHome.Team^2), data = training_data)
  } else if (final_model == "Ridge") {
    xTrain <- as.matrix(training_data[, c("numHome.Team", "numHomeTeamSquared")])
    yTrain <- training_data$attendance
    ridgeCV <- cv.glmnet(xTrain, yTrain, alpha = 0)
    optimalLambda <- ridgeCV$lambda.min
    model <- glmnet(xTrain, yTrain, alpha = 0, lambda = optimalLambda)
  } else if (final_model == "Spline") {
    model <- gam(attendance ~ s(numHome.Team), family = gaussian(), data = training_data)
  }
  finalPredictions <- predict(model, newdata = testing_data)

  finalRMSE <- sqrt(mean((finalPredictions - testing_data$attendance)^2))
  cat("Best Performance -", final_model, "Model\n")
  cat("Uncontaminated Out-sample RMSE:", finalRMSE)
}

linear_multivariate_model <- function(){
  #linearMV <- lm(attendance ~ numHome.Team + Goals.Home + home_possessions + home_shots, data = training_data)
  linearMV <- lm(attendance ~ numHome.Team + DayOfWeek + TimeNumeric, data = training_data)
  print(summary(linearMV))

  print("Linear Multivariate Model Performance - Error Metrics:")
  result <- calculate_error_metrics(linearMV)
}

regularizeMV_model <- function() {
  X_train <- model.matrix(attendance ~ numHome.Team + DayOfWeek + TimeNumeric, data = training_data)
  y_train <- training_data$attendance
  X_valid <- model.matrix(attendance ~ numHome.Team + DayOfWeek + TimeNumeric, data = validation_data)
  y_valid <- validation_data$attendance

  regMV <- cv.glmnet(X_train, y_train, alpha = 0)

  optimalLambda <- regMV$lambda.min

  ridgeMVModel <- glmnet(X_train, y_train, alpha = 0, lambda = optimalLambda)

  regMVTrainingPredictions <- as.vector(predict(ridgeMVModel, newx = X_train))
  regMVValidationPredictions <- as.vector(predict(ridgeMVModel, newx = X_valid))

  regMVTrainingMAE <- mean(abs(regMVTrainingPredictions - y_train))
  regMVTrainingRMSE <- sqrt(mean((regMVTrainingPredictions - y_train)^2))
  regMVTrainingRSQUARED <- cor(regMVTrainingPredictions, y_train)^2

  regMVValidationMAE <- mean(abs(regMVValidationPredictions - y_valid))
  regMVValidationRMSE <- sqrt(mean((regMVValidationPredictions - y_valid)^2))
  regMVValidationRSQUARED <- cor(regMVValidationPredictions, y_valid)^2

  cat("Regularize Multivariate Model Performance - Error Metrics:\n")
  cat("Optimal Lambda:", optimalLambda, "\n")
  cat("In-sample MAE:", regMVTrainingMAE, "\n")
  cat("In-sample RMSE:", regMVTrainingRMSE, "\n")
  cat("In-sample R-squared:", regMVTrainingRSQUARED, "\n")
  cat("\n")
  cat("Out-of-sample MAE:", regMVValidationMAE, "\n")
  cat("Out-of-sample RMSE:", regMVValidationRMSE, "\n")
  cat("Out-of-sample R-squared:", regMVValidationRSQUARED, "\n")
}

nonlinearMVModel <- function() {
  polyTrain <- poly(training_data[, c("numHome.Team", "DayOfWeek", "TimeNumeric")], 2)
  polyValid <- poly(validation_data[, c("numHome.Team", "DayOfWeek", "TimeNumeric")], 2)
  trainActual <- training_data$attendance
  validationActual <- validation_data$attendance

  nonlinearMVModel <- lm(attendance ~ polyTrain, data = training_data)

  print(summary(nonlinearMVModel))

  nonlinearTrainingPredictions <- predict(nonlinearMVModel, newdata = as.data.frame(polyTrain))
  nonlinearValidationPredictions <- predict(nonlinearMVModel, newdata = as.data.frame(polyValid))

  nonlinearTrainingMAE <- mean(abs(nonlinearTrainingPredictions - trainActual))
  nonlinearTrainingRMSE <- sqrt(mean((nonlinearTrainingPredictions - trainActual)^2))
  nonlinearTrainingRSQUARED <- cor(nonlinearTrainingPredictions, trainActual)^2

  nonlinearValidationMAE <- mean(abs(nonlinearValidationPredictions - validationActual))
  nonlinearValidationRMSE <- sqrt(mean((nonlinearValidationPredictions - validationActual)^2))
  nonlinearValidationRSQUARED <- cor(nonlinearValidationPredictions, validationActual)^2

  cat("Nonlinear Multivariate Model Performance - Error Metrics:\n")
  cat("In-sample MAE:", nonlinearTrainingMAE, "\n")
  cat("In-sample RMSE:", nonlinearTrainingRMSE, "\n")
  cat("In-sample R-squared:", nonlinearTrainingRSQUARED, "\n")
  cat("\n")
  cat("Out-of-sample MAE:", nonlinearValidationMAE, "\n")
  cat("Out-of-sample RMSE:", nonlinearValidationRMSE, "\n")
  cat("Out-of-sample R-squared:", nonlinearValidationRSQUARED, "\n")

}

#--------------- Classification Tasks Below ---------------

