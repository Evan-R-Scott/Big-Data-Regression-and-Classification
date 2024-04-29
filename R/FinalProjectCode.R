# load original, unmodified dataset
orig_df <- read.csv("Premier_League.csv")

# clean the dataset of any unnecessary columns and missing data
clean_df <- clean_data(orig_df)

clean_data <- function(orig_df) {
  clean_df <- subset(orig_df, select= -c(home_blocked, away_blocked, home_pass, away_pass,
                                         home_off, away_off, home_offside, away_offside,
                                         home_tackles, away_tackles, home_duels, away_duels,
                                         home_saves, away_saves, home_fouls, away_fouls, links))
  return (clean_df)
}
