## Author: Surabhi S Nath
## Description: This script implements mixed effects models.
## on mean expert and automated creativity ratings.
## Helper functions and code for plotting in utils/Utils.R
## Important things are saved in plots/ and model_fits/:
  ## tables written to model_fits/
  ## plots saved in plots/

# Imports

{
  library(lme4)
  library(ggplot2)
  library(dplyr)
  library(caret)
  library(lmerTest)
  source("utils/Utils.R")
  # Set seed 
  set.seed(20) # Seed set randomly for reproducibility
}

# Setup
{
  # Read data
  data <- read.csv("../../../csvs/all_product.csv")

  # Scale all variables in the data
  my_scale <- function(x) {
    as.numeric(scale(x))
  }

  data$pid <- factor(data$pid)
  data$creator <- factor(data$creator)
  data$subgroup <- factor(data$subgroup)
  data$stimuli <- factor(data$stimuli)

  data$line_thickness <- my_scale(data$line_thickness)
  data$ink_density <- my_scale(data$ink_density)
  data$ink_fraction_inside_stimulus <- my_scale(data$ink_fraction_inside_stimulus)
  data$number_of_components <- my_scale(data$number_of_components)
  data$number_of_lines <- my_scale(data$number_of_lines)
  data$inverted <- factor(data$inverted)
  data$used_stimulus <- my_scale(data$used_stimulus)

  data$dist_from_base_caption_gtelarge <- my_scale(data$dist_from_base_caption_gtelarge)
  data$dist_from_base_shape_clip <- my_scale(data$dist_from_base_shape_clip)
  data$text_10NN_gtelarge <- my_scale(data$text_10NN_gtelarge)
  data$image_10NN_clip <- my_scale(data$image_10NN_clip)
  data$inverse_category_frequency <- my_scale(data$inverse_category_frequency)
  data$inverse_cluster_frequency_gtelarge <- my_scale(data$inverse_cluster_frequency_gtelarge)
  data$category_hard_to_interpret <- factor(data$category_hard_to_interpret)
  data$gpt_hard_to_interpret <- factor(data$gpt_hard_to_interpret)

  data$mean_creativity <- my_scale(data$mean_creativity)
  data$mean_originality <- my_scale(data$mean_originality)
}

# stratified 3 fold cv:
{
    num_folds <- 3

    folds <- createFolds(data$subgroup, k = num_folds, list = TRUE)
    fold_data <- lapply(folds, function(fold_indices) {
      list(
        train = data[-fold_indices, ],  # Training set (all except fold_indices)
        test = data[fold_indices, ]     # Test set (only fold_indices)
      )
    })
    data_train_fold1 <- fold_data[[1]]$train
    data_test_fold1 <- fold_data[[1]]$test
    data_train_fold2 <- fold_data[[2]]$train
    data_test_fold2 <- fold_data[[2]]$test
    data_train_fold3 <- fold_data[[3]]$train
    data_test_fold3 <- fold_data[[3]]$test
}

############### MODELS ##############

models <- list(
  "1 + (1 | subgroup)",
  "ink_density + (1 | subgroup)",
  "ink_fraction_inside_stimulus + (1 | subgroup)",
  "number_of_components + (1 | subgroup)",
  "number_of_lines + (1 | subgroup)",
  "used_stimulus + (1 | subgroup)",
  "ink_density + ink_fraction_inside_stimulus + number_of_components + number_of_lines + used_stimulus + (1 | subgroup)",

  "dist_from_base_shape_clip + (1 | subgroup)",
  "text_10NN_gtelarge + (1 | subgroup)",
  "image_10NN_clip + (1 | subgroup)",
  "inverse_category_frequency + (1 | subgroup)",
  "inverse_cluster_frequency_gtelarge + (1 | subgroup)",
  "category_hard_to_interpret + (1 | subgroup)",
  "gpt_hard_to_interpret + (1 | subgroup)",
  "dist_from_base_shape_clip + text_10NN_gtelarge + image_10NN_clip + inverse_category_frequency + category_hard_to_interpret + (1 | subgroup)",
  "dist_from_base_shape_clip + text_10NN_gtelarge + image_10NN_clip + inverse_cluster_frequency_gtelarge + gpt_hard_to_interpret + (1 | subgroup)",

  "(dist_from_base_shape_clip + text_10NN_gtelarge) * (used_stimulus + category_hard_to_interpret) + (1 | subgroup)",
  "used_stimulus + category_hard_to_interpret + dist_from_base_shape_clip + text_10NN_gtelarge + (1 | subgroup)",
  "ink_density + dist_from_base_shape_clip + image_10NN_clip + (1 | subgroup)",
  "ink_density * (dist_from_base_shape_clip + image_10NN_clip) + (1 | subgroup)",
  "dist_from_base_shape_clip * (ink_density + image_10NN_clip) + (1 | subgroup)",

  "ink_density + dist_from_base_shape_clip + image_10NN_clip + text_10NN_gtelarge + used_stimulus + category_hard_to_interpret + (1 | subgroup)"
)


run_model_analysis <- function(target_variable, output_file, plot_file, coeff_file) {

  df <- data.frame(matrix(ncol = 17, nrow = 0, dimnames =
    list(NULL, c("Id", "model", "AIC", "BIC", "AIC/BIC Var",
    "Rsq train mean", "Rsq train var", "Rsq test mean", "Rsq test var",
    "Spearman train mean", "Spearman train var", "Spearman test mean", "Spearman test var",
    "RMSE train mean", "RMSE train var", "RMSE test mean", "RMSE test var"))))
  
  id <- 0
  for (formula in models) {
    id <- id + 1
    fullformula <- paste(target_variable, "~", formula)
    
    f1 <- lmer(fullformula, data = data_train_fold1, control = lmerControl(optimizer = "bobyqa"))
    f2 <- lmer(fullformula, data = data_train_fold2, control = lmerControl(optimizer = "bobyqa"))
    f3 <- lmer(fullformula, data = data_train_fold3, control = lmerControl(optimizer = "bobyqa"))
    
    metrics <- modelanalysis(target_variable, num_folds,
      list(f1, f2, f3), list(data_train_fold1, data_train_fold2, data_train_fold3),
      list(data_test_fold1, data_test_fold2, data_test_fold3),
      FALSE, FALSE, fullformula) # set second last param to TRUE for printing
    
    df[nrow(df) + 1, ] <- c(id, noquote(fullformula), metrics)
  }
  
  write.csv(df, output_file, row.names = FALSE)



  bestformula <- "ink_density + dist_from_base_shape_clip + image_10NN_clip + text_10NN_gtelarge + used_stimulus + category_hard_to_interpret + (1 | subgroup)"
  bestfullformula <- paste(target_variable, "~", bestformula)
  
  f1 <- lmer(bestfullformula, data = data_train_fold1, control = lmerControl(optimizer = "bobyqa"))
  f2 <- lmer(bestfullformula, data = data_train_fold2, control = lmerControl(optimizer = "bobyqa"))
  f3 <- lmer(bestfullformula, data = data_train_fold3, control = lmerControl(optimizer = "bobyqa"))
  
  # Plots data vs predictions on train and test data
  metrics <- modelanalysis(target_variable, num_folds,
    list(f1, f2, f3), list(data_train_fold1, data_train_fold2, data_train_fold3),
    list(data_test_fold1, data_test_fold2, data_test_fold3),
    FALSE, TRUE, bestfullformula)
  
  # Plots random effects
  f <- lmer(bestfullformula, data = data, control = lmerControl(optimizer = "bobyqa"))
  ggCaterpillar(ranef(f, condVar = TRUE))
  ggsave(plot_file)



  model_summary <- summary(f)
  coefficients <- fixef(f)
  standard_errors <- sqrt(diag(vcov(f)))
  variable_names <- rownames(summary(f)$coefficients)
  p_values <- coef(summary(f))[, "Pr(>|t|)"]
  signif_levels <- ifelse(p_values < 0.001, "***",
                          ifelse(p_values < 0.01, "**",
                                 ifelse(p_values < 0.05, "*",
                                        ifelse(p_values < 0.1, ".", "NS"))))
  results_df <- data.frame(Variable = variable_names,
                           Coefficients = coefficients,
                           StdError = standard_errors,
                           PValue = p_values,
                           Significance = signif_levels)
  write.csv(results_df, file = coeff_file, row.names = FALSE)
}

# Running for both cases
run_model_analysis("mean_creativity", "model_fits/models_expert.csv", "plots/random_effects_experts.pdf", "model_fits/coeff_experts.csv")
run_model_analysis("mean_originality", "model_fits/models_automated.csv", "plots/random_effects_automated.pdf", "model_fits/coeff_automated.csv")