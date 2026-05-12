# ======================================================================
# Pizzagate - Predictive Models for Delivery Delays
# Team RAD | ITEC 460 | Spring 2026
# Author: Sultan
#
# Two-stage analysis:
#   Stage 1: Classification (Is Delayed yes/no) - Logistic + Random Forest
#   Stage 2: Regression (Delay severity in minutes) - Linear Regression
#
# How to run: open in RStudio, click Source (top-right). File paths
# are hardcoded to ~/Desktop/BI/Pizzagate/
# ======================================================================


# ----------------------------------------------------------------------
# 1. SETUP - install + load packages
# ----------------------------------------------------------------------
required <- c("tidyverse", "tidymodels", "ranger", "broom", "yardstick",
              "vip", "janitor")

new_pkgs <- required[!(required %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(ranger)
  library(broom)
  library(yardstick)
  library(vip)
  library(janitor)
})

set.seed(42)


# ----------------------------------------------------------------------
# 2. FILE PATHS
# ----------------------------------------------------------------------
INPUT_CSV  <- "~/Desktop/BI/Pizzagate/TeamRAD_pizza_clean.csv"
OUTPUT_DIR <- "~/Desktop/BI/Pizzagate"


# ----------------------------------------------------------------------
# 3. LOAD DATA
# ----------------------------------------------------------------------
raw <- read_csv(INPUT_CSV, show_col_types = FALSE) %>%
  clean_names()

cat("Dataset loaded:", nrow(raw), "rows x", ncol(raw), "columns\n\n")


# ----------------------------------------------------------------------
# STAGE 1 - CLASSIFICATION: Will this order be delayed? (yes / no)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 4. FEATURE SELECTION
# ----------------------------------------------------------------------
model_df <- raw %>%
  transmute(
    is_delayed        = factor(as.character(is_delayed),
                               levels = c("TRUE", "FALSE")),
    traffic_level     = factor(traffic_level),
    pizza_size        = factor(pizza_size),
    region            = factor(region),
    restaurant_name   = factor(restaurant_name),
    is_peak_hour      = factor(as.character(is_peak_hour)),
    is_weekend        = factor(as.character(is_weekend)),
    distance_km       = distance_km,
    pizza_complexity  = pizza_complexity,
    toppings_count    = toppings_count,
    order_hour        = order_hour
  )

cat("Class balance:\n")
print(model_df %>% count(is_delayed) %>%
        mutate(pct = round(100 * n / sum(n), 1)))
cat("\n")


# ----------------------------------------------------------------------
# 5. TRAIN/TEST SPLIT
# ----------------------------------------------------------------------
split  <- initial_split(model_df, prop = 0.8, strata = is_delayed)
train  <- training(split)
test   <- testing(split)

cat("Train:", nrow(train), "rows |  Test:", nrow(test), "rows\n\n")


# ----------------------------------------------------------------------
# 6. PREPROCESSING RECIPE
# ----------------------------------------------------------------------
rec <- recipe(is_delayed ~ ., data = train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())


# ----------------------------------------------------------------------
# 7. MODEL 1 - LOGISTIC REGRESSION
# ----------------------------------------------------------------------
lr_spec <- logistic_reg() %>% set_engine("glm")

lr_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(lr_spec)

lr_fit <- fit(lr_wf, data = train)


# ----------------------------------------------------------------------
# 8. MODEL 2 - RANDOM FOREST
# ----------------------------------------------------------------------
rf_spec <- rand_forest(trees = 500) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_spec)

rf_fit <- fit(rf_wf, data = train)


# ----------------------------------------------------------------------
# 9. EVALUATION
# ----------------------------------------------------------------------
eval_model <- function(fit_obj, name) {
  preds <- predict(fit_obj, test, type = "class") %>%
    bind_cols(predict(fit_obj, test, type = "prob")) %>%
    bind_cols(test %>% select(is_delayed))

  metrics <- bind_rows(
    accuracy(preds,  truth = is_delayed, estimate = .pred_class),
    precision(preds, truth = is_delayed, estimate = .pred_class),
    recall(preds,    truth = is_delayed, estimate = .pred_class),
    f_meas(preds,    truth = is_delayed, estimate = .pred_class),
    roc_auc(preds,   truth = is_delayed, .pred_TRUE)
  ) %>% mutate(model = name, .before = 1)

  list(preds = preds, metrics = metrics)
}

lr_eval <- eval_model(lr_fit, "Logistic Regression")
rf_eval <- eval_model(rf_fit, "Random Forest")

results <- bind_rows(lr_eval$metrics, rf_eval$metrics) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

cat("=== STAGE 1: CLASSIFICATION PERFORMANCE ===\n")
print(results)
cat("\n")


# ----------------------------------------------------------------------
# 10. CONFUSION MATRIX
# ----------------------------------------------------------------------
cm <- conf_mat(rf_eval$preds, truth = is_delayed, estimate = .pred_class)
cat("=== RANDOM FOREST CONFUSION MATRIX ===\n")
print(cm)
cat("\n")

cm_plot <- autoplot(cm, type = "heatmap") +
  scale_fill_gradient(low = "#F5F5F5", high = "#D32F2F") +
  labs(title = "Random Forest - Confusion Matrix",
       subtitle = paste0("Test set (n = ", nrow(test), ")")) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(OUTPUT_DIR, "TeamRAD_confusion_matrix.png"), cm_plot,
       width = 6, height = 5, dpi = 300, bg = "white")


# ----------------------------------------------------------------------
# 11. FEATURE IMPORTANCE - RANDOM FOREST
# ----------------------------------------------------------------------
imp_plot <- rf_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 15, geom = "col",
      aesthetics = list(fill = "#1976D2")) +
  labs(title = "Top Predictors of Pizza Delivery Delays",
       subtitle = "Random Forest - Permutation Importance",
       x = NULL, y = "Importance") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(OUTPUT_DIR, "TeamRAD_feature_importance.png"), imp_plot,
       width = 8, height = 6, dpi = 300, bg = "white")


# ----------------------------------------------------------------------
# 12. LOGISTIC REGRESSION COEFFICIENTS
# ----------------------------------------------------------------------
lr_coefs <- lr_fit %>%
  extract_fit_parsnip() %>%
  tidy(exponentiate = TRUE, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    direction = if_else(estimate > 1, "Increases delay odds",
                                      "Decreases delay odds"),
    across(c(estimate, conf.low, conf.high), ~ round(.x, 3))
  ) %>%
  arrange(desc(abs(log(estimate))))

cat("=== LOGISTIC REGRESSION - ODDS RATIOS (top features) ===\n")
print(lr_coefs %>% select(term, estimate, p.value, direction) %>% head(10))
cat("\n")


# ----------------------------------------------------------------------
# STAGE 2 - REGRESSION: Among delayed orders, how bad is the delay?
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 13. LINEAR REGRESSION ON DELAY SEVERITY
# ----------------------------------------------------------------------
delayed_df <- raw %>%
  filter(as.character(is_delayed) == "TRUE") %>%
  transmute(
    delay_min         = delay_min,
    traffic_level     = factor(traffic_level),
    pizza_size        = factor(pizza_size),
    region            = factor(region),
    restaurant_name   = factor(restaurant_name),
    is_peak_hour      = factor(as.character(is_peak_hour)),
    is_weekend        = factor(as.character(is_weekend)),
    distance_km       = distance_km,
    pizza_complexity  = pizza_complexity,
    toppings_count    = toppings_count,
    order_hour        = order_hour
  )

cat("=== STAGE 2: DELAY SEVERITY ANALYSIS ===\n")
cat("Subset: only delayed orders (n =", nrow(delayed_df), ")\n")
cat("Delay (min) range:", min(delayed_df$delay_min), "to",
    max(delayed_df$delay_min), "\n")
cat("Mean delay:", round(mean(delayed_df$delay_min), 2), "min\n\n")

split_lm  <- initial_split(delayed_df, prop = 0.8)
train_lm  <- training(split_lm)
test_lm   <- testing(split_lm)

rec_lm <- recipe(delay_min ~ ., data = train_lm) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

lm_spec <- linear_reg() %>% set_engine("lm")

lm_wf <- workflow() %>%
  add_recipe(rec_lm) %>%
  add_model(lm_spec)

lm_fit <- fit(lm_wf, data = train_lm)

lm_preds <- predict(lm_fit, test_lm) %>%
  bind_cols(test_lm %>% select(delay_min))

lm_metrics <- bind_rows(
  rmse(lm_preds, truth = delay_min, estimate = .pred),
  mae(lm_preds,  truth = delay_min, estimate = .pred),
  rsq(lm_preds,  truth = delay_min, estimate = .pred)
) %>%
  mutate(.estimate = round(.estimate, 3))

cat("=== LINEAR REGRESSION PERFORMANCE ===\n")
print(lm_metrics)
cat("\n")

lm_coefs <- lm_fit %>%
  extract_fit_parsnip() %>%
  tidy(conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    direction = if_else(estimate > 0, "Makes delay worse (+min)",
                                      "Makes delay shorter (-min)"),
    across(c(estimate, std.error, conf.low, conf.high), ~ round(.x, 3))
  ) %>%
  arrange(desc(abs(estimate)))

cat("=== LINEAR REGRESSION COEFFICIENTS (top features) ===\n")
cat("Interpretation: estimate = how many minutes each feature adds/subtracts\n")
cat("(features are normalized, so estimate = effect per 1 SD change)\n\n")
print(lm_coefs %>% select(term, estimate, p.value, direction) %>% head(10))
cat("\n")

lm_imp_plot <- lm_coefs %>%
  slice_head(n = 15) %>%
  mutate(term = fct_reorder(term, abs(estimate))) %>%
  ggplot(aes(x = abs(estimate), y = term, fill = direction)) +
  geom_col() +
  scale_fill_manual(values = c("Makes delay worse (+min)"   = "#D32F2F",
                               "Makes delay shorter (-min)" = "#388E3C")) +
  labs(title = "Top Predictors of Delay Severity",
       subtitle = "Linear Regression - Among delayed orders only",
       x = "Coefficient magnitude (normalized)", y = NULL, fill = NULL) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom")

ggsave(file.path(OUTPUT_DIR, "TeamRAD_severity_importance.png"), lm_imp_plot,
       width = 8, height = 6, dpi = 300, bg = "white")


# ----------------------------------------------------------------------
# 14. EXPORT ALL RESULTS TABLES
# ----------------------------------------------------------------------
write_csv(results,    file.path(OUTPUT_DIR, "TeamRAD_model_performance.csv"))
write_csv(lr_coefs,   file.path(OUTPUT_DIR, "TeamRAD_logreg_coefficients.csv"))
write_csv(lm_metrics, file.path(OUTPUT_DIR, "TeamRAD_linreg_performance.csv"))
write_csv(lm_coefs,   file.path(OUTPUT_DIR, "TeamRAD_linreg_coefficients.csv"))

cat("=== FILES WRITTEN TO ", OUTPUT_DIR, " ===\n", sep = "")
cat("  TeamRAD_feature_importance.png\n")
cat("  TeamRAD_confusion_matrix.png\n")
cat("  TeamRAD_severity_importance.png\n")
cat("  TeamRAD_model_performance.csv\n")
cat("  TeamRAD_logreg_coefficients.csv\n")
cat("  TeamRAD_linreg_performance.csv\n")
cat("  TeamRAD_linreg_coefficients.csv\n\n")


